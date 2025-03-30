# File: app/services/trade_service.py

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="ta.trend")

import asyncio
import math
import json
import gc
from datetime import datetime, timezone

import pandas as pd
import numpy as np
import ta

from pybit.unified_trading import HTTP
from structlog import get_logger

from app.core import Database, Config
from app.services.ml_service import MLService, calculate_dmi
from app.services.backfill_service import maybe_backfill_candles
from app.utils.cache import redis_client

logger = get_logger(__name__)

def enhanced_get_market_regime(df, adx_period=14, base_threshold=25):
    try:
        logger.debug("Starting enhanced_get_market_regime", original_df_rows=len(df))
        df_clean = df[["high", "low", "close"]].ffill().dropna()
        required_rows = adx_period * 2
        if len(df_clean) < required_rows:
            logger.warning("Not enough data for ADX calculation; defaulting regime to sideways",
                           needed=required_rows, have=len(df_clean))
            return "sideways"

        adx_indicator = ta.trend.ADXIndicator(
            high=df_clean["high"],
            low=df_clean["low"],
            close=df_clean["close"],
            window=adx_period
        )
        adx_series = adx_indicator.adx().dropna()
        latest_adx = adx_series.iloc[-1] if not adx_series.empty else 0

        df_clean['SMA20'] = df_clean['close'].rolling(window=20).mean()
        df_clean['SMA50'] = df_clean['close'].rolling(window=50).mean()
        sma_diff = df_clean['SMA20'].iloc[-1] - df_clean['SMA50'].iloc[-1]
        ma_trend = "up" if sma_diff > 0 else "down"

        atr_indicator = ta.volatility.AverageTrueRange(
            high=df_clean["high"],
            low=df_clean["low"],
            close=df_clean["close"],
            window=14
        )
        atr_series = atr_indicator.average_true_range().dropna()
        atr_percentile = atr_series.rank(pct=True).iloc[-1] if not atr_series.empty else 0

        adaptive_threshold = base_threshold * (1 - 0.2 * atr_percentile)

        boll = ta.volatility.BollingerBands(
            close=df_clean["close"],
            window=20,
            window_dev=2
        )
        bb_width = boll.bollinger_wband()
        bb_width_percentile = bb_width.rank(pct=True).iloc[-1] if not bb_width.empty else 0

        if latest_adx > adaptive_threshold:
            if ma_trend == "up" and df_clean['close'].iloc[-1] > df_clean['SMA20'].iloc[-1]:
                regime = "uptrending"
            elif ma_trend == "down" and df_clean['close'].iloc[-1] < df_clean['SMA20'].iloc[-1]:
                regime = "downtrending"
            else:
                regime = "sideways"
        else:
            regime = "sideways"

        if atr_percentile < 0.3 or bb_width_percentile < 0.3:
            regime = "sideways"

        logger.info("Enhanced market regime detection",
                    adx=float(latest_adx),
                    adaptive_threshold=float(adaptive_threshold),
                    ma_trend=ma_trend,
                    sma_diff=float(sma_diff),
                    atr_percentile=float(atr_percentile),
                    bb_width_percentile=float(bb_width_percentile),
                    regime=regime)
        return regime
    except Exception as e:
        logger.error("Error in enhanced market regime detection", error=str(e))
        return "sideways"

class TradeService:
    def __init__(self):
        # Ensure MLService is configured to set its actual feature lists.
        self.ml_service = MLService(lookback=60, ensemble_size=3, n_trend_models=2)
        self.session = None
        self.min_qty = None
        self.current_position = None
        self.current_order_qty = None
        self.last_trade_time = None
        self.running = False
        self.scaled_in = False

    async def initialize(self):
        try:
            self.session = HTTP(
                testnet=Config.BYBIT_CONFIG['testnet'],
                api_key=Config.BYBIT_CONFIG['api_key'],
                api_secret=Config.BYBIT_CONFIG['api_secret'],
                recv_window=5000
            )
            await self._get_min_order_qty()
            await self.ml_service.initialize()
            logger.info("Trade service initialized successfully")
        except Exception as e:
            logger.critical("Fatal error initializing trade service", error=str(e))
            raise
        asyncio.create_task(self.ml_service.schedule_daily_retrain())
        asyncio.create_task(self.update_open_trade_stops_task())
        asyncio.create_task(self.run_trading_logic())
        self.running = True

    async def _get_min_order_qty(self):
        if not self.min_qty:
            info = await asyncio.to_thread(
                self.session.get_instruments_info,
                category=Config.BYBIT_CONFIG.get('category', 'inverse'),
                symbol=Config.TRADING_CONFIG['symbol']
            )
            self.min_qty = float(info['result']['list'][0]['lotSizeFilter']['minOrderQty'])
            logger.info("Loaded min order qty", min_qty=self.min_qty)

    async def check_open_trade(self) -> bool:
        try:
            pos_data = await asyncio.to_thread(
                self.session.get_positions,
                category=Config.BYBIT_CONFIG.get('category', 'inverse'),
                symbol=Config.TRADING_CONFIG['symbol']
            )
            if pos_data.get("retCode", -1) != 0:
                logger.error("Failed to fetch positions", data=pos_data)
                return False
            positions = pos_data["result"].get("list", [])
            for pos in positions:
                if float(pos.get("size", "0")) != 0:
                    logger.info("Open position detected", size=pos.get("size"))
                    return True
            return False
        except Exception as e:
            logger.error("Error checking open trade", error=str(e))
            return False

    def compute_adaptive_stop_loss_and_risk(self, current_price: float, atr_value: float, signal: str,
                                            market_regime: str, df_features: dict, base_risk_multiplier=1.5) -> (float, float):
        raw_risk = base_risk_multiplier * abs(atr_value)
        risk = min(raw_risk, 0.10 * current_price)
        if signal.lower() == "buy":
            stop_loss = current_price - risk
        elif signal.lower() == "sell":
            stop_loss = current_price + risk
        else:
            stop_loss = current_price
        return stop_loss, risk

    def compute_sl_tp_dynamic(self, current_price: float, atr_value: float, signal: str,
                              market_regime: str, df_features: dict, base_risk_multiplier=1.5,
                              min_reward_risk_ratio=2.0) -> (float, float):
        stop_loss, risk = self.compute_adaptive_stop_loss_and_risk(
            current_price, atr_value, signal, market_regime, df_features, base_risk_multiplier
        )
        base_reward = 3.0
        if market_regime.lower() != "trending":
            base_reward *= 0.8
        dynamic_reward = max(base_reward, min_reward_risk_ratio)
        if signal.lower() == "buy":
            take_profit = current_price + risk * dynamic_reward
        elif signal.lower() == "sell":
            take_profit = current_price - risk * dynamic_reward
        else:
            take_profit = current_price
        return stop_loss, take_profit

    async def update_trade_stops(self, stop_loss: float, take_profit: float):
        try:
            pos_data = await asyncio.to_thread(
                self.session.get_positions,
                category=Config.BYBIT_CONFIG.get('category', 'inverse'),
                symbol=Config.TRADING_CONFIG['symbol']
            )
            positions = pos_data["result"].get("list", [])
            open_size = 0.0
            for pos in positions:
                size_val = float(pos.get("size", "0"))
                if size_val != 0:
                    open_size = size_val
                    break
            if open_size <= 0:
                logger.warning("No open position found; skipping stopLoss/takeProfit update.")
                return
            params = {
                "category": Config.BYBIT_CONFIG.get("category", "inverse"),
                "symbol": Config.TRADING_CONFIG['symbol'],
                "stopLoss": str(stop_loss),
                "takeProfit": str(take_profit),
                "positionIdx": 0
            }
            response = await asyncio.to_thread(self.session.set_trading_stop, **params)
            ret_code = response.get("retCode", -1)
            if ret_code == 0:
                logger.info("Trade stops updated", data=response)
            else:
                logger.error("Error updating trade stops", data=response)
        except Exception as e:
            logger.error("Exception in update_trade_stops", error=str(e))

    async def update_open_trade_stops_task(self):
        while self.running:
            if self.current_position is not None:
                try:
                    recent_df = await self.get_recent_candles()
                    logger.debug("recent_df shape in update_open_trade_stops_task",
                                 shape=recent_df.shape if not recent_df.empty else "(empty)")
                    recent_df = recent_df.asfreq('1min')
                    df_15 = recent_df.resample('15min').agg({
                        'open': 'first',
                        'high': 'max',
                        'low': 'min',
                        'close': 'last',
                        'volume': 'sum'
                    }).ffill().dropna()
                    logger.debug("df_15 shape in update_open_trade_stops_task", shape=df_15.shape)
                    if len(df_15) < 14:
                        logger.warning("Not enough 15-minute data for updating trade stops.")
                        await asyncio.sleep(60)
                        continue

                    df_15["rsi"] = ta.momentum.rsi(df_15["close"], window=14).ffill()
                    macd = ta.trend.MACD(df_15["close"], window_slow=26, window_fast=12, window_sign=9)
                    df_15["macd"] = macd.macd()
                    df_15["macd_diff"] = macd.macd_diff().ffill()
                    boll = ta.volatility.BollingerBands(close=df_15["close"], window=20, window_dev=2)
                    df_15["bb_high"] = boll.bollinger_hband()
                    df_15["bb_low"] = boll.bollinger_lband()
                    df_15["bb_mavg"] = boll.bollinger_mavg()
                    atr_indicator = ta.volatility.AverageTrueRange(
                        high=df_15["high"], low=df_15["low"], close=df_15["close"], window=14)
                    df_15["atr"] = atr_indicator.average_true_range()
                    last_candle = df_15.iloc[-1]
                    bb_width = boll.bollinger_wband()
                    bb_width_percentile = (bb_width.rank(pct=True).iloc[-1]) if not bb_width.empty else 0.5
                    features = {
                        "rsi": last_candle["rsi"],
                        "macd_diff": last_candle["macd_diff"],
                        "bb_width_percentile": bb_width_percentile
                    }
                    market_regime = enhanced_get_market_regime(df_15)
                    current_price = recent_df.iloc[-1]["close"]
                    stop_loss, take_profit = self.compute_sl_tp_dynamic(
                        current_price, df_15["atr"].iloc[-1], self.current_position, market_regime, features
                    )
                    await self.update_trade_stops(stop_loss, take_profit)
                    logger.info("Updated open trade stops", stop_loss=stop_loss, take_profit=take_profit)
                except Exception as e:
                    logger.error("Error updating open trade stops", error=str(e))
            await asyncio.sleep(60)

    async def run_trading_logic(self):
        while not (self.ml_service.trend_model_ready and self.ml_service.signal_model_ready):
            logger.info("ML models not ready, waiting before evaluating trade signal.")
            await asyncio.sleep(10)
        while self.running:
            try:
                df_1min = None
                for attempt in range(3):
                    df_1min = await self._get_recent_candles_with_retries()
                    if df_1min is not None and len(df_1min) >= 60:
                        break
                    logger.warning("1-min data invalid or too small, retrying", attempt=attempt)
                    await asyncio.sleep(2)
                if df_1min is None or len(df_1min) < 60:
                    logger.warning("No valid 1-min data; skipping cycle.")
                    await asyncio.sleep(60)
                    continue

                logger.debug("df_1min final shape in run_trading_logic", shape=df_1min.shape)
                df_1min = df_1min.asfreq('1min')
                df_15 = df_1min.resample('15min').agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }).ffill().dropna()
                logger.debug("df_15 shape in run_trading_logic", shape=df_15.shape,
                             min_ts=str(df_15.index.min()), max_ts=str(df_15.index.max()),
                             freq=str(df_15.index.inferred_freq))
                if len(df_15) < 14:
                    logger.warning("Not enough 15-minute candle data for ML prediction.")
                    await asyncio.sleep(60)
                    continue

                atr_indicator = ta.volatility.AverageTrueRange(
                    high=df_15["high"], low=df_15["low"], close=df_15["close"], window=14
                )
                df_15["atr"] = atr_indicator.average_true_range()
                df_15["rsi"] = ta.momentum.rsi(df_15["close"], window=14).ffill()
                macd = ta.trend.MACD(df_15["close"], window_slow=26, window_fast=12, window_sign=9)
                df_15["macd_diff"] = macd.macd_diff().ffill()

                market_regime = enhanced_get_market_regime(df_15)
                logger.info("Market regime detected", regime=market_regime)
                if market_regime.lower() == "sideways":
                    logger.info("Market is sideways. Skipping trade entry.")
                    await asyncio.sleep(60)
                    continue

                # Attempt to produce 4h data for trend
                df_4h = df_1min.resample("4h").agg({
                    "open": "first", "high": "max", "low": "min",
                    "close": "last", "volume": "sum"
                }).ffill()
                logger.debug("df_4h shape in run_trading_logic", shape=df_4h.shape,
                             min_ts=str(df_4h.index.min()), max_ts=str(df_4h.index.max()),
                             freq=str(df_4h.index.inferred_freq))
                # Build minimal 4h feature set
                df_4h["sma20"] = df_4h["close"].rolling(window=20, min_periods=1).mean()
                df_4h["sma50"] = df_4h["close"].rolling(window=50, min_periods=1).mean()
                df_4h["sma_diff"] = df_4h["sma20"] - df_4h["sma50"]
                df_4h["adx"] = ta.trend.ADXIndicator(
                    high=df_4h["high"], low=df_4h["low"],
                    close=df_4h["close"], window=14
                ).adx()
                try:
                    dmi, plus_di, minus_di = calculate_dmi(df_4h["high"], df_4h["low"], df_4h["close"], window=14)
                    df_4h["dmi_diff"] = plus_di - minus_di
                except Exception as e:
                    logger.warning("Custom DMI calculation failed in predict_trend", error=str(e))
                    df_4h["dmi_diff"] = 0.0
                df_4h["close_lag1"] = df_4h["close"].shift(1)
                df_4h.ffill(inplace=True)
                df_4h.dropna(inplace=True)

                # Minimal shape check
                if len(df_4h) < self.ml_service.lookback:
                    trend_prediction = "Hold"
                    logger.warning("Not enough 4h data for trend prediction; defaulting to 'Hold'")
                else:
                    last_slice = df_4h.tail(self.ml_service.lookback).copy()
                    from sklearn.preprocessing import RobustScaler
                    used_cols = ["sma_diff", "adx", "dmi_diff", "close_lag1"]
                    scaler = RobustScaler()
                    last_slice[used_cols] = scaler.fit_transform(last_slice[used_cols])
                    seq = last_slice[used_cols].values
                    logger.debug("trend seq shape before np.expand_dims", shape=seq.shape)
                    if seq.size == 0 or seq.shape[0] <= 0 or seq.shape[1] <= 0:
                        logger.error("Trend seq is empty/invalid; defaulting to 'Hold'", shape=seq.shape)
                        trend_prediction = "Hold"
                    else:
                        try:
                            seq = np.expand_dims(seq, axis=0)
                        except Exception as expand_err:
                            logger.error("np.expand_dims for trend seq failed", error=str(expand_err))
                            trend_prediction = "Hold"
                            seq = None
                        if seq is None or seq.size == 0:
                            trend_prediction = "Hold"
                            logger.warning("Trend seq is None or empty; returning 'Hold'")
                        else:
                            logger.debug("trend seq shape after expand_dims", shape=seq.shape)
                            # Use ensemble
                            if not self.ml_service.trend_models:
                                trend_prediction = "Hold"
                                logger.warning("No trend models available; returning 'Hold'")
                            else:
                                # Collect probabilities
                                probs_ensemble = []
                                for model in self.ml_service.trend_models:
                                    out = model.predict(seq)
                                    probs_ensemble.append(out)
                                avg_probs = np.mean(np.array(probs_ensemble), axis=0)
                                conf = float(np.max(avg_probs))
                                logger.debug("Trend model avg_probs", avg_probs=avg_probs.tolist(), conf=conf)
                                if conf < self.ml_service.TREND_PROB_THRESHOLD:
                                    logger.info("Trend prediction confidence too low", confidence=conf)
                                    trend_prediction = "Hold"
                                else:
                                    trend_class = np.argmax(avg_probs, axis=1)[0]
                                    trend_prediction = {
                                        0: "Uptrending",
                                        1: "Downtrending",
                                        2: "Sideways"
                                    }.get(trend_class, "Sideways")

                # get the short-term signal from the 15-min data
                signal_prediction = self.predict_signal(df_15)
                logger.info("Trend prediction", trend=trend_prediction)
                logger.info("15-minute signal prediction", signal=signal_prediction)

                if trend_prediction.lower() == "uptrending" and signal_prediction.lower() != "buy":
                    logger.info("Trend & signal not in confluence (expected Buy). Holding trade.")
                    await asyncio.sleep(60)
                    continue
                if trend_prediction.lower() == "downtrending" and signal_prediction.lower() != "sell":
                    logger.info("Trend & signal not in confluence (expected Sell). Holding trade.")
                    await asyncio.sleep(60)
                    continue
                signal = signal_prediction
                if await self.check_open_trade():
                    if self.current_position is not None and self.current_position.lower() != signal.lower():
                        logger.info("Signal reversal detected. Exiting current trade before new one.")
                        await self.exit_trade()
                        await asyncio.sleep(1)
                    else:
                        logger.info("Existing position matches new signal; skipping new trade.")
                        await asyncio.sleep(60)
                        continue
                else:
                    self.current_position = None
                    self.current_order_qty = None
                current_price = df_1min.iloc[-1]["close"]
                atr_value = df_15["atr"].iloc[-1]
                if atr_value <= 0:
                    logger.warning("ATR value non-positive, cannot compute position size.")
                    await asyncio.sleep(60)
                    continue
                stop_distance = abs(
                    current_price - self.compute_adaptive_stop_loss_and_risk(
                        current_price,
                        atr_value,
                        signal,
                        trend_prediction,
                        {"rsi": df_15["rsi"].iloc[-1], "macd_diff": df_15["macd_diff"].iloc[-1]}
                    )[0]
                )
                if stop_distance == 0:
                    logger.warning("Stop distance is 0; cannot compute position size.")
                    await asyncio.sleep(60)
                    continue
                portfolio_value = await self.get_portfolio_value()
                if portfolio_value <= 0:
                    logger.warning("Portfolio value is 0; cannot size position.")
                    await asyncio.sleep(60)
                    continue
                risk_per_trade = 0.01 * portfolio_value
                computed_qty = (risk_per_trade / stop_distance) * current_price
                order_qty = max(computed_qty, self.min_qty)
                order_qty = round(order_qty)
                logger.info("ATR-based position sizing computed",
                            portfolio_value=portfolio_value,
                            risk_per_trade=risk_per_trade,
                            stop_distance=stop_distance,
                            computed_qty=computed_qty,
                            final_qty=order_qty)
                fixed_leverage = 1
                await self.set_trade_leverage(fixed_leverage)
                boll_temp = ta.volatility.BollingerBands(close=df_15["close"], window=20, window_dev=2)
                bb_width = boll_temp.bollinger_wband()
                bb_width_percentile = (bb_width.rank(pct=True).iloc[-1]) if not bb_width.empty else 0.5
                features = {
                    "rsi": df_15["rsi"].iloc[-1],
                    "macd_diff": df_15["macd_diff"].iloc[-1],
                    "bb_width_percentile": bb_width_percentile
                }
                stop_loss, take_profit = self.compute_sl_tp_dynamic(
                    current_price, atr_value, signal, trend_prediction, features, base_risk_multiplier=1.5
                )
                await self.execute_trade(
                    side=signal,
                    qty=str(order_qty),
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    leverage=str(fixed_leverage)
                )
                self.last_trade_time = datetime.now(timezone.utc)
                self.current_position = signal.lower()
                self.current_order_qty = order_qty

            except Exception as e:
                logger.error("Error in trading logic", error=str(e))
            await asyncio.sleep(60)

    async def exit_trade(self):
        if self.current_position is None:
            logger.warning("No current position to exit.")
            return
        exit_side = "Sell" if self.current_position.lower() == "buy" else "Buy"
        order_qty = self.current_order_qty if self.current_order_qty is not None else self.min_qty
        logger.info("Exiting trade", exit_side=exit_side, qty=order_qty)
        await self.execute_trade(
            side=exit_side,
            qty=str(order_qty),
            stop_loss=None,
            take_profit=None,
            leverage="1"
        )
        self.current_position = None
        self.current_order_qty = None
        self.scaled_in = False

    async def scale_in_trade(self, side: str, additional_qty: int, delay: int = 60):
        await asyncio.sleep(delay)
        recent_df = await self.get_recent_candles()
        new_signal = self.predict_signal(recent_df)
        if new_signal.lower() == side.lower():
            logger.info("Scaling in additional position", side=side, additional_qty=additional_qty)
            await self.execute_trade(
                side=side,
                qty=str(additional_qty),
                stop_loss=None,
                take_profit=None,
                leverage="1"
            )
            self.scaled_in = True
        else:
            logger.info("Market conditions not favorable for scaling in")

    async def get_recent_candles(self):
        MIN_REQUIRED = 210
        cached = redis_client.get("recent_candles")
        if cached:
            data = json.loads(cached)
            df = pd.DataFrame(data)
            if len(df) < MIN_REQUIRED:
                rows = await Database.fetch("""
                    SELECT time, open, high, low, close, volume
                    FROM candles
                    ORDER BY time DESC
                    LIMIT 1800
                """)
                df = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close", "volume"])
                if not df.empty:
                    df['time'] = pd.to_datetime(df['time'])
                    df.sort_values("time", inplace=True)
                    df.set_index("time", inplace=True)
                return df
            df['time'] = pd.to_datetime(df['time'])
            df.sort_values("time", inplace=True)
            df.set_index('time', inplace=True)
            return df
        else:
            rows = await Database.fetch("""
                SELECT time, open, high, low, close, volume
                FROM candles
                ORDER BY time DESC
                LIMIT 1800
            """)
            df = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close", "volume"])
            if not df.empty:
                df['time'] = pd.to_datetime(df['time'])
                df.sort_values("time", inplace=True)
                df.set_index("time", inplace=True)
            return df

    async def _get_recent_candles_with_retries(self):
        for attempt in range(2):
            df = await self.get_recent_candles()
            if df is not None and not df.empty:
                return df
            logger.warning("get_recent_candles returned empty, attempt", attempt=attempt)
            await asyncio.sleep(1)
        return None

    async def get_portfolio_value(self) -> float:
        try:
            balance_data = await asyncio.to_thread(
                self.session.get_wallet_balance,
                accountType="UNIFIED"
            )
            logger.info("Wallet balance data", data=balance_data)
            total_balance = 0.0
            account_list = balance_data["result"].get("list", [])
            if not account_list:
                return 0.0
            coins = account_list[0].get("coin", [])
            for coin_data in coins:
                if coin_data["coin"] == "BTC":
                    usd_val = float(coin_data.get("usdValue", 0.0))
                    total_balance += usd_val
            return total_balance
        except Exception as e:
            logger.error("Failed to fetch portfolio value", error=str(e))
            return 0.0

    async def _log_trade(self, side: str, qty: str):
        if Database._pool is None:
            logger.warning("Database is closed; skipping trade logging.")
            return
        try:
            await Database.execute('''
                INSERT INTO trades (time, side, price, quantity)
                VALUES ($1, $2, $3, $4)
            ''', datetime.now(timezone.utc), side, 0, float(qty))
        except Exception as e:
            logger.error("Failed to log trade", error=str(e))

    async def set_trade_leverage(self, leverage: int):
        if Config.BYBIT_CONFIG.get("category", "inverse") == "inverse":
            logger.info("Skipping set_leverage for inverse contracts")
            return
        try:
            response = await asyncio.to_thread(
                self.session.set_leverage,
                category=Config.BYBIT_CONFIG.get("category", "inverse"),
                symbol=Config.TRADING_CONFIG['symbol'],
                leverage=str(leverage)
            )
            logger.info("Set leverage response", data=response)
        except Exception as e:
            logger.error("Error setting leverage", error=str(e))

    async def execute_trade(self, side: str, qty: str,
                            stop_loss: float = None,
                            take_profit: float = None,
                            leverage: str = None):
        if not self.running:
            return
        if float(qty) < self.min_qty:
            logger.warning("Computed position size below minimum; adjusting to minimum",
                           required=self.min_qty, actual=qty)
            qty = str(self.min_qty)
        logger.info("Placing trade",
                    side=side,
                    size=qty,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    leverage=leverage)
        order_params = {
            "category": Config.BYBIT_CONFIG.get("category", "inverse"),
            "symbol": Config.TRADING_CONFIG['symbol'],
            "side": side,
            "orderType": "Market",
            "qty": qty,
            "stopLoss": str(stop_loss) if stop_loss is not None else None,
            "takeProfit": str(take_profit) if take_profit is not None else None,
            "leverage": leverage if leverage is not None else None
        }
        order_params = {k: v for k, v in order_params.items() if v is not None}
        try:
            response = await asyncio.to_thread(self.session.place_order, **order_params)
            if response['retCode'] == 0:
                await self._log_trade(side, qty)
                logger.info("Trade executed successfully", order_id=response['result']['orderId'])
            else:
                error_msg = response.get('retMsg', 'Unknown error')
                logger.error("Trade failed", error=error_msg)
        except Exception as e:
            logger.error("Trade execution error", error=str(e))
            self.current_position = None

    def predict_signal(self, recent_data: pd.DataFrame) -> str:
        if not self.ml_service.signal_models or not self.ml_service.signal_model_ready:
            logger.warning("No signal model ensemble available or not ready; returning 'Hold'")
            return "Hold"

        # minimal shape check
        if recent_data is None or recent_data.empty:
            logger.warning("predict_signal: recent_data is empty; returning 'Hold'")
            return "Hold"

        from sklearn.preprocessing import RobustScaler
        # We do a basic 15min resample to match the approach in the ml_service
        df_15 = recent_data.asfreq("1min").resample("15min").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum"
        }).ffill().dropna()
        if len(df_15) < self.ml_service.lookback:
            logger.warning("predict_signal: Not enough bars in df_15 for inference; returning 'Hold'",
                           have=len(df_15), needed=self.ml_service.lookback)
            return "Hold"

        used_cols = self.ml_service.actual_signal_cols or self.ml_service.signal_feature_cols
        # if used_cols is empty, skip
        if not used_cols:
            logger.warning("predict_signal: no actual_signal_cols available; returning 'Hold'")
            return "Hold"

        # minimal approach: create feature columns if missing
        for c in used_cols:
            if c not in df_15.columns:
                df_15[c] = 0.0

        # scale them
        scaler = RobustScaler()
        df_15[used_cols] = scaler.fit_transform(df_15[used_cols])
        # build last slice
        last_slice = df_15.tail(self.ml_service.lookback).copy()
        if len(last_slice) < self.ml_service.lookback:
            logger.warning("predict_signal: last_slice too short; returning 'Hold'",
                           have=len(last_slice), needed=self.ml_service.lookback)
            return "Hold"
        seq = last_slice[used_cols].values
        logger.debug("predict_signal seq shape before expand_dims", shape=seq.shape)
        if seq.size == 0 or seq.shape[0] <= 0 or seq.shape[1] <= 0:
            logger.error("predict_signal seq invalid shape; returning 'Hold'", shape=seq.shape)
            return "Hold"
        try:
            seq = np.expand_dims(seq, axis=0)
        except Exception as exp_err:
            logger.error("np.expand_dims in predict_signal failed", error=str(exp_err))
            return "Hold"
        logger.debug("predict_signal seq shape after expand_dims", shape=seq.shape)

        # do ensemble predictions
        try:
            preds = [model.predict(seq) for model in self.ml_service.signal_models]
        except Exception as e:
            logger.error("Signal model predict failed", error=str(e), seq_shape=seq.shape)
            return "Hold"
        avg_pred = np.mean(np.array(preds), axis=0)
        max_conf = float(np.max(avg_pred))
        logger.debug("predict_signal avg_pred", avg_pred=avg_pred.tolist(), confidence=max_conf)
        if max_conf < SIGNAL_PROB_THRESHOLD:
            logger.info("Signal prediction confidence too low", probability=max_conf)
            return "Hold"
        signal_class = np.argmax(avg_pred, axis=1)[0]
        signal_label = {0: "Sell", 1: "Buy", 2: "Hold"}.get(signal_class, "Hold")
        return signal_label

async def periodic_gc(interval_seconds: int = 300):
    while True:
        collected = gc.collect()
        logger.info("Garbage collection complete", objects_collected=collected)
        await asyncio.sleep(interval_seconds)

if __name__ == "__main__":
    async def main():
        asyncio.create_task(periodic_gc(300))
        trade_service = TradeService()
        await trade_service.initialize()
        while True:
            await asyncio.sleep(60)
    asyncio.run(main())
