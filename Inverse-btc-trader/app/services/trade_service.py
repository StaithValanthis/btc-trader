# File: app/services/trade_service.py

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="ta.trend")

import asyncio
import math
from pybit.unified_trading import HTTP
from structlog import get_logger
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import ta
import json

from app.core import Database, Config
from app.services.ml_service import MLService
from app.services.backfill_service import maybe_backfill_candles
from app.utils.cache import redis_client

logger = get_logger(__name__)

def enhanced_get_market_regime(df, adx_period=14, threshold=25):
    try:
        df_clean = df[["high", "low", "close"]].ffill().dropna()
        required_rows = adx_period * 2
        if len(df_clean) < required_rows:
            logger.warning("Not enough data for ADX calculation; defaulting regime to sideways")
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
        ma_trend = "up" if df_clean['SMA20'].iloc[-1] > df_clean['SMA50'].iloc[-1] else "down"

        atr_indicator = ta.volatility.AverageTrueRange(
            high=df_clean["high"],
            low=df_clean["low"],
            close=df_clean["close"],
            window=14
        )
        atr_series = atr_indicator.average_true_range().dropna()
        atr_percentile = (atr_series.rank(pct=True).iloc[-1]) if not atr_series.empty else 0

        boll = ta.volatility.BollingerBands(
            close=df_clean["close"],
            window=20,
            window_dev=2
        )
        bb_width = boll.bollinger_wband()
        bb_width_percentile = (bb_width.rank(pct=True).iloc[-1]) if not bb_width.empty else 0

        if latest_adx > threshold:
            if ma_trend == "up" and df_clean['close'].iloc[-1] > df_clean['SMA20'].iloc[-1]:
                regime = "uptrending"
            elif ma_trend == "down" and df_clean['close'].iloc[-1] < df_clean['SMA20'].iloc[-1]:
                regime = "downtrending"
            else:
                regime = "sideways"
        else:
            regime = "sideways"
        if regime != "sideways" and (atr_percentile <= 0.5 or bb_width_percentile <= 0.5):
            regime = "sideways"
        logger.info("Enhanced market regime detection",
                    adx=latest_adx,
                    ma_trend=ma_trend,
                    atr_percentile=atr_percentile,
                    bb_width_percentile=bb_width_percentile,
                    regime=regime)
        return regime
    except Exception as e:
        logger.error("Error in enhanced market regime detection", error=str(e))
        return "sideways"

class TradeService:
    def __init__(self):
        # Using a 60-minute lookback for ML predictions
        self.ml_service = MLService(lookback=60)
        self.session = None
        self.min_qty = None            # Minimum order size (in contracts)
        self.current_position = None   # "buy" or "sell"
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

        # Start scheduled tasks: ML retraining, updating stops, and trading logic.
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

    def calculate_best_leverage(self, current_price: float, atr_value: float) -> float:
        vol_ratio = abs(atr_value) / current_price
        if vol_ratio >= 0.02:
            leverage = 1.0
        elif vol_ratio <= 0.005:
            leverage = 20.0
        else:
            leverage = 20.0 - ((vol_ratio - 0.005) / 0.015) * (20.0 - 1.0)
        return round(leverage, 2)

    def compute_dynamic_reward_ratio(self, current_price: float, atr_value: float, signal: str,
                                     df_features: dict, market_regime: str) -> float:
        base_reward = 3.0
        dynamic_reward = base_reward if market_regime == "trending" else base_reward * 0.8
        rsi = df_features.get("rsi", 50)
        macd_diff = df_features.get("macd_diff", 0)
        bb_width_percentile = df_features.get("bb_width_percentile", 0.5)
        if signal.lower() == "buy":
            dynamic_reward *= 0.9 if rsi > 70 else 1.1 if rsi < 30 else 1.0
            dynamic_reward *= 1.05 if macd_diff > 0 else 0.95 if macd_diff < 0 else 1.0
        elif signal.lower() == "sell":
            dynamic_reward *= 0.9 if rsi < 30 else 1.1 if rsi > 70 else 1.0
            dynamic_reward *= 1.05 if macd_diff < 0 else 0.95 if macd_diff > 0 else 1.0
        dynamic_reward *= (1 + 0.2 * (bb_width_percentile - 0.5))
        return max(dynamic_reward, 1.0)

    def compute_adaptive_stop_loss_and_risk(self, current_price: float, atr_value: float, signal: str,
                                            market_regime: str, df_features: dict, base_risk_multiplier=1.5):
        risk = base_risk_multiplier * abs(atr_value)
        rsi = df_features.get("rsi", 50)
        if signal.lower() == "buy":
            risk *= 0.9 if rsi > 70 else 1.1 if rsi < 30 else 1.0
        elif signal.lower() == "sell":
            risk *= 0.9 if rsi < 30 else 1.1 if rsi > 70 else 1.0
        macd_diff = df_features.get("macd_diff", 0)
        if signal.lower() == "buy":
            risk *= 1.05 if macd_diff > 0 else 0.95 if macd_diff < 0 else 1.0
        elif signal.lower() == "sell":
            risk *= 1.05 if macd_diff < 0 else 0.95 if macd_diff > 0 else 1.0
        if market_regime != "trending":
            risk *= 0.8
        stop_loss = current_price - risk if signal.lower() == "buy" else current_price + risk if signal.lower() == "sell" else None
        return stop_loss, risk

    def compute_sl_tp_dynamic(self, current_price: float, atr_value: float, signal: str,
                              market_regime: str, df_features: dict, base_risk_multiplier=1.5):
        stop_loss, risk = self.compute_adaptive_stop_loss_and_risk(current_price, atr_value, signal, market_regime, df_features, base_risk_multiplier)
        dynamic_reward = self.compute_dynamic_reward_ratio(current_price, atr_value, signal, df_features, market_regime)
        take_profit = current_price + risk * dynamic_reward if signal.lower() == "buy" else current_price - risk * dynamic_reward if signal.lower() == "sell" else None
        return stop_loss, take_profit

    async def update_trade_stops(self, stop_loss: float, take_profit: float):
        params = {
            "category": Config.BYBIT_CONFIG.get("category", "inverse"),
            "symbol": Config.TRADING_CONFIG['symbol'],
            "stopLoss": str(stop_loss),
            "takeProfit": str(take_profit),
            "positionIdx": 0
        }
        try:
            response = await asyncio.to_thread(self.session.set_trading_stop, **params)
            logger.info("Trade stops updated", data=response)
        except Exception as e:
            logger.error("Error updating trade stops", error=str(e))

    async def update_open_trade_stops_task(self):
        while self.running:
            if self.current_position is not None:
                try:
                    recent_df = await self.get_recent_candles()
                    recent_df = recent_df.asfreq('1min')
                    df_15 = recent_df.resample('15min').agg({
                        'open': 'first',
                        'high': 'max',
                        'low': 'min',
                        'close': 'last',
                        'volume': 'sum'
                    }).ffill().dropna()
                    if len(df_15) < 14:
                        logger.warning("Not enough 15-minute data for updating trade stops.")
                        await asyncio.sleep(60)
                        continue

                    df_15["rsi"] = ta.momentum.rsi(df_15["close"], window=14)
                    macd = ta.trend.MACD(close=df_15["close"], window_slow=26, window_fast=12, window_sign=9)
                    df_15["macd"] = macd.macd()
                    df_15["macd_diff"] = macd.macd_diff()
                    boll = ta.volatility.BollingerBands(close=df_15["close"], window=20, window_dev=2)
                    df_15["bb_high"] = boll.bollinger_hband()
                    df_15["bb_low"] = boll.bollinger_lband()
                    df_15["bb_mavg"] = boll.bollinger_mavg()
                    atr_indicator = ta.volatility.AverageTrueRange(
                        high=df_15["high"], low=df_15["low"], close=df_15["close"], window=14)
                    df_15["atr"] = atr_indicator.average_true_range()
                    last_candle = df_15.iloc[-1]
                    boll_temp = ta.volatility.BollingerBands(close=df_15["close"], window=20, window_dev=2)
                    bb_width = boll_temp.bollinger_wband()
                    bb_width_percentile = (bb_width.rank(pct=True).iloc[-1]) if not bb_width.empty else 0.5
                    features = {
                        "rsi": last_candle["rsi"],
                        "macd_diff": last_candle["macd_diff"],
                        "bb_width_percentile": bb_width_percentile
                    }
                    market_regime = enhanced_get_market_regime(df_15)
                    current_price = recent_df.iloc[-1]["close"]
                    stop_loss, take_profit = self.compute_sl_tp_dynamic(
                        current_price, df_15["atr"].iloc[-1], self.current_position, market_regime, features, base_risk_multiplier=1.5
                    )
                    await self.update_trade_stops(stop_loss, take_profit)
                    logger.info("Updated open trade stops", stop_loss=stop_loss, take_profit=take_profit)
                except Exception as e:
                    logger.error("Error updating open trade stops", error=str(e))
            await asyncio.sleep(60)

    async def run_trading_logic(self):
        # Wait until at least one ML model is ready.
        while not (self.ml_service.signal_model_ready or self.ml_service.multi_task_model is not None):
            logger.info("ML models not ready, waiting before evaluating trade signal.")
            await asyncio.sleep(10)
        # Main trading loop.
        while self.running:
            try:
                # Retrieve recent 1-min candle data (cached or from DB)
                cached = redis_client.get("recent_candles")
                if cached:
                    data = json.loads(cached)
                    df_1min = pd.DataFrame(data)
                    df_1min['time'] = pd.to_datetime(df_1min['time'])
                    df_1min.sort_values("time", inplace=True)
                    df_1min.set_index("time", inplace=True)
                else:
                    rows = await Database.fetch("""
                        SELECT time, open, high, low, close, volume
                        FROM candles
                        ORDER BY time DESC
                        LIMIT 1800
                    """)
                    if not rows or len(rows) < 60:
                        logger.warning("Not enough 1-minute candle data for prediction yet.")
                        await asyncio.sleep(60)
                        continue
                    df_1min = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close", "volume"])
                    df_1min['time'] = pd.to_datetime(df_1min['time'])
                    df_1min.sort_values("time", inplace=True)
                    df_1min.set_index("time", inplace=True)
                    redis_client.setex("recent_candles", 60, df_1min.reset_index().to_json(orient="records"))

                df_1min = df_1min.asfreq('1min')
                df_15 = df_1min.resample('15min').agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }).ffill().dropna()

                if len(df_15) < 14:
                    logger.warning("Not enough 15-minute candle data for ML prediction.")
                    await asyncio.sleep(60)
                    continue

                df_15["rsi"] = ta.momentum.rsi(df_15["close"], window=14)
                macd = ta.trend.MACD(close=df_15["close"], window_slow=26, window_fast=12, window_sign=9)
                df_15["macd"] = macd.macd()
                df_15["macd_diff"] = macd.macd_diff()
                boll = ta.volatility.BollingerBands(close=df_15["close"], window=20, window_dev=2)
                df_15["bb_high"] = boll.bollinger_hband()
                df_15["bb_low"] = boll.bollinger_lband()
                df_15["bb_mavg"] = boll.bollinger_mavg()
                atr_indicator = ta.volatility.AverageTrueRange(
                    high=df_15["high"],
                    low=df_15["low"],
                    close=df_15["close"],
                    window=14
                )
                df_15["atr"] = atr_indicator.average_true_range()
                df_15.dropna(subset=["close", "atr", "rsi"], inplace=True)
                if len(df_15) < 14:
                    logger.warning("Not enough 15-minute candle data after cleaning.")
                    await asyncio.sleep(60)
                    continue

                market_regime = enhanced_get_market_regime(df_15)
                logger.info("Market regime detected", regime=market_regime)

                signal = self.ml_service.predict_signal(df_15)
                logger.info("15-minute ML prediction", signal=signal)
                if signal.lower() == "hold":
                    await asyncio.sleep(60)
                    continue

                last_candle = df_15.iloc[-1]
                pivot = (last_candle["high"] + last_candle["low"] + last_candle["close"]) / 3
                resistance = pivot + (last_candle["high"] - pivot)
                support = pivot - (pivot - last_candle["low"])
                buffer_pct = 0.01
                resistance_threshold = last_candle["close"] * (1 + buffer_pct)
                support_threshold = last_candle["close"] * (1 - buffer_pct)
                if signal.lower() == "buy" and last_candle["close"] > resistance_threshold:
                    logger.info("15-minute price near resistance; skipping Buy trade.")
                    await asyncio.sleep(60)
                    continue
                if signal.lower() == "sell" and last_candle["close"] < support_threshold:
                    logger.info("15-minute price near support; skipping Sell trade.")
                    await asyncio.sleep(60)
                    continue

                if await self.check_open_trade():
                    if self.current_position is not None and self.current_position.lower() != signal.lower():
                        logger.info("Signal reversal detected. Exiting current trade before new trade.")
                        await self.exit_trade()
                        await asyncio.sleep(1)
                    else:
                        logger.info("Existing position matches new signal; skipping new trade.")
                        await asyncio.sleep(60)
                        continue
                else:
                    self.current_position = None
                    self.current_order_qty = None

                # Dynamic Position Sizing
                current_price = df_1min.iloc[-1]["close"]
                # Use 15-min ATR for position sizing.
                atr_value = df_15["atr"].iloc[-1]
                if atr_value <= 0:
                    logger.warning("ATR value non-positive, cannot compute position size.")
                    await asyncio.sleep(60)
                    continue
                stop_distance = abs(current_price - self.compute_adaptive_stop_loss_and_risk(current_price, atr_value, signal, market_regime, {"rsi": last_candle["rsi"], "macd_diff": last_candle["macd_diff"]})[0])
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
                # Compute stops using our dynamic function.
                boll_temp = ta.volatility.BollingerBands(close=df_15["close"], window=20, window_dev=2)
                bb_width = boll_temp.bollinger_wband()
                bb_width_percentile = (bb_width.rank(pct=True).iloc[-1]) if not bb_width.empty else 0.5
                features = {
                    "rsi": last_candle["rsi"],
                    "macd_diff": last_candle["macd_diff"],
                    "bb_width_percentile": bb_width_percentile
                }
                stop_loss, take_profit = self.compute_sl_tp_dynamic(
                    current_price, atr_value, signal, market_regime, features, base_risk_multiplier=1.5
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
        new_signal = self.ml_service.predict_signal(recent_df)
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
        MIN_REQUIRED = 210  # Approximately 3.5 hours of 1-minute candles
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
                              stop_loss: float = None, take_profit: float = None,
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

# End of file
