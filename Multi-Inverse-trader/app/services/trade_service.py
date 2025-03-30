# trade_service.py

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="ta.trend")

import asyncio
import math
import json
from datetime import datetime, timezone

import pandas as pd
import numpy as np
import ta

from pybit.unified_trading import HTTP
from structlog import get_logger

from app.core import Database, Config
from app.services.ml_service import MLService
from app.services.backfill_service import maybe_backfill_candles
from app.utils.cache import redis_client

logger = get_logger(__name__)

def enhanced_get_market_regime(df, adx_period=14, base_threshold=25):
    """
    Determines the market regime (uptrending, downtrending, sideways)
    by referencing ADX, MAs, and volatility.
    """
    try:
        df_clean = df[["high", "low", "close"]].ffill().dropna()
        required_rows = adx_period * 2
        if len(df_clean) < required_rows:
            logger.warning("Not enough data for ADX calculation; defaulting regime to sideways")
            return "sideways"

        adx_indicator = ta.trend.ADXIndicator(
            high=df_clean["high"], low=df_clean["low"],
            close=df_clean["close"], window=adx_period
        )
        adx_series = adx_indicator.adx().dropna()
        latest_adx = adx_series.iloc[-1] if not adx_series.empty else 0

        df_clean['SMA20'] = df_clean['close'].rolling(window=20).mean()
        df_clean['SMA50'] = df_clean['close'].rolling(window=50).mean()
        sma_diff = df_clean['SMA20'].iloc[-1] - df_clean['SMA50'].iloc[-1]
        ma_trend = "up" if sma_diff > 0 else "down"

        atr_indicator = ta.volatility.AverageTrueRange(
            high=df_clean["high"], low=df_clean["low"],
            close=df_clean["close"], window=14
        )
        atr_series = atr_indicator.average_true_range().dropna()
        atr_percentile = atr_series.rank(pct=True).iloc[-1] if not atr_series.empty else 0

        boll = ta.volatility.BollingerBands(
            close=df_clean["close"], window=20, window_dev=2
        )
        bb_width = boll.bollinger_wband()
        bb_width_percentile = bb_width.rank(pct=True).iloc[-1] if not bb_width.empty else 0

        adaptive_threshold = base_threshold * (1 - 0.2 * atr_percentile)

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
                    adx=latest_adx,
                    adaptive_threshold=adaptive_threshold,
                    ma_trend=ma_trend,
                    sma_diff=sma_diff,
                    atr_percentile=atr_percentile,
                    bb_width_percentile=bb_width_percentile,
                    regime=regime)
        return regime
    except Exception as e:
        logger.error("Error in enhanced market regime detection", error=str(e))
        return "sideways"


class TradeService:
    def __init__(self, symbol):
        self.symbol = symbol
        self.ml_service = MLService(
            symbol,
            lookback=60,
            ensemble_size=3,
            use_tuned_trend_model=True,
            use_tuned_signal_model=True
        )
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
            logger.info(f"Trade service for {self.symbol} initialized successfully")
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
                symbol=self.symbol
            )
            self.min_qty = float(info['result']['list'][0]['lotSizeFilter']['minOrderQty'])
            logger.info("Loaded min order qty", min_qty=self.min_qty, symbol=self.symbol)

    async def check_open_trade(self) -> bool:
        try:
            pos_data = await asyncio.to_thread(
                self.session.get_positions,
                category=Config.BYBIT_CONFIG.get('category', 'inverse'),
                symbol=self.symbol
            )
            if pos_data.get("retCode", -1) != 0:
                logger.error("Failed to fetch positions", data=pos_data)
                return False
            positions = pos_data["result"].get("list", [])
            for pos in positions:
                if float(pos.get("size", "0")) != 0:
                    logger.info("Open position detected", size=pos.get("size"), symbol=self.symbol)
                    return True
            return False
        except Exception as e:
            logger.error("Error checking open trade", error=str(e), symbol=self.symbol)
            return False

    def compute_adaptive_stop_loss_and_risk(self, current_price: float, atr_value: float,
                                            signal: str, market_regime: str,
                                            df_features: dict,
                                            base_risk_multiplier=1.5) -> (float, float):
        raw_risk = base_risk_multiplier * abs(atr_value)
        risk = min(raw_risk, 0.10 * current_price)
        if signal.lower() == "buy":
            stop_loss = current_price - risk
        elif signal.lower() == "sell":
            stop_loss = current_price + risk
        else:
            stop_loss = current_price
        return stop_loss, risk

    def compute_sl_tp_dynamic(self, current_price: float, atr_value: float,
                              signal: str, market_regime: str,
                              df_features: dict,
                              base_risk_multiplier=1.5, min_reward_risk_ratio=2.0) -> (float, float):
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
        params = {
            "category": Config.BYBIT_CONFIG.get("category", "inverse"),
            "symbol": self.symbol,
            "stopLoss": str(stop_loss),
            "takeProfit": str(take_profit),
            "positionIdx": 0
        }
        try:
            response = await asyncio.to_thread(self.session.set_trading_stop, **params)
            logger.info("Trade stops updated", data=response, symbol=self.symbol)
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
                        logger.warning("Not enough 15-minute data for updating trade stops.", symbol=self.symbol)
                        await asyncio.sleep(60)
                        continue

                    df_15["rsi"] = ta.momentum.rsi(df_15["close"], window=14).ffill()
                    macd = ta.trend.MACD(close=df_15["close"], window_slow=26, window_fast=12, window_sign=9)
                    df_15["macd"] = macd.macd()
                    df_15["macd_diff"] = macd.macd_diff().ffill()

                    boll = ta.volatility.BollingerBands(close=df_15["close"], window=20, window_dev=2)
                    df_15["bb_high"] = boll.bollinger_hband()
                    df_15["bb_low"] = boll.bollinger_lband()
                    df_15["bb_mavg"] = boll.bollinger_mavg()

                    atr_indicator = ta.volatility.AverageTrueRange(
                        high=df_15["high"], low=df_15["low"], close=df_15["close"], window=14
                    )
                    df_15["atr"] = atr_indicator.average_true_range()

                    boll_temp = ta.volatility.BollingerBands(close=df_15["close"], window=20, window_dev=2)
                    bb_width = boll_temp.bollinger_wband()
                    bb_width_percentile = (bb_width.rank(pct=True).iloc[-1]) if not bb_width.empty else 0.5
                    features = {
                        "rsi": df_15["rsi"].iloc[-1],
                        "macd_diff": df_15["macd_diff"].iloc[-1],
                        "bb_width_percentile": bb_width_percentile
                    }
                    market_regime = enhanced_get_market_regime(df_15)
                    current_price = recent_df.iloc[-1]["close"]
                    stop_loss, take_profit = self.compute_sl_tp_dynamic(
                        current_price, df_15["atr"].iloc[-1],
                        self.current_position, market_regime, features,
                        base_risk_multiplier=1.5
                    )
                    await self.update_trade_stops(stop_loss, take_profit)
                    logger.info("Updated open trade stops", stop_loss=stop_loss,
                                take_profit=take_profit, symbol=self.symbol)
                except Exception as e:
                    logger.error("Error updating open trade stops", error=str(e), symbol=self.symbol)
            await asyncio.sleep(60)

    async def run_trading_logic(self):
        # Wait for ML models to be ready
        while not (self.ml_service.trend_model_ready and self.ml_service.signal_model_ready):
            logger.info("ML models not ready, waiting before evaluating trade signal.", symbol=self.symbol)
            await asyncio.sleep(10)

        while self.running:
            try:
                redis_key = f"recent_candles_{self.symbol}"
                cached = redis_client.get(redis_key)
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
                        WHERE symbol=$1
                        ORDER BY time DESC
                        LIMIT 1800
                    """, self.symbol)
                    if not rows or len(rows) < 60:
                        logger.warning("Not enough 1-minute candle data for prediction yet.", symbol=self.symbol)
                        await asyncio.sleep(60)
                        continue
                    df_1min = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close", "volume"])
                    df_1min['time'] = pd.to_datetime(df_1min['time'])
                    df_1min.sort_values("time", inplace=True)
                    df_1min.set_index("time", inplace=True)
                    redis_client.setex(redis_key, 60, df_1min.reset_index().to_json(orient="records"))

                df_1min = df_1min.asfreq('1min')
                df_15 = df_1min.resample('15min').agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }).ffill().dropna()
                if len(df_15) < 14:
                    logger.warning("Not enough 15-minute candle data for ML prediction.", symbol=self.symbol)
                    await asyncio.sleep(60)
                    continue

                atr_indicator = ta.volatility.AverageTrueRange(
                    high=df_15["high"], low=df_15["low"], close=df_15["close"], window=14
                )
                df_15["atr"] = atr_indicator.average_true_range()
                df_15["rsi"] = ta.momentum.rsi(df_1min["close"], window=14).ffill()
                macd = ta.trend.MACD(close=df_15["close"], window_slow=26, window_fast=12, window_sign=9)
                df_15["macd_diff"] = macd.macd_diff().ffill()

                market_regime = enhanced_get_market_regime(df_15)
                logger.info("Market regime detected", regime=market_regime, symbol=self.symbol)
                if market_regime.lower() == "sideways":
                    logger.info("Market is sideways. Skipping trade entry.", symbol=self.symbol)
                    await asyncio.sleep(60)
                    continue

                # Predict
                trend_prediction = self.ml_service.predict_trend(df_15)
                signal_prediction = self.ml_service.predict_signal(df_15)
                logger.info("Trend prediction", trend=trend_prediction, symbol=self.symbol)
                logger.info("15-minute signal prediction", signal=signal_prediction, symbol=self.symbol)

                # Confluence check
                if trend_prediction.lower() == "uptrending" and signal_prediction.lower() != "buy":
                    logger.info("Trend not matching signal=Buy => skip", symbol=self.symbol)
                    await asyncio.sleep(60)
                    continue
                if trend_prediction.lower() == "downtrending" and signal_prediction.lower() != "sell":
                    logger.info("Trend not matching signal=Sell => skip", symbol=self.symbol)
                    await asyncio.sleep(60)
                    continue

                signal = signal_prediction
                if await self.check_open_trade():
                    if self.current_position and self.current_position.lower() != signal.lower():
                        logger.info("Signal reversal => Exiting current trade", symbol=self.symbol)
                        await self.exit_trade()
                        await asyncio.sleep(1)
                    else:
                        logger.info("Existing position matches new signal => skip new trade", symbol=self.symbol)
                        await asyncio.sleep(60)
                        continue
                else:
                    self.current_position = None
                    self.current_order_qty = None

                current_price = df_1min.iloc[-1]["close"]
                atr_value = df_15["atr"].iloc[-1]
                if atr_value <= 0:
                    logger.warning("ATR value non-positive, cannot compute position size", symbol=self.symbol)
                    await asyncio.sleep(60)
                    continue
                stop_distance = abs(
                    current_price
                    - self.compute_adaptive_stop_loss_and_risk(
                        current_price, atr_value, signal, trend_prediction, {"rsi": df_15["rsi"].iloc[-1]}
                    )[0]
                )
                if stop_distance == 0:
                    logger.warning("Stop distance=0 => cannot compute position size", symbol=self.symbol)
                    await asyncio.sleep(60)
                    continue

                portfolio_value = await self.get_portfolio_value()
                if portfolio_value <= 0:
                    logger.warning("Portfolio value=0 => cannot size position", symbol=self.symbol)
                    await asyncio.sleep(60)
                    continue

                # Basic risk-based sizing
                risk_per_trade = 0.01 * portfolio_value
                computed_qty = (risk_per_trade / stop_distance) * current_price
                order_qty = max(computed_qty, self.min_qty)
                order_qty = round(order_qty)
                logger.info("ATR-based position sizing computed",
                            portfolio_value=portfolio_value,
                            risk_per_trade=risk_per_trade,
                            stop_distance=stop_distance,
                            computed_qty=computed_qty,
                            final_qty=order_qty,
                            symbol=self.symbol)
                fixed_leverage = 1
                await self.set_trade_leverage(fixed_leverage)

                # Additional features
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
                logger.error("Error in trading logic", error=str(e), symbol=self.symbol)
            await asyncio.sleep(60)

    async def exit_trade(self):
        if not self.current_position:
            logger.warning("No current position to exit", symbol=self.symbol)
            return
        exit_side = "Sell" if self.current_position.lower() == "buy" else "Buy"
        order_qty = self.current_order_qty if self.current_order_qty else self.min_qty
        logger.info("Exiting trade", exit_side=exit_side, qty=order_qty, symbol=self.symbol)
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

    async def scale_in_trade(self, side: str, additional_qty: int, delay: int=60):
        await asyncio.sleep(delay)
        recent_df = await self.get_recent_candles()
        new_signal = self.ml_service.predict_signal(recent_df)
        if new_signal.lower() == side.lower():
            logger.info("Scaling in additional position", side=side, additional_qty=additional_qty, symbol=self.symbol)
            await self.execute_trade(side=side, qty=str(additional_qty))
            self.scaled_in = True
        else:
            logger.info("Market conditions not favorable for scaling in", symbol=self.symbol)

    async def get_recent_candles(self):
        MIN_REQUIRED = 210
        redis_key = f"recent_candles_{self.symbol}"
        cached = redis_client.get(redis_key)
        if cached:
            data = json.loads(cached)
            df = pd.DataFrame(data)
            if len(df) < MIN_REQUIRED:
                rows = await Database.fetch("""
                    SELECT time, open, high, low, close, volume
                    FROM candles
                    WHERE symbol=$1
                    ORDER BY time DESC
                    LIMIT 1800
                """, self.symbol)
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
                WHERE symbol=$1
                ORDER BY time DESC
                LIMIT 1800
            """, self.symbol)
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
            logger.info("Wallet balance data", data=balance_data, symbol=self.symbol)
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
            logger.error("Failed to fetch portfolio value", error=str(e), symbol=self.symbol)
            return 0.0

    async def _log_trade(self, side: str, qty: str):
        if Database._pool is None:
            logger.warning("Database is closed; skipping trade logging.", symbol=self.symbol)
            return
        try:
            await Database.execute('''
                INSERT INTO trades (symbol, time, side, price, quantity)
                VALUES ($1, $2, $3, $4, $5)
            ''', self.symbol, datetime.now(timezone.utc), side, 0, float(qty))
        except Exception as e:
            logger.error("Failed to log trade", error=str(e), symbol=self.symbol)

    async def set_trade_leverage(self, leverage: int):
        if Config.BYBIT_CONFIG.get("category", "inverse") == "inverse":
            logger.info("Skipping set_leverage for inverse contracts", symbol=self.symbol)
            return
        try:
            response = await asyncio.to_thread(
                self.session.set_leverage,
                category=Config.BYBIT_CONFIG.get("category", "inverse"),
                symbol=self.symbol,
                leverage=str(leverage)
            )
            logger.info("Set leverage response", data=response, symbol=self.symbol)
        except Exception as e:
            logger.error("Error setting leverage", error=str(e), symbol=self.symbol)

    async def execute_trade(self, side: str, qty: str,
                            stop_loss: float = None,
                            take_profit: float = None,
                            leverage: str = None):
        if not self.running:
            return
        if float(qty) < self.min_qty:
            logger.warning("Computed position size below minimum; adjusting to minimum",
                           required=self.min_qty, actual=qty, symbol=self.symbol)
            qty = str(self.min_qty)
        logger.info("Placing trade",
                    side=side,
                    size=qty,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    leverage=leverage,
                    symbol=self.symbol)
        order_params = {
            "category": Config.BYBIT_CONFIG.get("category", "inverse"),
            "symbol": self.symbol,
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
                logger.info("Trade executed successfully",
                            order_id=response['result']['orderId'],
                            symbol=self.symbol)
            else:
                error_msg = response.get('retMsg', 'Unknown error')
                logger.error("Trade failed", error=error_msg, symbol=self.symbol)
        except Exception as e:
            logger.error("Trade execution error", error=str(e), symbol=self.symbol)
            self.current_position = None
