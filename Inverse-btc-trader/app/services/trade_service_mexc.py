import warnings
# Suppress RuntimeWarnings from ta.trend (e.g., division warnings)
warnings.filterwarnings("ignore", category=RuntimeWarning, module="ta.trend")

import asyncio
import math
from structlog import get_logger
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import ta

# Use the MEXC client instead of Bybit's client
from app.core.mexc_client import MEXCClient
from app.core import Database, Config
from app.services.ml_service import MLService
from app.services.backfill_service import maybe_backfill_candles

logger = get_logger(__name__)

def get_market_regime(df, adx_period=14, threshold=25):
    try:
        df_clean = df[["high", "low", "close"]].ffill().dropna()
        required_rows = adx_period * 2
        if len(df_clean) < required_rows:
            logger.warning(f"Not enough cleaned data for ADX calculation; require at least {required_rows} rows, got {len(df_clean)}. Defaulting regime to sideways")
            return "sideways"
        adx_indicator = ta.trend.ADXIndicator(
            high=df_clean["high"],
            low=df_clean["low"],
            close=df_clean["close"],
            window=adx_period
        )
        adx_series = adx_indicator.adx().dropna()
        if len(adx_series) < 1:
            logger.warning("ADX series is empty after dropna; defaulting regime to sideways")
            return "sideways"
        latest_adx = adx_series.iloc[-1]
        logger.info("Market regime ADX", adx=latest_adx)
        return "trending" if latest_adx > threshold else "sideways"
    except Exception as e:
        logger.error("Error computing market regime", error=str(e))
        return "sideways"

class TradeService:
    def __init__(self):
        self.ml_service = MLService(lookback=60)
        self.session = None
        self.min_qty = None            # Minimum order quantity (in desired units)
        self.current_position = None   # "buy" or "sell"
        self.current_order_qty = None
        self.last_trade_time = None
        self.running = False
        self.trailing_stop = None

    async def initialize(self):
        try:
            self.session = MEXCClient(
                api_key=Config.MEXC_CONFIG["api_key"],
                api_secret=Config.MEXC_CONFIG["api_secret"],
                base_url=Config.MEXC_CONFIG.get("base_url", "https://contract.mexc.com")
            )
            await self._get_min_order_qty()
            await self.ml_service.initialize()
            logger.info("Trade service initialized successfully (MEXC)")
        except Exception as e:
            logger.critical("Fatal error initializing trade service", error=str(e))
            raise

        asyncio.create_task(self.ml_service.schedule_daily_retrain())
        asyncio.create_task(self.update_trailing_stop_task())
        self.running = True

    async def _get_min_order_qty(self):
        if not self.min_qty:
            info = await asyncio.to_thread(
                self.session.get_instruments_info,
                symbol=Config.TRADING_CONFIG['symbol']
            )
            # Check if the API response indicates an error
            if info.get("code", 0) != 0:
                err_msg = info.get("message", "Unknown error")
                logger.error("Error fetching instrument info", info=info)
                raise KeyError(f"Instrument info error: {err_msg}. Check that the symbol '{Config.TRADING_CONFIG['symbol']}' is valid for MEXC.")
            try:
                # Try 'data' first, then 'result'
                if "data" in info and "list" in info["data"]:
                    self.min_qty = float(info["data"]["list"][0]["minQty"])
                elif "result" in info and "list" in info["result"]:
                    self.min_qty = float(info["result"]["list"][0]["minQty"])
                else:
                    logger.error("MEXC API response does not contain expected keys", info=info)
                    raise KeyError("Expected key 'data' or 'result' not found in API response")
                logger.info("Loaded min order qty", min_qty=self.min_qty)
            except Exception as e:
                logger.error("Error extracting min order qty", error=str(e), info=info)
                raise

    async def check_open_trade(self) -> bool:
        try:
            pos_data = await asyncio.to_thread(
                self.session.get_positions,
                symbol=Config.TRADING_CONFIG['symbol']
            )
            if pos_data.get("code", -1) != 0:
                logger.error("Failed to fetch positions", data=pos_data)
                return False
            positions = pos_data.get("data", [])
            for pos in positions:
                if float(pos.get("positionAmt", "0")) != 0:
                    logger.info("Open position detected", size=pos.get("positionAmt"))
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

    def compute_sl_tp_dynamic(self, current_price: float, atr_value: float, signal: str,
                              market_regime: str, risk_multiplier=1.5, reward_ratio=3.0):
        if market_regime == "trending":
            risk = risk_multiplier * abs(atr_value)
            adjusted_reward_ratio = reward_ratio
        else:
            risk = (risk_multiplier * 0.8) * abs(atr_value)
            adjusted_reward_ratio = reward_ratio * 0.8

        if signal.lower() == "buy":
            stop_loss = current_price - risk
            take_profit = current_price + risk * adjusted_reward_ratio
        elif signal.lower() == "sell":
            stop_loss = current_price + risk
            take_profit = current_price - risk * adjusted_reward_ratio
        else:
            return None, None
        return stop_loss, take_profit

    def compute_trailing_stop(self, current_price: float, atr_value: float, signal: str, trail_multiplier=1.5):
        trail_distance = trail_multiplier * abs(atr_value)
        min_trail_distance = 0.10 * current_price
        if trail_distance < min_trail_distance:
            trail_distance = min_trail_distance
        if signal.lower() == "buy":
            return current_price - trail_distance
        elif signal.lower() == "sell":
            return current_price + trail_distance
        return None

    async def set_trailing_stop(self, trailing_stop: float):
        params = {
            "trailingStop": str(trailing_stop),
            "positionIdx": 0,
            "symbol": Config.TRADING_CONFIG['symbol']
        }
        try:
            response = await asyncio.to_thread(self.session.set_trading_stop, **params)
            logger.info("Trailing stop set", data=response)
        except Exception as e:
            logger.error("Error setting trailing stop", error=str(e))

    async def update_trailing_stop_task(self):
        while self.running:
            if self.current_position is not None and self.trailing_stop is not None:
                try:
                    await self.set_trailing_stop(self.trailing_stop)
                    logger.info("Updated trailing stop", trailing_stop=self.trailing_stop)
                except Exception as ex:
                    logger.error("Error updating trailing stop", error=str(ex))
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
            qty=order_qty,
            stop_loss=None,
            take_profit=None,
            leverage="5",
            trailing_stop=None
        )
        self.current_position = None
        self.current_order_qty = None

    async def run_trading_logic(self):
        if not self.running:
            return
        if Database._pool is None:
            logger.warning("Database is closed; skipping trade logic.")
            return
        try:
            query = """
                SELECT time, open, high, low, close, volume
                FROM candles
                ORDER BY time DESC
                LIMIT 1800
            """
            rows = await Database.fetch(query)
            if not rows or len(rows) < 60:
                logger.warning("Not enough 1-minute candle data for prediction yet.")
                return

            df_1min = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close", "volume"])
            df_1min["time"] = pd.to_datetime(df_1min["time"])
            df_1min.sort_values("time", inplace=True)
            df_1min.set_index("time", inplace=True)
            df_1min = df_1min.asfreq('1min')

            # Resample to 15-minute candles for ML predictions and risk filters
            df_15 = df_1min.resample('15min').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            })
            df_15 = df_15.ffill().dropna(subset=["close"])

            if len(df_15) < 14:
                logger.warning(f"Not enough 15-minute candle data after dropping NaNs; required >= 14 rows, got {len(df_15)}. Fetching extended data...")
                query_ext = """
                    SELECT time, open, high, low, close, volume
                    FROM candles
                    ORDER BY time ASC
                """
                rows_ext = await Database.fetch(query_ext)
                if not rows_ext:
                    logger.warning("No extended historical data available.")
                    return
                df_ext = pd.DataFrame(rows_ext, columns=["time", "open", "high", "low", "close", "volume"])
                df_ext["time"] = pd.to_datetime(df_ext["time"])
                df_ext.sort_values("time", inplace=True)
                df_ext.set_index("time", inplace=True)
                df_ext = df_ext.asfreq('1min')
                df_15 = df_ext.resample('15min').agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                })
                df_15 = df_15.ffill().dropna(subset=["close"])
                if len(df_15) < 14:
                    logger.warning(f"Still not enough 15-minute candle data after extended fetch; required >= 14 rows, got {len(df_15)}.")
                    return

            logger.info(f"1-minute candles count: {len(df_1min)}; Time range: {df_1min.index.min()} to {df_1min.index.max()}")
            logger.info(f"15-minute candles count: {len(df_15)}; Time range: {df_15.index.min()} to {df_15.index.max()}")

            if len(df_15) < 14:
                logger.warning(f"Not enough 15-minute candle data for ML prediction; required >= 14 rows, got {len(df_15)}.")
                return

            df_15["returns"] = df_15["close"].pct_change()
            df_15["rsi"] = ta.momentum.rsi(df_15["close"], window=14)
            macd = ta.trend.MACD(close=df_15["close"], window_slow=26, window_fast=12, window_sign=9)
            df_15["macd"] = macd.macd()
            df_15["macd_signal"] = macd.macd_signal()
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
                logger.warning(f"Not enough 15-minute candle data after dropping NaNs; required >= 14 rows, got {len(df_15)}.")
                return

            market_regime = get_market_regime(df_15)
            logger.info("Market regime detected", regime=market_regime)

            signal = self.ml_service.predict_signal(df_15)
            logger.info("15-minute ML prediction", signal=signal)
            if signal.lower() == "hold":
                return

            last_candle = df_15.iloc[-1]

            pivot = (last_candle["high"] + last_candle["low"] + last_candle["close"]) / 3
            resistance = pivot + (last_candle["high"] - pivot)
            support = pivot - (pivot - last_candle["low"])
            buffer_pct = 0.01
            resistance_threshold = last_candle["close"] * (1 + buffer_pct)
            support_threshold = last_candle["close"] * (1 - buffer_pct)
            if signal.lower() == "buy" and last_candle["close"] > resistance_threshold:
                logger.info("15-minute price near resistance; skipping Buy trade.")
                return
            if signal.lower() == "sell" and last_candle["close"] < support_threshold:
                logger.info("15-minute price near support; skipping Sell trade.")
                return

            if not await self.check_open_trade():
                self.current_position = None
                self.current_order_qty = None

            if self.current_position is not None:
                if self.current_position.lower() != signal.lower():
                    logger.info("Signal reversal detected (internal state). Exiting current trade before new trade.")
                    await self.exit_trade()
                    await asyncio.sleep(1)
                else:
                    logger.info("Existing position matches new signal; skipping new trade.")
                    return
            else:
                if await self.check_open_trade():
                    logger.info("Exchange indicates an open trade but no internal state; skipping new trade.")
                    return

            current_price = df_1min.iloc[-1]["close"]
            fixed_leverage = 5
            logger.info("Using fixed leverage", leverage=fixed_leverage)

            portfolio_value = await self.get_portfolio_value()
            if portfolio_value <= 0:
                logger.warning("Portfolio value is 0; cannot size position.")
                return
            effective_trade_value = math.floor(portfolio_value * fixed_leverage)
            order_qty = effective_trade_value

            stop_loss, take_profit = self.compute_sl_tp_dynamic(
                current_price, df_15["atr"].iloc[-1], signal, market_regime,
                risk_multiplier=1.5, reward_ratio=3.0
            )
            trailing_stop = self.compute_trailing_stop(
                current_price, df_15["atr"].iloc[-1], signal, trail_multiplier=1.5
            )
            logger.info("Computed trailing stop", trailing_stop=trailing_stop)

            logger.info("Trade parameters computed", current_price=current_price,
                        effective_trade_value=effective_trade_value, order_qty=order_qty,
                        stop_loss=stop_loss, take_profit=take_profit, leverage=fixed_leverage)

            await self.set_trade_leverage(fixed_leverage)
            await self.execute_trade(
                side=signal,
                qty=order_qty,
                stop_loss=stop_loss,
                take_profit=take_profit,
                leverage=str(fixed_leverage),
                trailing_stop=None
            )
            await self.set_trailing_stop(trailing_stop)

            self.last_trade_time = datetime.now(timezone.utc)
            self.current_position = signal.lower()
            self.current_order_qty = order_qty

        except Exception as e:
            logger.error("Error in run_trading_logic", error=str(e))

    async def set_trade_leverage(self, leverage: float):
        try:
            if hasattr(self.session, "set_leverage"):
                response = await asyncio.to_thread(
                    self.session.set_leverage,
                    symbol=Config.TRADING_CONFIG['symbol'],
                    leverage=str(leverage)
                )
                logger.info("Set leverage response", data=response)
            else:
                logger.info("No set_leverage method available; relying on order parameter only.")
        except Exception as e:
            logger.error("Error setting leverage", error=str(e))

    async def execute_trade(self, side: str, qty: float,
                              stop_loss: float = None, take_profit: float = None,
                              leverage: float = None, trailing_stop: float = None):
        if not self.running:
            return
        if Database._pool is None:
            logger.warning("Database is closed; skipping trade execution.")
            return
        if qty < self.min_qty:
            logger.error("Position size below minimum", required=self.min_qty, actual=qty)
            return

        logger.info("Placing trade",
                    side=side,
                    size=qty,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    leverage=leverage,
                    trailing_stop=trailing_stop)

        order_params = {
            "symbol": Config.TRADING_CONFIG['symbol'],
            "side": side.upper(),
            "orderType": "MARKET",
            "quantity": str(qty),
            "stopLoss": str(stop_loss) if stop_loss is not None else None,
            "takeProfit": str(take_profit) if take_profit is not None else None,
            "leverage": str(leverage) if leverage is not None else None
        }
        order_params = {k: v for k, v in order_params.items() if v is not None}

        try:
            response = await asyncio.to_thread(
                self.session.place_order,
                **order_params
            )
            if response.get("code") == 0:
                await self._log_trade(side, qty)
                logger.info("Trade executed successfully",
                            order_id=response["result"].get("orderId", ""))
            else:
                error_msg = response.get("message", "Unknown error")
                logger.error("Trade failed", error=error_msg)
        except Exception as e:
            logger.error("Trade execution error", error=str(e))
            self.current_position = None

    async def set_trailing_stop(self, trailing_stop: float):
        params = {
            "trailingStop": str(trailing_stop),
            "positionIdx": 0,
            "symbol": Config.TRADING_CONFIG['symbol']
        }
        try:
            response = await asyncio.to_thread(self.session.set_trading_stop, **params)
            logger.info("Trailing stop set", data=response)
        except Exception as e:
            logger.error("Error setting trailing stop", error=str(e))

    async def get_portfolio_value(self) -> float:
        try:
            balance_data = await asyncio.to_thread(
                self.session.get_wallet_balance,
                accountType="UNIFIED"
            )
            logger.info("Wallet balance data", data=balance_data)
            total_balance = 0.0
            account_list = balance_data.get("data", [])
            if not account_list:
                return 0.0
            for asset in account_list:
                if asset.get("coin") == "BTC":
                    total_balance += float(asset.get("usdValue", 0.0))
            return total_balance
        except Exception as e:
            logger.error("Failed to fetch portfolio value", error=str(e))
            return 0.0

    async def _log_trade(self, side: str, qty: float):
        if Database._pool is None:
            logger.warning("Database is closed; skipping trade logging.")
            return
        try:
            await Database.execute('''
                INSERT INTO trades (time, side, price, quantity)
                VALUES ($1, $2, $3, $4)
            ''', datetime.now(timezone.utc), side, 0, qty)
        except Exception as e:
            logger.error("Failed to log trade", error=str(e))

    async def stop(self):
        self.running = False
        if hasattr(self.session, 'close'):
            await asyncio.to_thread(self.session.close)
        logger.info("Trade service stopped")
