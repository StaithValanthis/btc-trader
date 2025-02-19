# File: app/services/trade_service.py

import asyncio
from pybit.unified_trading import HTTP
from structlog import get_logger
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import ta

from app.core import Database, Config
from app.services.ml_service import MLService
from app.services.backfill_service import maybe_backfill_candles

logger = get_logger(__name__)

class TradeService:
    def __init__(self):
        # Initialize MLService with your desired lookback (e.g., 60)
        self.ml_service = MLService(lookback=60)
        self.session = None
        self.min_qty = None            # Minimum order quantity (in USD notional)
        self.current_position = None   # Current trade direction, if any
        self.last_trade_time = None
        self.running = False

    async def initialize(self):
        """Initialize the Bybit session, database, and ML service."""
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

        # Start ML daily retraining (if desired)
        asyncio.create_task(self.ml_service.schedule_daily_retrain())
        self.running = True

    async def _get_min_order_qty(self):
        """Fetch the minimum order quantity for the inverse BTCUSD contract."""
        if not self.min_qty:
            info = await asyncio.to_thread(
                self.session.get_instruments_info,
                category="inverse",
                symbol=Config.TRADING_CONFIG['symbol']
            )
            self.min_qty = float(info['result']['list'][0]['lotSizeFilter']['minOrderQty'])
            logger.info("Loaded min order qty", min_qty=self.min_qty)

    async def check_open_trade(self) -> bool:
        """
        Check for open positions on Bybit for the configured symbol.
        Returns True if an open position exists; otherwise, False.
        """
        try:
            pos_data = await asyncio.to_thread(
                self.session.get_positions,
                category="inverse",
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
        """
        Calculate dynamic leverage based on the volatility ratio (ATR/current_price).
        (This function is retained for reference; in our fixed-leverage version we use 10x.)
        """
        vol_ratio = abs(atr_value) / current_price
        if vol_ratio >= 0.02:
            leverage = 1.0
        elif vol_ratio <= 0.005:
            leverage = 20.0
        else:
            leverage = 20.0 - ((vol_ratio - 0.005) / 0.015) * (20.0 - 1.0)
        return round(leverage, 2)

    def compute_sl_tp(self, current_price: float, atr_value: float, support: float, resistance: float, signal: str, multiplier: float = 2.0):
        """
        Compute stop loss and take profit levels using a weighted combination of ATR-based
        and support/resistance–based levels.
        
        For Buy orders:
          SL = current_price - [0.5*(multiplier*ATR) + 0.5*(current_price - support)]
          TP = current_price + [0.5*(multiplier*ATR) + 0.5*(resistance - current_price)]
          
        For Sell orders:
          SL = current_price + [0.5*(multiplier*ATR) + 0.5*(resistance - current_price)]
          TP = current_price - [0.5*(multiplier*ATR) + 0.5*(current_price - support)]
        """
        atr_component = multiplier * abs(atr_value)
        if signal.lower() == "buy":
            stop_loss = current_price - (0.5 * atr_component + 0.5 * (current_price - support))
            take_profit = current_price + (0.5 * atr_component + 0.5 * (resistance - current_price))
            if stop_loss >= current_price:
                stop_loss = current_price - atr_component
            return stop_loss, take_profit
        elif signal.lower() == "sell":
            stop_loss = current_price + (0.5 * atr_component + 0.5 * (resistance - current_price))
            take_profit = current_price - (0.5 * atr_component + 0.5 * (current_price - support))
            if stop_loss <= current_price:
                stop_loss = current_price + atr_component
            return stop_loss, take_profit
        else:
            return None, None

    async def run_trading_logic(self):
        """
        Execute the trading logic:
        
        1. Fetch recent 1-minute candle data from the 'candles' table.
        2. If fewer than 900 rows are available, trigger backfill.
        3. Ensure the 1-minute data is continuous.
        4. Resample the 1-minute data to 15-minute candles using .agg() with a custom dictionary.
        5. Forward fill missing values and drop rows missing the 'close' value.
        6. Log the 1-minute and 15-minute data details (row count and time range).
        7. Compute technical indicators (RSI, MACD, Bollinger Bands, ATR) on the 15-minute data.
        8. Generate an ML prediction signal using the 15-minute data.
        9. Compute support/resistance from the most recent 15-minute candle.
        10. Check for open trades on the exchange.
        11. Use fixed leverage of 10x.
        12. Compute effective trade value = portfolio_value * 0.02 * 10 (rounded to the nearest dollar)
             and use this value as the order quantity.
        13. Compute stop loss and take profit using a weighted combination of ATR and support/resistance.
        14. Override the exchange default leverage with fixed 10x.
        15. Execute the trade.
        """
        if not self.running:
            return
        if Database._pool is None:
            logger.warning("Database is closed; skipping trade logic.")
            return

        try:
            # 1. Fetch the most recent 900 1-minute candles.
            query = """
                SELECT time, open, high, low, close, volume
                FROM candles
                ORDER BY time DESC
                LIMIT 900
            """
            rows = await Database.fetch(query)
            if not rows or len(rows) < 60:
                logger.warning("Not enough 1-minute candle data for prediction yet.")
                return

            # 2. Trigger backfill if fewer than 900 rows are available.
            if len(rows) < 900:
                logger.info(f"Only {len(rows)} 1-minute candles available; initiating backfill.")
                await maybe_backfill_candles(min_rows=900, symbol=Config.TRADING_CONFIG['symbol'], interval=1, days_to_fetch=7)
                rows = await Database.fetch(query)
                if not rows or len(rows) < 60:
                    logger.warning("Still not enough 1-minute candle data after backfill.")
                    return

            df_1min = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close", "volume"])
            df_1min["time"] = pd.to_datetime(df_1min["time"])
            df_1min.sort_values("time", inplace=True)
            df_1min.set_index("time", inplace=True)
            
            # 3. Ensure the 1-minute data is continuous.
            df_1min = df_1min.asfreq('1min')
            
            # 4. Resample to 15-minute candles using .agg() with a custom dictionary.
            df_15 = df_1min.resample('15min').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            })
            
            # 5. Forward fill missing values and drop rows missing the 'close' value.
            df_15 = df_15.ffill().dropna(subset=["close"])
            
            # 6. Log details of the 1-minute and 15-minute data.
            min_time_1min = df_1min.index.min()
            max_time_1min = df_1min.index.max()
            logger.info(f"1-minute candles count: {len(df_1min)}; Time range: {min_time_1min} to {max_time_1min}")
            min_time_15 = df_15.index.min()
            max_time_15 = df_15.index.max()
            logger.info(f"15-minute candles count: {len(df_15)}; Time range: {min_time_15} to {max_time_15}")
            
            if len(df_15) < 14:
                logger.warning(f"Not enough 15-minute candle data for ML prediction; required >= 14 rows, got {len(df_15)}.")
                return

            # 7. Compute technical indicators on the 15-minute data.
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

            # 8. Generate ML prediction signal using the 15-minute data.
            signal = self.ml_service.predict_signal(df_15)
            logger.info("15-minute ML prediction", signal=signal)
            if signal == "Hold":
                return

            # 9. Compute support/resistance from the most recent 15-minute candle.
            last_candle = df_15.iloc[-1]
            pivot = (last_candle["high"] + last_candle["low"] + last_candle["close"]) / 3
            resistance = pivot + (last_candle["high"] - pivot)
            support = pivot - (pivot - last_candle["low"])
            buffer_pct = 0.01  # 1% buffer
            resistance_threshold = last_candle["close"] * (1 + buffer_pct)
            support_threshold = last_candle["close"] * (1 - buffer_pct)
            if signal.lower() == "buy" and last_candle["close"] > resistance_threshold:
                logger.info("15-minute price near resistance; skipping Buy trade.")
                return
            if signal.lower() == "sell" and last_candle["close"] < support_threshold:
                logger.info("15-minute price near support; skipping Sell trade.")
                return

            # 10. Check for open trades.
            if await self.check_open_trade():
                logger.info("An open trade exists; skipping new trade.")
                return

            current_price = last_candle["close"]

            # 11. Use fixed leverage of 10x.
            fixed_leverage = 10.0
            logger.info("Using fixed leverage", leverage=fixed_leverage)

            # 12. Compute effective trade value.
            portfolio_value = await self.get_portfolio_value()
            if portfolio_value <= 0:
                logger.warning("Portfolio value is 0; cannot size position.")
                return
            effective_trade_value = round(portfolio_value * 0.02 * fixed_leverage)
            order_qty = effective_trade_value
            order_qty = max(order_qty, self.min_qty)

            # 13. Compute stop loss and take profit using a weighted combination of ATR and support/resistance.
            multiplier = 2.0
            if signal.lower() == "buy":
                stop_loss, take_profit = self.compute_sl_tp(
                    current_price, df_15["atr"].iloc[-1],
                    support, resistance, "buy", multiplier)
                if stop_loss >= current_price:
                    stop_loss = current_price - (multiplier * abs(df_15["atr"].iloc[-1])) - (0.0001 * current_price)
            elif signal.lower() == "sell":
                stop_loss, take_profit = self.compute_sl_tp(
                    current_price, df_15["atr"].iloc[-1],
                    support, resistance, "sell", multiplier)
                if stop_loss <= current_price:
                    stop_loss = current_price + (multiplier * abs(df_15["atr"].iloc[-1])) + (0.0001 * current_price)
            else:
                return

            logger.info("Trade parameters computed", current_price=current_price,
                        effective_trade_value=effective_trade_value, order_qty=order_qty,
                        stop_loss=stop_loss, take_profit=take_profit, leverage=fixed_leverage)

            # 14. Override the exchange leverage.
            await self.set_trade_leverage(fixed_leverage)

            # 15. Execute the trade.
            await self.execute_trade(
                side=signal,
                qty=order_qty,
                stop_loss=stop_loss,
                take_profit=take_profit,
                leverage=fixed_leverage
            )

            self.last_trade_time = datetime.now(timezone.utc)
            self.current_position = signal.lower()

        except Exception as e:
            logger.error("Error in run_trading_logic", error=str(e))

    async def set_trade_leverage(self, leverage: float):
        """
        Set the leverage for the symbol explicitly using the Bybit API.
        This call overrides the account's default leverage for this symbol.
        """
        try:
            if hasattr(self.session, "set_leverage"):
                response = await asyncio.to_thread(
                    self.session.set_leverage,
                    category="inverse",
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
                              leverage: float = None):
        """
        Execute a market order on Bybit inverse BTCUSD with optional stop loss, take profit, and leverage.
        Note: The order is executed at market price (no price parameter is sent).
        """
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
                    leverage=leverage)

        order_params = {
            "category": "inverse",
            "symbol": Config.TRADING_CONFIG['symbol'],
            "side": side,
            "orderType": "Market",
            "qty": str(qty),
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
            if response['retCode'] == 0:
                await self._log_trade(side, qty)
                logger.info("Trade executed successfully",
                            order_id=response['result']['orderId'])
            else:
                error_msg = response.get('retMsg', 'Unknown error')
                logger.error("Trade failed", error=error_msg)
        except Exception as e:
            logger.error("Trade execution error", error=str(e))
            self.current_position = None

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
        """Stop the trade service so no more trades are attempted."""
        self.running = False
        if hasattr(self.session, 'close'):
            await asyncio.to_thread(self.session.close)
        logger.info("Trade service stopped")
