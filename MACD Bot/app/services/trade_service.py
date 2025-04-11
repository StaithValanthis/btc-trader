# File: app/services/trade_service.py

import asyncio
import json
from datetime import datetime, timezone

import pandas as pd
import ta
from pybit.unified_trading import HTTP
from structlog import get_logger

from app.core import Database, Config
from app.services.backfill_service import maybe_backfill_candles
from app.utils.cache import redis_client

logger = get_logger(__name__)


class TradeService:
    def __init__(self):
        # No ML model – we now use our MACD/ATR strategy.
        self.session = None
        self.min_qty = None            # Minimum order size (in contracts)
        self.current_position = None   # "long" or "short"
        self.entry_price = None        # Record the entry price at trade entry
        self.running = False

    async def initialize(self):
        try:
            self.session = HTTP(
                testnet=Config.BYBIT_CONFIG['testnet'],
                api_key=Config.BYBIT_CONFIG['api_key'],
                api_secret=Config.BYBIT_CONFIG['api_secret'],
                recv_window=5000
            )
            await self._get_min_order_qty()
            logger.info("Trade service initialized successfully")
        except Exception as e:
            logger.critical("Fatal error initializing trade service", error=str(e))
            raise

        # Start scheduled tasks
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

    async def run_trading_logic(self):
        """
        Every 10 minutes, this loop:
          1. Fetches recent 1-minute candles from the database.
          2. Resamples these into 10-minute, 15-minute, and 60-minute dataframes.
          3. Computes a MACD on the 60-minute data and ATR on the 15-minute data.
          4. Forward fills these values onto the 10-minute bars.
          5. Checks for MACD crossovers to generate long or short signals.
          6. If no open trade exists and a crossover occurs, enters a trade.
          7. If a trade is open and a new signal is in the opposite direction, reverses the trade.
          8. Otherwise, if the signal is in the same direction, it ignores the new signal.
          9. Also checks for exit conditions based on ATR.
        """
        while self.running:
            try:
                # Fetch recent 1-minute candle data
                rows = await Database.fetch("""
                    SELECT time, open, high, low, close, volume
                    FROM candles
                    ORDER BY time DESC
                    LIMIT 1000
                """)
                if not rows or len(rows) < 60:
                    logger.warning("Not enough 1-minute candle data for strategy evaluation.")
                    await asyncio.sleep(60)
                    continue

                df_1min = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close", "volume"])
                df_1min['time'] = pd.to_datetime(df_1min['time'])
                df_1min.sort_values("time", inplace=True)
                df_1min.set_index("time", inplace=True)

                # Resample to 10-minute bars (for trade signals)
                df_10 = df_1min.resample("10min").agg({
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum"
                }).dropna()

                # Resample to 60-minute bars for MACD calculation
                df_60 = df_1min.resample("60min").agg({
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum"
                }).dropna()

                # Resample to 15-minute bars for ATR calculation
                df_15 = df_1min.resample("15min").agg({
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum"
                }).dropna()

                # Compute MACD on the 60-minute data
                macd_indicator = ta.trend.MACD(
                    close=df_60["close"],
                    window_fast=12,
                    window_slow=26,
                    window_sign=9
                )
                df_60["macd"] = macd_indicator.macd()
                df_60["signal_line"] = macd_indicator.macd_signal()

                # Compute ATR on the 15-minute data
                atr_indicator = ta.volatility.AverageTrueRange(
                    high=df_15["high"],
                    low=df_15["low"],
                    close=df_15["close"],
                    window=14
                )
                df_15["atr"] = atr_indicator.average_true_range()

                # Forward-fill MACD and ATR values onto the 10-minute dataframe
                df_10["macd"] = df_60["macd"].reindex(df_10.index, method="ffill")
                df_10["signal_line"] = df_60["signal_line"].reindex(df_10.index, method="ffill")
                df_10["atr"] = df_15["atr"].reindex(df_10.index, method="ffill")
                df_10.sort_index(inplace=True)

                # Identify MACD crossovers on the 10-minute bars
                df_10["prev_macd"] = df_10["macd"].shift(1)
                df_10["prev_signal"] = df_10["signal_line"].shift(1)
                df_10["long_signal"] = (df_10["macd"] > df_10["signal_line"]) & (df_10["prev_macd"] <= df_10["prev_signal"])
                df_10["short_signal"] = (df_10["macd"] < df_10["signal_line"]) & (df_10["prev_macd"] >= df_10["prev_signal"])

                # Use the most recent 10-minute bar to check for a signal
                current_bar = df_10.iloc[-1]
                signal = None
                if current_bar["long_signal"]:
                    signal = "long"
                elif current_bar["short_signal"]:
                    signal = "short"
                else:
                    signal = "none"

                # Check if there is an open trade (using our internal flag)
                open_trade = await self.check_open_trade()

                if open_trade:
                    # If a trade is open, decide based on the new signal:
                    if signal == "none":
                        # No new signal; check exit conditions based on ATR
                        current_price = current_bar["close"]
                        atr_value = current_bar["atr"]
                        if self.current_position == "long":
                            if current_price <= (self.entry_price - atr_value) or current_price >= (self.entry_price + 1.5 * atr_value):
                                await self.exit_trade(current_price)
                        elif self.current_position == "short":
                            if current_price >= (self.entry_price + atr_value) or current_price <= (self.entry_price - 1.5 * atr_value):
                                await self.exit_trade(current_price)
                    elif signal == self.current_position:
                        logger.info("Existing trade in same direction; ignoring new signal.")
                    else:
                        # New signal is in the opposite direction: reverse the trade.
                        current_price = current_bar["close"]
                        atr_value = current_bar["atr"]
                        await self.exit_trade(current_price)
                        # Enter new trade in opposite direction with updated SL/TP.
                        entry_price = current_price
                        if signal == "long":
                            stop_loss = entry_price - atr_value
                            take_profit = entry_price + 1.5 * atr_value
                        else:  # signal == "short"
                            stop_loss = entry_price + atr_value
                            take_profit = entry_price - 1.5 * atr_value
                        await self.execute_trade(
                            side=signal,
                            qty=None,  # Use configured POSITION_SIZE
                            entry_price=entry_price,
                            stop_loss=stop_loss,
                            take_profit=take_profit
                        )
                        self.current_position = signal
                        self.entry_price = entry_price
                        logger.info(f"Reversed trade: entered {signal.upper()} trade at {entry_price}, SL: {stop_loss}, TP: {take_profit}")
                else:
                    if signal in ["long", "short"]:
                        entry_price = current_bar["close"]
                        atr_value = current_bar["atr"]
                        if signal == "long":
                            stop_loss = entry_price - atr_value
                            take_profit = entry_price + 1.5 * atr_value
                        else:
                            stop_loss = entry_price + atr_value
                            take_profit = entry_price - 1.5 * atr_value
                        await self.execute_trade(
                            side=signal,
                            qty=None,  # Use configured POSITION_SIZE
                            entry_price=entry_price,
                            stop_loss=stop_loss,
                            take_profit=take_profit
                        )
                        self.current_position = signal
                        self.entry_price = entry_price
                        logger.info(f"Entered {signal.upper()} trade at {entry_price}, SL: {stop_loss}, TP: {take_profit}")
                    else:
                        logger.info("No new signal generated, waiting.")

            except Exception as e:
                logger.error("Error in trading logic", error=str(e))
            # Sleep until the next 10-minute bar (600 seconds)
            await asyncio.sleep(600)

    async def check_open_trade(self) -> bool:
        # We use our internal flag to indicate an open position.
        # In a production bot, you might query your exchange to verify.
        return self.current_position is not None

    async def exit_trade(self, exit_price: float):
        # Exit the trade by placing a market order in the opposite direction.
        exit_side = "short" if self.current_position == "long" else "long"
        await self.execute_trade(
            side=exit_side,
            qty=None,
            entry_price=exit_price
        )
        logger.info(f"Exited {self.current_position.upper()} trade at {exit_price}")
        self.current_position = None
        self.entry_price = None

    async def execute_trade(self, side: str, qty: str = None, entry_price: float = None,
                            stop_loss: float = None, take_profit: float = None, leverage: str = "1"):
        """
        Executes a market order. For simplicity, this example uses the position size
        from the configuration (Config.TRADING_CONFIG['position_size']). In production,
        you might also compute position size dynamically.
        """
        position_size = Config.TRADING_CONFIG.get("position_size", 1.0)
        if position_size < self.min_qty:
            position_size = self.min_qty

        logger.info("Placing order", side=side, size=position_size,
                    entry_price=entry_price, stop_loss=stop_loss, take_profit=take_profit)

        order_params = {
            "category": Config.BYBIT_CONFIG.get("category", "inverse"),
            "symbol": Config.TRADING_CONFIG["symbol"],
            "side": "Buy" if side == "long" else "Sell",
            "orderType": "Market",
            "qty": str(position_size),
            "stopLoss": str(stop_loss) if stop_loss is not None else None,
            "takeProfit": str(take_profit) if take_profit is not None else None,
            "leverage": leverage
        }
        # Remove any keys with None values
        order_params = {k: v for k, v in order_params.items() if v is not None}
        try:
            response = await asyncio.to_thread(self.session.place_order, **order_params)
            if response.get('retCode', -1) == 0:
                logger.info("Trade executed successfully", order_id=response["result"].get("orderId"))
            else:
                logger.error("Trade execution failed", response=response)
        except Exception as e:
            logger.error("Trade execution error", error=str(e))

    async def get_portfolio_value(self) -> float:
        try:
            balance_data = await asyncio.to_thread(self.session.get_wallet_balance, accountType="UNIFIED")
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

    def stop(self):
        self.running = False
        logger.info("TradeService stopped.")
