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

def get_market_regime(df, adx_period=14, threshold=25):
    """
    Compute the ADX using the high, low, and close values in the DataFrame.
    Returns "trending" if the latest ADX exceeds the threshold, otherwise "sideways".
    """
    try:
        adx_indicator = ta.trend.ADXIndicator(
            high=df["high"],
            low=df["low"],
            close=df["close"],
            window=adx_period
        )
        adx_series = adx_indicator.adx()
        if adx_series.empty:
            return "sideways"
        latest_adx = adx_series.iloc[-1]
        logger.info("Market regime ADX", adx=latest_adx)
        return "trending" if latest_adx > threshold else "sideways"
    except Exception as e:
        logger.error("Error computing market regime", error=str(e))
        return "sideways"

class TradeService:
    def __init__(self):
        # Initialize MLService with your desired lookback (e.g., 60)
        self.ml_service = MLService(lookback=60)
        self.session = None
        self.min_qty = None            # Minimum order quantity (in USD notional)
        self.current_position = None   # Current trade direction ("buy" or "sell")
        self.current_order_qty = None  # Order quantity of the open trade
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

    def compute_sl_tp_dynamic(self, current_price: float, atr_value: float, signal: str,
                              market_regime: str, risk_multiplier=1.5, reward_ratio=3.0):
        """
        Compute dynamic stop loss and take profit levels based on ATR, with adjustments
        based on the market regime.
        
        In trending markets, we may allow wider SL and a more aggressive TP.
        In sideways markets, we tighten SL and target a lower reward.
        """
        if market_regime == "trending":
            risk = risk_multiplier * abs(atr_value)
            adjusted_reward_ratio = reward_ratio
        else:  # sideways
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
        """
        Compute a trailing stop level based on ATR.
        
        - trail_multiplier: Multiplies ATR to set the trailing stop distance.
        """
        trail_distance = trail_multiplier * abs(atr_value)
        if signal.lower() == "buy":
            trailing_stop = current_price - trail_distance
        elif signal.lower() == "sell":
            trailing_stop = current_price + trail_distance
        else:
            trailing_stop = None
        return trailing_stop

    async def exit_trade(self):
        """
        Exit the currently open trade by placing a market order with the opposite side.
        This method assumes that executing an order in the opposite direction will close the position.
        """
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
            leverage=10.0
        )
        self.current_position = None
        self.current_order_qty = None

    async def run_trading_logic(self):
        """
        Execute the trading logic:
        
        1. Fetch recent 1-minute candle data from the 'candles' table.
        2. If fewer than 900 rows are available, trigger backfill.
        3. Ensure the 1-minute data is continuous.
        4. Resample the 1-minute data to 15-minute candles.
        5. Forward fill missing values and drop rows missing the 'close' value.
        6. Log details of the 1-minute and 15-minute data.
        7. Compute technical indicators (RSI, MACD, Bollinger Bands, ATR) on the 15-minute data.
        8. Determine market regime using ADX.
        9. Generate an ML prediction signal using the 15-minute data.
        10. Compute support/resistance from the most recent 15-minute candle.
        11. Check for open trades using internal state first:
             - If an internal position exists and it differs from the new signal, exit the current trade.
             - Otherwise, if an internal position exists and matches the new signal, skip trade entry.
             - If no internal state, check the exchange.
        12. Use fixed leverage of 10x.
        13. Compute effective trade value and order quantity.
        14. Compute dynamic SL and TP using the ATR-based risk/reward strategy and market regime.
        15. Compute the initial trailing stop level.
        16. Override exchange default leverage and execute the trade.
        17. Store details of the new open trade.
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
            
            # 4. Resample to 15-minute candles.
            df_15 = df_1min.resample('15min').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            })
            
            # 5. Forward fill missing values and drop rows missing the 'close' value.
            df_15 = df_15.ffill().dropna(subset=["close"])
            
            # 6. Log details.
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

            # 8. Determine market regime using ADX.
            market_regime = get_market_regime(df_15)
            logger.info("Market regime detected", regime=market_regime)

            # 9. Generate ML prediction signal using the 15-minute data.
            signal = self.ml_service.predict_signal(df_15)
            logger.info("15-minute ML prediction", signal=signal)
            if signal.lower() == "hold":
                return

            # 10. Compute support/resistance from the most recent 15-minute candle.
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

            # 11. Check for open trades using internal state first.
            if self.current_position is not None:
                if self.current_position.lower() != signal.lower():
                    logger.info("Signal reversal detected (internal state). Exiting current trade before new trade.")
                    await self.exit_trade()
                    await asyncio.sleep(1)
                else:
                    logger.info("Existing position matches new signal; skipping new trade.")
                    return
            else:
                # If internal state is not set, check the exchange.
                if await self.check_open_trade():
                    logger.info("Exchange indicates an open trade but no internal state; skipping new trade.")
                    return

            current_price = last_candle["close"]

            # 12. Use fixed leverage of 10x.
            fixed_leverage = 10.0
            logger.info("Using fixed leverage", leverage=fixed_leverage)

            # 13. Compute effective trade value and order quantity.
            portfolio_value = await self.get_portfolio_value()
            if portfolio_value <= 0:
                logger.warning("Portfolio value is 0; cannot size position.")
                return
            effective_trade_value = round(portfolio_value * 0.02 * fixed_leverage)
            order_qty = effective_trade_value
            order_qty = max(order_qty, self.min_qty)

            # 14. Compute dynamic SL and TP using ATR-based risk/reward strategy and market regime.
            stop_loss, take_profit = self.compute_sl_tp_dynamic(
                current_price, df_15["atr"].iloc[-1], signal, market_regime,
                risk_multiplier=1.5, reward_ratio=3.0
            )
            
            # 15. Compute the initial trailing stop level.
            trailing_stop = self.compute_trailing_stop(
                current_price, df_15["atr"].iloc[-1], signal, trail_multiplier=1.5
            )
            logger.info("Computed trailing stop", trailing_stop=trailing_stop)

            logger.info("Trade parameters computed", current_price=current_price,
                        effective_trade_value=effective_trade_value, order_qty=order_qty,
                        stop_loss=stop_loss, take_profit=take_profit, leverage=fixed_leverage)

            # 16. Override the exchange leverage.
            await self.set_trade_leverage(fixed_leverage)

            # 17. Execute the trade.
            await self.execute_trade(
                side=signal,
                qty=order_qty,
                stop_loss=stop_loss,
                take_profit=take_profit,
                leverage=fixed_leverage
            )

            # 18. Store details of the new open trade.
            self.last_trade_time = datetime.now(timezone.utc)
            self.current_position = signal.lower()
            self.current_order_qty = order_qty

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
