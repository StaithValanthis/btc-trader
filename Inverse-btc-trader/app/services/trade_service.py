import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="ta.trend")

import asyncio
import math
import json
import pandas as pd
import numpy as np
import ta
from datetime import datetime, timezone
from pybit.unified_trading import HTTP
from structlog import get_logger

from app.core import Database, Config
from app.services.ml_service import MLService
from app.services.backfill_service import maybe_backfill_candles
from app.utils.cache import redis_client

logger = get_logger(__name__)

def enhanced_get_market_regime(df, adx_period=14, threshold=25):
    """
    Example market-regime detection logic. (Unchanged from original code)
    """
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

        if latest_adx > threshold and (
            (ma_trend == "up" and df_clean['close'].iloc[-1] > df_clean['SMA20'].iloc[-1]) or
            (ma_trend == "down" and df_clean['close'].iloc[-1] < df_clean['SMA20'].iloc[-1])
        ):
            if atr_percentile > 0.5 and bb_width_percentile > 0.5:
                regime = "trending"
            else:
                regime = "sideways"
        else:
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
        self.ml_service = MLService(lookback=60)
        self.session = None
        self.min_qty = None
        self.current_position = None   # "buy" or "sell"
        self.current_order_qty = None
        self.last_trade_time = None
        self.running = False

        # NEW: Track fill price & trailing-stop status
        self.entry_price = None
        self.trailing_stop_set = False

    async def initialize(self):
        """
        Initializes the trade service, sets up Bybit session, gets min_qty, starts ML retrain + trailing-stop tasks.
        """
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

        # Kick off ML retrain & trailing-stop tasks
        asyncio.create_task(self.ml_service.schedule_daily_retrain())
        asyncio.create_task(self.update_trailing_stop_task())
        self.running = True

    async def _get_min_order_qty(self):
        """
        Fetch minimum order quantity from Bybit instruments info.
        """
        if not self.min_qty:
            info = await asyncio.to_thread(
                self.session.get_instruments_info,
                category=Config.BYBIT_CONFIG.get('category', 'inverse'),
                symbol=Config.TRADING_CONFIG['symbol']
            )
            if info.get("retCode", -1) != 0:
                logger.error("Failed to fetch instrument info", data=info)
                raise RuntimeError("Error fetching instrument info from Bybit.")

            # Bybit v5 "inverse" => info['result']['list'][0]['lotSizeFilter']['minOrderQty']
            self.min_qty = float(info['result']['list'][0]['lotSizeFilter']['minOrderQty'])
            logger.info("Loaded min order qty", min_qty=self.min_qty)

    async def check_open_trade(self) -> bool:
        """
        Returns True if there's an existing open position (size != 0).
        """
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

    async def exit_trade(self):
        """
        Closes any current position by taking the opposite side in the same quantity.
        """
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
            leverage="5",       # or whatever you prefer
            trailing_stop=None
        )

        # Reset internal state
        self.current_position = None
        self.current_order_qty = None
        self.entry_price = None
        self.trailing_stop_set = False

    async def run_trading_logic(self):
        """
        Main method to be called periodically (e.g. every minute).
        1) Gathers recent candle data
        2) ML prediction
        3) If new trade signal, compute 3x portfolio - 1, place new trade
        4) If existing position conflicts with new signal, exit
        """
        if not self.running:
            return

        # Pull recent 1-min candles from cache or DB
        df_1min = await self._get_recent_1min_candles()
        if df_1min is None or len(df_1min) < 60:
            logger.warning("Not enough 1-minute candle data for prediction yet.")
            return

        # Convert to 15-min for ML
        df_15 = df_1min.resample('15min').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).ffill().dropna()
        if len(df_15) < 14:
            logger.warning("Not enough 15-minute candle data for ML prediction.")
            return

        # Optionally detect regime (not strictly required for your trailing stop)
        market_regime = enhanced_get_market_regime(df_15)
        logger.info("Market regime detected", regime=market_regime)

        # Predict signal via ML
        signal = self.ml_service.predict_signal(df_15)
        logger.info("15-minute ML prediction", signal=signal)
        if signal.lower() not in ["buy", "sell"]:
            return

        # Pivot/resistance check (unchanged from your code)
        last_candle = df_15.iloc[-1]
        pivot = (last_candle["high"] + last_candle["low"] + last_candle["close"]) / 3
        resistance_threshold = last_candle["close"] * 1.01
        support_threshold = last_candle["close"] * 0.99
        if signal.lower() == "buy" and last_candle["close"] > resistance_threshold:
            logger.info("15-minute price near resistance; skipping Buy trade.")
            return
        if signal.lower() == "sell" and last_candle["close"] < support_threshold:
            logger.info("15-minute price near support; skipping Sell trade.")
            return

        # Handle existing positions
        if await self.check_open_trade():
            # If we do have an open position but the new signal is opposite, exit first
            if self.current_position and self.current_position.lower() != signal.lower():
                logger.info("Signal reversal detected. Exiting current trade before new trade.")
                await self.exit_trade()
                await asyncio.sleep(1)
            else:
                logger.info("Existing position matches new signal; skipping new trade.")
                return
        else:
            # No open position found at the exchange
            self.current_position = None
            self.current_order_qty = None

        # ============ New position sizing: 3x portfolio - 1 ============
        portfolio_value = await self.get_portfolio_value()
        if portfolio_value <= 0:
            logger.warning("Portfolio value is 0; cannot size position.")
            return

        raw_amount = 3 * portfolio_value
        entry_value = round(raw_amount) - 1  # nearest dollar, then minus 1
        if entry_value < 1:
            logger.warning(f"Calculated entry_value is too small: {entry_value}")
            return

        # Example: Set a fixed leverage if you like
        leverage = 5
        await self.set_trade_leverage(leverage)

        # Execute the trade (no trailing stop, no SL/TP yet)
        await self.execute_trade(
            side=signal,
            qty=str(entry_value),
            stop_loss=None,
            take_profit=None,
            leverage=str(leverage),
            trailing_stop=None
        )

        # Record internal state
        self.last_trade_time = datetime.now(timezone.utc)
        self.current_position = signal.lower()
        self.current_order_qty = entry_value
        self.trailing_stop_set = False  # Will be set by update_trailing_stop_task once +5%

    async def _get_recent_1min_candles(self):
        """
        Pulls 1-min candles from Redis if cached, else from the DB, up to ~1800 bars.
        """
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
            if not rows:
                return None
            df_1min = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close", "volume"])
            df_1min['time'] = pd.to_datetime(df_1min['time'])
            df_1min.sort_values("time", inplace=True)
            df_1min.set_index("time", inplace=True)
            redis_client.setex("recent_candles", 60, df_1min.reset_index().to_json(orient="records"))

        df_1min = df_1min.asfreq("1min")
        return df_1min

    async def update_trailing_stop_task(self):
        """
        Runs periodically to check if position is +5% in profit.
        If so, sets a 3% trailing stop once (for either a long or short).
        """
        while self.running:
            try:
                if (
                    self.current_position
                    and not self.trailing_stop_set
                    and self.entry_price
                ):
                    current_price = await self._fetch_current_price()
                    if self.current_position == "buy":
                        # +5% => current_price >= entry_price * 1.05
                        if current_price >= self.entry_price * 1.05:
                            # trailing offset = 3% of current price
                            offset = round(current_price * 0.03, 2)
                            await self.set_trailing_stop(offset)
                            self.trailing_stop_set = True
                    elif self.current_position == "sell":
                        # +5% (for a short) => current_price <= entry_price * 0.95
                        if current_price <= self.entry_price * 0.95:
                            offset = round(current_price * 0.03, 2)
                            await self.set_trailing_stop(offset)
                            self.trailing_stop_set = True

            except Exception as ex:
                logger.error("update_trailing_stop_task error", error=str(ex))

            await asyncio.sleep(60)  # adjust as needed

    async def set_trailing_stop(self, offset: float):
        """
        Bybit v5 Inverse trailingStop is an absolute offset in USD.
        We set 'trailingStop' param to e.g. '100' => 100 USD behind market.
        """
        params = {
            "category": Config.BYBIT_CONFIG.get("category", "inverse"),
            "symbol": Config.TRADING_CONFIG['symbol'],
            "trailingStop": str(offset),
            "positionIdx": 0
        }
        try:
            response = await asyncio.to_thread(self.session.set_trading_stop, **params)
            logger.info("Trailing stop set", data=response)
        except Exception as e:
            logger.error("Error setting trailing stop", error=str(e))

    async def execute_trade(self, side: str, qty: str,
                            stop_loss: float = None, take_profit: float = None,
                            leverage: str = None, trailing_stop: float = None):
        """
        Places a Market order. We parse the fill price & store in self.entry_price.
        """
        if not self.running:
            return
        if float(qty) < self.min_qty:
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
            "category": Config.BYBIT_CONFIG.get("category", "inverse"),
            "symbol": Config.TRADING_CONFIG['symbol'],
            "side": side,
            "orderType": "Market",
            "qty": qty,
        }
        if leverage:
            order_params["leverage"] = leverage

        try:
            response = await asyncio.to_thread(self.session.place_order, **order_params)
            if response['retCode'] == 0:
                logger.info("Trade executed successfully", order_id=response['result']['orderId'])

                # Attempt to parse fill price from response
                fill_price = None
                try:
                    # Bybit typically puts fill info in response['result']['list']
                    fill_price = float(response['result']['list'][0]['execPrice'])
                except Exception:
                    pass

                if fill_price:
                    self.entry_price = fill_price
                    logger.info("Recorded entry price", entry_price=fill_price)
                else:
                    logger.warning("Could not parse fill_price from Bybit response")
                    self.entry_price = None

                # Log the trade in DB
                await self._log_trade(side, qty)
            else:
                error_msg = response.get('retMsg', 'Unknown error')
                logger.error("Trade failed", error=error_msg)
                # Reset if order fails
                self.current_position = None
                self.current_order_qty = None
                self.entry_price = None
                self.trailing_stop_set = False

        except Exception as e:
            logger.error("Trade execution error", error=str(e))
            self.current_position = None
            self.current_order_qty = None
            self.entry_price = None
            self.trailing_stop_set = False

    async def _fetch_current_price(self) -> float:
        """
        Fetches latest price from Bybit's instruments info. 
        Or you could query your own market_data table.
        """
        resp = await asyncio.to_thread(
            self.session.get_instruments_info,
            category=Config.BYBIT_CONFIG.get('category', 'inverse'),
            symbol=Config.TRADING_CONFIG['symbol']
        )
        last_price = float(resp['result']['list'][0]['lastPrice'])
        return last_price

    async def set_trade_leverage(self, leverage: int):
        """
        Set leverage if category != 'inverse' or if your inverse contract supports it.
        """
        if Config.BYBIT_CONFIG.get("category", "inverse") == "inverse":
            # Bybit inverse might not strictly require set_leverage, but we show it here.
            logger.info("Setting leverage on inverse (if supported)", leverage=leverage)
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

    async def get_portfolio_value(self) -> float:
        """
        Fetch wallet balances. Sum up USD value of relevant coins. 
        For an inverse BTCUSD scenario, we often check the BTC coin's 'usdValue'.
        """
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
        """
        Simple DB logger for executed trades.
        """
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
