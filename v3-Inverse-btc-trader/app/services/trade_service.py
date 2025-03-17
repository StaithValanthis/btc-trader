# File: v2-Inverse-btc-trader/app/services/trade_service.py

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
from app.utils.cache import redis_client
from app.services.backfill_service import maybe_backfill_candles
from app.ml.ml_inference import TrendModelService, SignalModelService

logger = get_logger(__name__)

def enhanced_get_market_regime(df: pd.DataFrame, adx_period: int=14, base_threshold: float=25) -> str:
    """
    Determines market regime based on ADX and moving average crossover + volatility checks.
    Returns 'uptrending', 'downtrending', or 'sideways'.
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
        sma_diff = df_clean['SMA20'].iloc[-1] - df_clean['SMA50'].iloc[-1]
        ma_trend = "up" if sma_diff > 0 else "down"

        atr_indicator = ta.volatility.AverageTrueRange(
            high=df_clean["high"], low=df_clean["low"], close=df_clean["close"],
            window=14
        )
        atr_series = atr_indicator.average_true_range().dropna()
        atr_percentile = atr_series.rank(pct=True).iloc[-1] if not atr_series.empty else 0

        adaptive_threshold = base_threshold * (1 - 0.2 * atr_percentile)

        boll = ta.volatility.BollingerBands(
            close=df_clean["close"], window=20, window_dev=2
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
    """
    Orchestrates actual trading, using loaded ML models. 
    Handles position sizing, risk management, and Bybit REST calls.
    """

    def __init__(self) -> None:
        self.trend_service = TrendModelService(lookback=120)
        self.signal_service = SignalModelService(lookback=120)
        self.session = None
        self.min_qty = None
        self.current_position = None
        self.current_order_qty = None
        self.last_trade_time = None
        self.running = False
        self.scaled_in = False

    async def initialize(self) -> None:
        """
        Initialize trade service, session, and ML models.
        """
        try:
            self.session = HTTP(
                testnet=Config.BYBIT_CONFIG['testnet'],
                api_key=Config.BYBIT_CONFIG['api_key'],
                api_secret=Config.BYBIT_CONFIG['api_secret'],
                recv_window=5000
            )
            await self._get_min_order_qty()

            # Load or fallback to training if no models found
            trend_loaded = self.trend_service.load()
            signal_loaded = self.signal_service.load()
            if not trend_loaded or not signal_loaded:
                logger.info("One or more ML models not found; you could trigger advanced training if needed.")
                # e.g. call an advanced training function here if you want to train on the fly

            logger.info("Trade service initialized successfully")
            asyncio.create_task(self.update_open_trade_stops_task())
            asyncio.create_task(self.run_trading_logic())
            self.running = True
        except Exception as e:
            logger.critical("Fatal error initializing trade service", error=str(e))
            raise

    async def stop(self) -> None:
        """
        Stop the TradeService tasks.
        """
        self.running = False
        logger.info("TradeService stopped.")

    async def _get_min_order_qty(self) -> None:
        """
        Fetch the minimum allowed order quantity from Bybit.
        """
        if not self.min_qty:
            info = await asyncio.to_thread(
                self.session.get_instruments_info,
                category=Config.TRADING_CONFIG.get("category", "inverse"),
                symbol=Config.TRADING_CONFIG['symbol']
            )
            self.min_qty = float(info['result']['list'][0]['lotSizeFilter']['minOrderQty'])
            logger.info("Loaded min order qty", min_qty=self.min_qty)

    async def check_open_trade(self) -> bool:
        """
        Check if there's an open position on Bybit for the configured symbol.
        """
        try:
            pos_data = await asyncio.to_thread(
                self.session.get_positions,
                category=Config.TRADING_CONFIG.get('category', 'inverse'),
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

    async def set_trade_leverage(self, leverage: int) -> None:
        """
        Set the trade leverage if using a linear category. 
        Skips for inverse (which usually doesn't require set_leverage).
        """
        cat = Config.TRADING_CONFIG.get("category", "inverse")
        if cat == "inverse":
            logger.info("Skipping set_leverage for inverse contracts.")
            return
        try:
            response = await asyncio.to_thread(
                self.session.set_leverage,
                category=cat,
                symbol=Config.TRADING_CONFIG['symbol'],
                leverage=str(leverage)
            )
            logger.info("Set leverage response", data=response)
        except Exception as e:
            logger.error("Error setting leverage", error=str(e))

    def compute_sl_tp_dynamic(
        self, 
        current_price: float,
        atr_value: float,
        signal: str,
        market_regime: str,
        base_risk_multiplier: float=1.5,
        min_reward_risk_ratio: float=2.0
    ) -> (float, float):
        """
        Compute dynamic stop-loss and take-profit using ATR and
        a base risk multiplier, adjusted by market regime if desired.

        Args:
            current_price (float): The last trade price
            atr_value (float): ATR from recent data
            signal (str): 'buy', 'sell', or 'hold'
            market_regime (str): 'uptrending', 'downtrending', 'sideways'
            base_risk_multiplier (float): multiplier for ATR
            min_reward_risk_ratio (float): minimal R:R ratio

        Returns:
            (stop_loss, take_profit) as floats
        """
        raw_risk = base_risk_multiplier * abs(atr_value)
        # Cap risk at 10% of current price
        risk = min(raw_risk, 0.10 * current_price)

        if signal.lower() == "buy":
            stop_loss = current_price - risk
            take_profit = current_price + risk * max(min_reward_risk_ratio, 2.0)
        elif signal.lower() == "sell":
            stop_loss = current_price + risk
            take_profit = current_price - risk * max(min_reward_risk_ratio, 2.0)
        else:
            stop_loss = current_price
            take_profit = current_price
        return stop_loss, take_profit

    async def update_open_trade_stops_task(self) -> None:
        """
        Background task to periodically update stops on any open trade.
        """
        while self.running:
            if self.current_position is not None:
                try:
                    recent_df = await self.get_recent_candles()
                    df_15 = recent_df.resample('15min').agg({
                        'open': 'first',
                        'high': 'max',
                        'low': 'min',
                        'close': 'last',
                        'volume': 'sum'
                    }).ffill().dropna()
                    if len(df_15) < 14:
                        await asyncio.sleep(60)
                        continue

                    atr_indicator = ta.volatility.AverageTrueRange(
                        high=df_15["high"],
                        low=df_15["low"],
                        close=df_15["close"],
                        window=14
                    )
                    df_15["atr"] = atr_indicator.average_true_range()

                    current_price = recent_df.iloc[-1]["close"]
                    market_regime = enhanced_get_market_regime(df_15)
                    stop_loss, take_profit = self.compute_sl_tp_dynamic(
                        current_price, df_15["atr"].iloc[-1],
                        self.current_position, market_regime
                    )
                    await self.update_trade_stops(stop_loss, take_profit)
                except Exception as e:
                    logger.error("Error updating open trade stops", error=str(e))
            await asyncio.sleep(60)

    async def update_trade_stops(self, stop_loss: float, take_profit: float) -> None:
        """
        Update stop loss & take profit for an open position.
        """
        params = {
            "category": Config.TRADING_CONFIG.get("category", "inverse"),
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

    async def run_trading_logic(self) -> None:
        """
        Main trading logic loop. Periodically fetches data, calls ML predictions,
        checks confluence, and places trades if needed.
        """
        while self.running:
            try:
                df_1min = await self._fetch_recent_candles_cached()
                if df_1min is None or len(df_1min) < 60:
                    await asyncio.sleep(60)
                    continue

                df_15 = df_1min.resample('15min').agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }).ffill().dropna()

                if len(df_15) < 14:
                    await asyncio.sleep(60)
                    continue

                atr_indicator = ta.volatility.AverageTrueRange(
                    high=df_15["high"], low=df_15["low"], close=df_15["close"], window=14
                )
                df_15["atr"] = atr_indicator.average_true_range()

                market_regime = enhanced_get_market_regime(df_15)
                logger.info("Market regime", regime=market_regime)
                if market_regime.lower() == "sideways":
                    logger.info("Market is sideways; skipping trade entry.")
                    await asyncio.sleep(60)
                    continue

                # Use the ML inference
                trend_prediction = self.trend_service.predict_trend(df_15)
                signal_prediction = self.signal_service.predict_signal(df_1min)

                logger.info("Trend prediction", trend=trend_prediction)
                logger.info("Signal prediction", signal=signal_prediction)

                # Confluence check
                if trend_prediction.lower() == "uptrending" and signal_prediction.lower() != "buy":
                    logger.info("Trend up but signal not Buy. Holding.")
                    await asyncio.sleep(60)
                    continue
                if trend_prediction.lower() == "downtrending" and signal_prediction.lower() != "sell":
                    logger.info("Trend down but signal not Sell. Holding.")
                    await asyncio.sleep(60)
                    continue

                if await self.check_open_trade():
                    if self.current_position and self.current_position.lower() != signal_prediction.lower():
                        logger.info("Signal reversal detected. Exiting current trade.")
                        await self.exit_trade()
                        await asyncio.sleep(2)
                    else:
                        logger.info("Existing position matches new signal; skipping.")
                        await asyncio.sleep(60)
                        continue
                else:
                    self.current_position = None
                    self.current_order_qty = None

                current_price = df_1min.iloc[-1]["close"]
                atr_value = df_15["atr"].iloc[-1]
                if atr_value <= 0:
                    await asyncio.sleep(60)
                    continue

                stop_loss, _ = self.compute_sl_tp_dynamic(
                    current_price, atr_value, signal_prediction, market_regime
                )
                stop_distance = abs(current_price - stop_loss)
                if stop_distance == 0:
                    await asyncio.sleep(60)
                    continue

                portfolio_value = await self.get_portfolio_value()
                if portfolio_value <= 0:
                    await asyncio.sleep(60)
                    continue

                risk_per_trade = 0.01 * portfolio_value
                computed_qty = (risk_per_trade / stop_distance) * current_price
                order_qty = max(computed_qty, self.min_qty)
                order_qty = round(order_qty)
                logger.info("Computed position size",
                            portfolio_value=portfolio_value,
                            risk_per_trade=risk_per_trade,
                            stop_distance=stop_distance,
                            computed_qty=computed_qty,
                            final_qty=order_qty)

                await self.set_trade_leverage(1)
                final_stop, final_tp = self.compute_sl_tp_dynamic(
                    current_price, atr_value, signal_prediction, market_regime
                )

                await self.execute_trade(
                    side=signal_prediction,
                    qty=str(order_qty),
                    stop_loss=final_stop,
                    take_profit=final_tp,
                    leverage="1"
                )
                self.last_trade_time = datetime.now(timezone.utc)
                self.current_position = signal_prediction.lower()
                self.current_order_qty = order_qty

            except Exception as e:
                logger.error("Error in trading logic", error=str(e))
            await asyncio.sleep(60)

    async def execute_trade(
        self,
        side: str,
        qty: str,
        stop_loss: float = None,
        take_profit: float = None,
        leverage: str = None
    ) -> None:
        """
        Place a market order with optional stop_loss / take_profit, then log the trade.
        Attempt to capture fill price from the API response if available.

        Args:
            side (str): 'Buy' or 'Sell'
            qty (str): Order quantity
            stop_loss (float): Stop loss price
            take_profit (float): Take profit price
            leverage (str): Leverage to use (if linear)
        """
        if not self.running:
            return
        if float(qty) < self.min_qty:
            qty = str(self.min_qty)

        logger.info("Placing trade", side=side, size=qty, stop_loss=stop_loss, take_profit=take_profit, leverage=leverage)

        order_params = {
            "category": Config.TRADING_CONFIG.get("category", "inverse"),
            "symbol": Config.TRADING_CONFIG['symbol'],
            "side": side.capitalize(),
            "orderType": "Market",
            "qty": qty
        }
        if stop_loss is not None:
            order_params["stopLoss"] = str(stop_loss)
        if take_profit is not None:
            order_params["takeProfit"] = str(take_profit)
        if leverage is not None:
            order_params["leverage"] = leverage

        try:
            response = await asyncio.to_thread(self.session.place_order, **order_params)
            if response['retCode'] == 0:
                if "retExtInfo" in response and "errCode" in response["retExtInfo"]:
                    err_code = response["retExtInfo"]["errCode"]
                    if err_code == "130013":
                        logger.error("Insufficient margin to place trade!")
                        return

                result = response.get('result', {})
                fill_price = 0.0
                fill_qty = float(qty)
                if "lastPrice" in result:
                    fill_price = float(result["lastPrice"])

                logger.info("Trade executed successfully", order_id=result.get('orderId'))
                await self._log_trade(side, fill_price, fill_qty)
            else:
                error_msg = response.get('retMsg', 'Unknown error')
                logger.error("Trade failed", error=error_msg, retCode=response['retCode'])
        except Exception as e:
            logger.error("Trade execution error", error=str(e))
            self.current_position = None

    async def _log_trade(self, side: str, price: float, qty: float) -> None:
        """
        Log executed trade to 'trades' table with actual fill price & quantity if possible.
        """
        if Database._pool is None:
            logger.warning("Database is closed; skipping trade logging.")
            return
        try:
            await Database.execute('''
                INSERT INTO trades (time, side, price, quantity)
                VALUES ($1, $2, $3, $4)
            ''', datetime.now(timezone.utc), side, price, qty)
        except Exception as e:
            logger.error("Failed to log trade", error=str(e))

    async def exit_trade(self) -> None:
        """
        Exit any current position by taking the opposite side.
        """
        if not self.current_position:
            logger.warning("No current position to exit.")
            return
        exit_side = "Buy" if self.current_position.lower() == "sell" else "Sell"
        order_qty = self.current_order_qty or self.min_qty
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

    async def get_portfolio_value(self) -> float:
        """
        Fetch the total USD value of the BTC balance from the Bybit wallet.
        """
        try:
            balance_data = await asyncio.to_thread(
                self.session.get_wallet_balance,
                accountType="UNIFIED"
            )
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

    async def get_recent_candles(self) -> pd.DataFrame:
        """
        Fetches the most recent ~1800 1-min candles from the DB as a DataFrame.
        """
        rows = await Database.fetch("""
            SELECT time, open, high, low, close, volume
            FROM candles
            ORDER BY time DESC
            LIMIT 1800
        """)
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close", "volume"])
        df['time'] = pd.to_datetime(df['time'])
        df.sort_values("time", inplace=True)
        df.set_index("time", inplace=True)
        return df.asfreq("1min")

    async def _fetch_recent_candles_cached(self) -> pd.DataFrame:
        """
        Helper to check Redis cache for recent candles (1-min). If not found, queries DB.
        """
        cached = redis_client.get("recent_candles")
        if cached:
            data = json.loads(cached)
            df = pd.DataFrame(data)
            df['time'] = pd.to_datetime(df['time'])
            df.sort_values("time", inplace=True)
            df.set_index("time", inplace=True)
            return df.asfreq("1min")
        else:
            df = await self.get_recent_candles()
            if not df.empty:
                # Cache for 60s
                redis_client.setex("recent_candles", 60, df.reset_index().to_json(orient="records"))
            return df
