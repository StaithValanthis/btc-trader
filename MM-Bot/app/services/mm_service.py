# File: app/services/mm_service.py
import asyncio
import math
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from structlog import get_logger
from pybit.unified_trading import HTTP

from app.core.config import Config
from app.core.database import Database

# Import tensorflow and keras_tuner for hyperparameter tuning
import tensorflow as tf
import keras_tuner as kt

logger = get_logger(__name__)

class MMService:
    """
    Adaptive Market Making service implementing the Avellaneda–Stoikov model with dynamic spread adjustment,
    advanced ML-based hyperparameter tuning, and equity-based order sizing using 3× leverage.

    When no position is held (inventory == 0), the bot posts bid and ask orders.
    Once a position is taken, it cancels MM orders and manages the open position with stop-loss (SL)
    and take-profit (TP) orders.

    Historical data is loaded using the days_to_fetch parameter (e.g., from startup_check.py).

    Key parameters:
      - gamma: risk aversion coefficient.
      - k: market order arrival decay parameter.
      - T: time horizon (in days).

    Tunable hyperparameters:
      - baseline_sigma: baseline volatility used for dynamic spread adjustment.
      - refresh_rate: interval (in seconds) to refresh orders.
      - stop_loss_pct: percentage for stop loss exit (min: 0.5% or 0.005).
      - take_profit_pct: percentage for take profit exit (min: 0.1% or 0.001).

    Order sizing is based on available equity: number of contracts = round(equity × leverage × 0.35)
    with fixed leverage = 3×. If the computed qty is less than the exchange minimum, the minimum is enforced.

    A minimum effective spread is enforced in both live trading and simulation.
    In the simulation, if the computed bid/ask spread is less than 0.1% of the midprice,
    the simulation returns an extreme loss so that the tuner only considers parameter sets
    that yield a spread equal to or above 0.1% of the current price.
    """
    def __init__(self, risk_aversion=0.1, k=1.0, T=1.0, days_to_fetch=7):
        self.gamma = risk_aversion         # risk aversion coefficient (γ)
        self.k = k                         # market order arrival decay parameter
        self.T = T                         # time horizon in days
        self.days_to_fetch = days_to_fetch # number of days of historical data to load
        self.sigma = None                  # estimated daily volatility (σ)
        self.inventory = 0                 # current net inventory; update externally
        self.entry_price = None            # entry price when a position is taken
        self.position_qty = None           # number of contracts held in open position
        self.session = HTTP(
            testnet=Config.BYBIT_CONFIG['testnet'],
            api_key=Config.BYBIT_CONFIG['api_key'],
            api_secret=Config.BYBIT_CONFIG['api_secret']
        )
        self.current_bid_order_id = None
        self.current_ask_order_id = None
        self.running = False
        self._historical_df = None         # Historical candle data for tuning

        # Default tunable parameters (to be updated by ML tuning)
        self.baseline_sigma = 0.02
        self.refresh_rate = 60
        self.stop_loss_pct = 0.02
        self.take_profit_pct = 0.04

        # Minimum order quantity (in contract units); updated from instrument info.
        self.min_qty = None

    async def update_volatility(self, window_minutes=1440):
        try:
            query = (
                "SELECT time, close FROM candles "
                "WHERE time >= NOW() - INTERVAL '%s minutes' "
                "ORDER BY time ASC" % window_minutes
            )
            rows = await Database.fetch(query)
            if not rows or len(rows) < 10:
                logger.warning("Not enough candle data for volatility estimation")
                return
            df = pd.DataFrame(rows, columns=["time", "close"])
            df["time"] = pd.to_datetime(df["time"])
            df.set_index("time", inplace=True)
            df["log_return"] = np.log(df["close"] / df["close"].shift(1))
            df.dropna(inplace=True)
            sigma_min = df["log_return"].std()
            self.sigma = sigma_min * np.sqrt(1440)
            logger.info("Volatility updated", sigma_daily=self.sigma)
        except Exception as e:
            logger.error("Error updating volatility", error=str(e))

    def compute_bid_ask(self, midprice, inventory, baseline_sigma):
        if self.sigma is None:
            logger.warning("Volatility not set; using default sigma of 0.005")
            self.sigma = 0.005
        reservation_price = midprice - inventory * self.gamma * (self.sigma ** 2) * self.T
        risk_component = (self.gamma * (self.sigma ** 2) * self.T) / 2
        arrival_component = (1 / self.gamma) * np.log(1 + self.gamma / self.k)
        half_spread = risk_component + arrival_component
        multiplier = self.sigma / baseline_sigma
        half_spread_adjusted = half_spread * multiplier
        bid = reservation_price - half_spread_adjusted
        ask = reservation_price + half_spread_adjusted
        logger.debug("Computed bid/ask",
                     midprice=midprice,
                     reservation_price=reservation_price,
                     half_spread=half_spread,
                     multiplier=multiplier,
                     adjusted_half_spread=half_spread_adjusted,
                     bid=bid,
                     ask=ask,
                     inventory=inventory)
        return bid, ask

    async def cancel_existing_order(self, order_id):
        try:
            response = await asyncio.to_thread(
                self.session.cancel_order,
                category=Config.BYBIT_CONFIG.get("category", "inverse"),
                symbol=Config.TRADING_CONFIG["symbol"],
                orderId=order_id
            )
            logger.info("Order cancellation response", order_id=order_id, response=response)
        except Exception as e:
            logger.error("Error cancelling order", order_id=order_id, error=str(e))

    async def get_portfolio_value(self):
        try:
            balance_data = await asyncio.to_thread(
                self.session.get_wallet_balance,
                accountType="UNIFIED"
            )
            total_equity = 0.0
            for account in balance_data["result"].get("list", []):
                for coin in account.get("coin", []):
                    if coin["coin"] == "BTC":
                        usd_val = float(coin.get("usdValue", 0.0))
                        total_equity += usd_val
            logger.info("Portfolio value fetched", equity=total_equity)
            return total_equity
        except Exception as e:
            logger.error("Error getting portfolio value", error=str(e))
            return 0.0

    async def get_min_order_qty(self):
        try:
            info = await asyncio.to_thread(
                self.session.get_instruments_info,
                category=Config.BYBIT_CONFIG.get("category", "inverse"),
                symbol=Config.TRADING_CONFIG["symbol"]
            )
            if info and "result" in info and "list" in info["result"] and info["result"]["list"]:
                self.min_qty = float(info["result"]["list"][0]["lotSizeFilter"]["minOrderQty"])
                logger.info("Minimum order quantity updated", min_qty=self.min_qty)
        except Exception as e:
            logger.error("Error updating minimum order quantity", error=str(e))

    async def place_limit_order(self, side, price, qty):
        try:
            params = {
                "category": Config.BYBIT_CONFIG.get("category", "inverse"),
                "symbol": Config.TRADING_CONFIG["symbol"],
                "side": side,
                "orderType": "Limit",
                "price": str(price),
                "qty": str(qty),
                "timeInForce": "GTC",
                "leverage": "3"
            }
            response = await asyncio.to_thread(self.session.place_order, **params)
            logger.info("Limit order placed", side=side, price=price, qty=qty, response=response)
            return response
        except Exception as e:
            logger.error("Error placing limit order", side=side, price=price, qty=qty, error=str(e))
            return None

    async def update_orders(self):
        # If a position is open, manage it instead.
        if self.inventory != 0:
            logger.info("Position open; managing open position with SL/TP orders.")
            await self.manage_open_position()
            return
        try:
            await self.update_volatility(window_minutes=1440)
            if self.min_qty is None:
                await self.get_min_order_qty()
            latest_query = "SELECT close FROM candles ORDER BY time DESC LIMIT 1"
            latest_price = await Database.fetchval(latest_query)
            if latest_price is None:
                logger.warning("Could not fetch latest price.")
                return
            midprice = float(latest_price)
            bid_price, ask_price = self.compute_bid_ask(midprice, self.inventory, self.baseline_sigma)
            # Enforce that the effective spread (ask - bid) is at least 0.1% of the midprice.
            min_delta = midprice * 0.001  # 0.1% of the current price
            effective_spread = ask_price - bid_price
            if effective_spread < min_delta:
                bid_price = midprice - (min_delta / 2)
                ask_price = midprice + (min_delta / 2)
                logger.info("Forcing minimum delta spread", bid_price=bid_price, ask_price=ask_price)
            equity = await self.get_portfolio_value()
            if equity <= 0:
                logger.warning("No equity available for order sizing.")
                return
            leverage = 3
            qty = round(equity * leverage * 0.35)
            if self.min_qty is not None and qty < self.min_qty:
                qty = int(self.min_qty)
            # Cancel any existing MM orders.
            if self.current_bid_order_id:
                await self.cancel_existing_order(self.current_bid_order_id)
                self.current_bid_order_id = None
            if self.current_ask_order_id:
                await self.cancel_existing_order(self.current_ask_order_id)
                self.current_ask_order_id = None
            bid_response = await self.place_limit_order("Buy", bid_price, qty)
            ask_response = await self.place_limit_order("Sell", ask_price, qty)
            if bid_response and bid_response.get("result", {}).get("orderId"):
                self.current_bid_order_id = bid_response["result"]["orderId"]
            if ask_response and ask_response.get("result", {}).get("orderId"):
                self.current_ask_order_id = ask_response["result"]["orderId"]
            logger.info("Updated MM orders", bid_price=bid_price, ask_price=ask_price, order_qty=qty)
        except Exception as e:
            logger.error("Error updating MM orders", error=str(e))

    async def manage_open_position(self):
        # Cancel any outstanding MM orders.
        if self.current_bid_order_id:
            await self.cancel_existing_order(self.current_bid_order_id)
            self.current_bid_order_id = None
        if self.current_ask_order_id:
            await self.cancel_existing_order(self.current_ask_order_id)
            self.current_ask_order_id = None
        if self.entry_price is None or self.position_qty is None:
            logger.warning("Position details missing; cannot manage SL/TP orders.")
            return
        midprice = float(await Database.fetchval("SELECT close FROM candles ORDER BY time DESC LIMIT 1"))
        if self.inventory > 0:
            sl_price = self.entry_price * (1 - self.stop_loss_pct)
            tp_price = self.entry_price * (1 + self.take_profit_pct)
            exit_side = "Sell"
        elif self.inventory < 0:
            sl_price = self.entry_price * (1 + self.stop_loss_pct)
            tp_price = self.entry_price * (1 - self.take_profit_pct)
            exit_side = "Buy"
        else:
            return
        qty = self.position_qty
        if self.min_qty is not None and qty < self.min_qty:
            qty = int(self.min_qty)
        sl_response = await self.place_limit_order(exit_side, sl_price, qty)
        tp_response = await self.place_limit_order(exit_side, tp_price, qty)
        logger.info("Placed SL and TP orders", sl_price=sl_price, tp_price=tp_price, qty=qty)

    async def load_historical_data(self):
        try:
            query = (
                "SELECT time, open, high, low, close FROM candles "
                "WHERE time >= NOW() - INTERVAL '%d days' "
                "ORDER BY time ASC" % self.days_to_fetch
            )
            rows = await Database.fetch(query)
            if not rows:
                logger.warning("No historical data found for tuning.")
                return
            df = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close"])
            df["time"] = pd.to_datetime(df["time"])
            df.sort_values("time", inplace=True)
            df.set_index("time", inplace=True)
            self._historical_df = df
            logger.info("Historical data loaded for tuning", rows=len(df))
        except Exception as e:
            logger.error("Error loading historical data for tuning", error=str(e))

    def simulate_mm_performance(self, gamma, k, baseline_sigma, refresh_rate, stop_loss_pct, take_profit_pct):
        if self._historical_df is None or self._historical_df.empty:
            logger.error("Historical data not loaded for simulation.")
            return 1e6
        df = self._historical_df.copy()
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))
        df.dropna(inplace=True)
        sigma = df["log_return"].std() * np.sqrt(1440)
        sample_interval = max(int(refresh_rate / 60), 1)
        df_sim = df.iloc[::sample_interval].copy()
        equity = 1000.0
        position = 0  # 0: no position, 1: long, -1: short
        entry_price = None
        leverage = 3
        for idx, row in df_sim.iterrows():
            midprice = row["close"]
            risk_component = (gamma * (sigma ** 2) * self.T) / 2
            arrival_component = (1 / gamma) * np.log(1 + gamma / k)
            half_spread = risk_component + arrival_component
            multiplier = sigma / baseline_sigma
            half_spread_adjusted = half_spread * multiplier
            reservation_price = midprice - position * gamma * (sigma ** 2) * self.T
            bid = reservation_price - half_spread_adjusted
            ask = reservation_price + half_spread_adjusted
            # Enforce minimum effective spread based on percentage: minimum delta of 0.1% of midprice.
            min_delta = midprice * 0.001
            if (ask - bid) < min_delta:
                return 1e6  # Penalize parameters that yield a spread below the minimum delta.
            try:
                order_qty = round(equity * leverage * 0.35)
            except OverflowError:
                return 1e6
            if not np.isfinite(order_qty):
                return 1e6
            if position == 0:
                if row["low"] <= bid:
                    position = 1
                    entry_price = bid
                elif row["high"] >= ask:
                    position = -1
                    entry_price = ask
            elif position == 1:
                stop_loss = entry_price * (1 - stop_loss_pct)
                take_profit = entry_price * (1 + take_profit_pct)
                if row["low"] <= stop_loss:
                    profit = order_qty * (stop_loss - entry_price)
                    if not np.isfinite(profit):
                        return 1e6
                    equity += profit
                    if not np.isfinite(equity):
                        return 1e6
                    position = 0
                    entry_price = None
                elif row["high"] >= take_profit:
                    profit = order_qty * (take_profit - entry_price)
                    if not np.isfinite(profit):
                        return 1e6
                    equity += profit
                    if not np.isfinite(equity):
                        return 1e6
                    position = 0
                    entry_price = None
            elif position == -1:
                stop_loss = entry_price * (1 + stop_loss_pct)
                take_profit = entry_price * (1 - take_profit_pct)
                if row["high"] >= stop_loss:
                    profit = order_qty * (entry_price - stop_loss)
                    if not np.isfinite(profit):
                        return 1e6
                    equity += profit
                    if not np.isfinite(equity):
                        return 1e6
                    position = 0
                    entry_price = None
                elif row["low"] <= take_profit:
                    profit = order_qty * (entry_price - take_profit)
                    if not np.isfinite(profit):
                        return 1e6
                    equity += profit
                    if not np.isfinite(equity):
                        return 1e6
                    position = 0
                    entry_price = None
        if position != 0 and entry_price is not None:
            final_price = df_sim["close"].iloc[-1]
            if position == 1:
                profit = order_qty * (final_price - entry_price)
            else:
                profit = order_qty * (entry_price - final_price)
            equity += profit
        loss = -(equity - 1000.0)
        return loss

    class MMTuner(kt.RandomSearch):
        def __init__(self, simulation_fn, **kwargs):
            self.simulation_fn = simulation_fn
            super().__init__(objective="loss", max_trials=10, executions_per_trial=1, **kwargs)
        def run_trial(self, trial, *args, **kwargs):
            hp = trial.hyperparameters
            gamma = hp.Float("gamma", 0.01, 0.5, sampling="log", default=0.1)
            k = hp.Float("k", 0.1, 5.0, sampling="log", default=1.0)
            baseline_sigma = hp.Float("baseline_sigma", 0.01, 0.05, sampling="linear", default=0.02)
            refresh_rate = hp.Choice("refresh_rate", [20, 30, 60, 120], default=60)
            stop_loss_pct = hp.Float("stop_loss_pct", 0.005, 0.05, sampling="linear", default=0.02)
            take_profit_pct = hp.Float("take_profit_pct", 0.001, 0.1, sampling="linear", default=0.04)
            loss = self.simulation_fn(gamma, k, baseline_sigma, refresh_rate, stop_loss_pct, take_profit_pct)
            self.oracle.update_trial(trial.trial_id, {"loss": loss})

    async def tune_parameters_ml(self):
        await self.load_historical_data()
        def simulation_fn(gamma, k, baseline_sigma, refresh_rate, stop_loss_pct, take_profit_pct):
            return self.simulate_mm_performance(gamma, k, baseline_sigma, refresh_rate, stop_loss_pct, take_profit_pct)
        tuner = self.MMTuner(
            simulation_fn=simulation_fn,
            directory="mm_tuner_dir",
            project_name="mm_parameter_tuning"
        )
        tuner.search(x=np.zeros((1, 1)), y=np.zeros((1,)))
        best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
        self.gamma = best_hp.get("gamma")
        self.k = best_hp.get("k")
        self.baseline_sigma = best_hp.get("baseline_sigma")
        self.refresh_rate = best_hp.get("refresh_rate")
        self.stop_loss_pct = best_hp.get("stop_loss_pct")
        self.take_profit_pct = best_hp.get("take_profit_pct")
        logger.info("ML-based parameter tuning complete",
                    best_gamma=self.gamma,
                    best_k=self.k,
                    baseline_sigma=self.baseline_sigma,
                    refresh_rate=self.refresh_rate,
                    stop_loss_pct=self.stop_loss_pct,
                    take_profit_pct=self.take_profit_pct)

    async def run(self, update_interval=None):
        self.running = True
        await self.tune_parameters_ml()
        interval = update_interval if update_interval is not None else self.refresh_rate or 60
        while self.running:
            if self.inventory != 0:
                logger.info("Position open; managing open position with SL/TP orders.")
                await self.manage_open_position()
            else:
                await self.update_orders()
            await asyncio.sleep(interval)

    async def manage_open_position(self):
        if self.current_bid_order_id:
            await self.cancel_existing_order(self.current_bid_order_id)
            self.current_bid_order_id = None
        if self.current_ask_order_id:
            await self.cancel_existing_order(self.current_ask_order_id)
            self.current_ask_order_id = None
        if self.entry_price is None or self.position_qty is None:
            logger.warning("Position details missing; cannot manage SL/TP orders.")
            return
        midprice = float(await Database.fetchval("SELECT close FROM candles ORDER BY time DESC LIMIT 1"))
        if self.inventory > 0:
            sl_price = self.entry_price * (1 - self.stop_loss_pct)
            tp_price = self.entry_price * (1 + self.take_profit_pct)
            exit_side = "Sell"
        elif self.inventory < 0:
            sl_price = self.entry_price * (1 + self.stop_loss_pct)
            tp_price = self.entry_price * (1 - self.take_profit_pct)
            exit_side = "Buy"
        else:
            return
        qty = self.position_qty
        if self.min_qty is not None and qty < self.min_qty:
            qty = int(self.min_qty)
        sl_response = await self.place_limit_order(exit_side, sl_price, qty)
        tp_response = await self.place_limit_order(exit_side, tp_price, qty)
        logger.info("Placed SL and TP orders", sl_price=sl_price, tp_price=tp_price, qty=qty)

    async def stop(self):
        self.running = False
        if self.current_bid_order_id:
            await self.cancel_existing_order(self.current_bid_order_id)
        if self.current_ask_order_id:
            await self.cancel_existing_order(self.current_ask_order_id)
        logger.info("MMService stopped.")


# Example standalone run:
if __name__ == "__main__":
    async def main():
        mm_service = MMService(risk_aversion=0.1, k=1.0, T=1.0, days_to_fetch=7)
        await mm_service.run(update_interval=None)
    asyncio.run(main())
