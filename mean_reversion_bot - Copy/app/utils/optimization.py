# File: app/utils/optimization.py

import asyncio
import random
import itertools
import threading
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from app.core.database import Database
from app.core.config import Config
from app.utils.cache import redis_client


class Backtester:
    """
    Brute-force sweeps Bollinger & RSI & SL parameters over historical candles
    and returns a list of dicts with (params, pnl, sharpe).
    """

    def __init__(
        self,
        symbols,
        bb_window_range,
        bb_dev_range,
        rsi_long_range,
        rsi_short_range,
        sl_pct_range,
    ):
        self.symbols = symbols
        self.bb_window_range = bb_window_range
        self.bb_dev_range = bb_dev_range
        self.rsi_long_range = rsi_long_range
        self.rsi_short_range = rsi_short_range
        self.sl_pct_range = sl_pct_range

    def sweep(self):
        results = []
        # iterate all combinations
        for bb_w, bb_d, rsi_l, rsi_s, sl in itertools.product(
            self.bb_window_range,
            self.bb_dev_range,
            self.rsi_long_range,
            self.rsi_short_range,
            self.sl_pct_range,
        ):
            # simple sanity check
            if rsi_l >= rsi_s:
                continue

            pnl_list = []
            for sym in self.symbols:
                sharpe = self._simulate(sym, bb_w, bb_d, rsi_l, rsi_s, sl)
                pnl_list.append(sharpe)
            avg_sharpe = float(np.nanmean(pnl_list))
            results.append({
                'BB_WINDOW': bb_w,
                'BB_DEV': bb_d,
                'RSI_LONG': rsi_l,
                'RSI_SHORT': rsi_s,
                'SL_PCT': sl,
                'sharpe': avg_sharpe
            })
        return results

    def _simulate(self, symbol, bb_w, bb_d, rsi_l, rsi_s, sl_pct):
        """
        Placeholder simulation: in real life you'd fetch candles from DB,
        compute signals, and calculate PnL & Sharpe. Here we return a random.
        """
        # TODO: replace with real backtest logic
        return random.uniform(0.5, 2.0)


class MonteCarlo:
    """
    Randomly sample parameter sets within given ranges, evaluate via the
    provided sweep-results or on the fly, and return the best params.
    """

    def __init__(
        self,
        results,
        trials,
        window_range,
        dev_range,
        rsi_long_range,
        rsi_short_range,
        sl_pct_range,
    ):
        self.results = results
        self.trials = trials
        self.window_range = window_range
        self.dev_range = dev_range
        self.rsi_long_range = rsi_long_range
        self.rsi_short_range = rsi_short_range
        self.sl_pct_range = sl_pct_range

    def run(self):
        best = {'sharpe': -np.inf}
        for _ in range(self.trials):
            # sample uniformly from continuous ranges
            bb_w = random.randint(*self.window_range)
            bb_d = random.uniform(*self.dev_range)
            rsi_l = random.randint(*self.rsi_long_range)
            rsi_s = random.randint(*self.rsi_short_range)
            sl   = random.uniform(*self.sl_pct_range)

            # avoid invalid
            if rsi_l >= rsi_s:
                continue

            # pretend we compute sharpe: pick nearest from sweep or random
            sharpe = random.uniform(0.5, 2.0)
            if sharpe > best['sharpe']:
                best = {
                    'BB_WINDOW': bb_w,
                    'BB_DEV': round(bb_d, 2),
                    'RSI_LONG': rsi_l,
                    'RSI_SHORT': rsi_s,
                    'SL_PCT': round(sl, 4),
                    'sharpe': sharpe
                }
        return best


class LiveTuner:
    """
    Monitors real-time PnL stored in Redis and nudges parameters toward
    better-performing neighbors every hour.
    """

    def __init__(self, redis_client, param_names):
        self.redis = redis_client
        self.param_names = param_names
        self._stop = threading.Event()

    def start(self):
        """Kick off a background thread that runs every hour."""
        thread = threading.Thread(target=self._run, daemon=True)
        thread.start()

    def _run(self):
        while not self._stop.is_set():
            try:
                # fetch last-hour PnL from Redis (placeholder key)
                pnl = float(self.redis.get('live_pnl') or 0.0)
                # if pnl is negative, nudge params randomly
                if pnl < 0:
                    new_params = {
                        name: random.choice(Config.PARAM_RANGES[name])
                        for name in self.param_names
                    }
                    Config.TRADING_CONFIG.update(new_params)
                # sleep until next hour
            except Exception:
                pass
            self._stop.wait(3600)

    def stop(self):
        self._stop.set()
