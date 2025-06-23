# File: mean_reversion_bot/app/init.py

import asyncio
import contextlib
from structlog import get_logger

from app.core import Database
from app.core.bybit_client import BybitMarketData
from app.services.candle_service import CandleService
from app.services.trade_service import TradeService
from app.debug.startup_check import StartupChecker
from app.core.config import Config
from app.utils.symbols import fetch_top_symbols

logger = get_logger(__name__)

class TradingBot:
    def __init__(self):
        # Active symbols & services
        self.symbols: list[str] = []
        self.md_service    = BybitMarketData()
        self.candle_svcs   = {}  # symbol → CandleService
        self.trade_svcs    = {}  # symbol → TradeService
        self.running       = False
        self._refresh_task = None

    async def run(self):
        # 1) Pre‐run checks & initial backfill
        await StartupChecker.run_checks()

        # 2) Start market‐data WebSocket
        self.running = True
        await self.md_service.run()

        # 3) Fetch top‐30 symbols & spin up their services
        initial_syms = await asyncio.to_thread(fetch_top_symbols, 30)
        await self._update_symbols(initial_syms)

        # 4) Begin hourly refresh
        self._refresh_task = asyncio.create_task(self._symbol_refresh_loop())

        # 5) Keep alive
        while self.running:
            await asyncio.sleep(1)

    async def stop(self):
        # stop refresh loop
        self.running = False
        if self._refresh_task:
            self._refresh_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._refresh_task

        # stop market‐data WS
        await self.md_service.stop()

        # stop candle & trade services
        for cs in self.candle_svcs.values():
            await cs.stop()
        for ts in self.trade_svcs.values():
            ts.stop()

        # close DB
        await Database.close()
        logger.info("TradingBot stopped")

    async def _symbol_refresh_loop(self, interval_s: int = 3600):
        """Re-fetch top symbols every `interval_s` seconds and reconcile."""
        while self.running:
            try:
                new_syms = await asyncio.to_thread(fetch_top_symbols, 30)
                await self._update_symbols(new_syms)
            except Exception as e:
                logger.warning("Failed to refresh symbols", error=str(e))
            await asyncio.sleep(interval_s)

    async def _update_symbols(self, new_symbols: list[str]):
        """
        Add services for new symbols and remove services for dropped ones.
        Only symbols whose TradeService.initialization succeeds are kept.
        """
        old_set = set(self.symbols)
        new_set = set(new_symbols)

        to_remove = old_set - new_set
        to_add    = new_set - old_set

        # 1) stop & remove dropped symbols
        for sym in to_remove:
            logger.info("Removing symbol", symbol=sym)
            cs = self.candle_svcs.pop(sym, None)
            if cs:
                await cs.stop()
            ts = self.trade_svcs.pop(sym, None)
            if ts:
                ts.stop()

        # 2) add new symbols
        for sym in to_add:
            logger.info("Adding symbol", symbol=sym)
            # start candle service
            cs = CandleService(sym)
            await cs.start()
            # initialize trade service
            ts = TradeService(sym)
            ok = await ts.initialize()
            if ok:
                # both candle & trade succeeded
                self.candle_svcs[sym] = cs
                self.trade_svcs[sym] = ts
            else:
                # teardown candle if trade failed
                logger.warning("TradeService init failed; skipping symbol", symbol=sym)
                await cs.stop()

        # 3) update our list & global config
        kept = list(set(self.candle_svcs.keys()) & set(self.trade_svcs.keys()))
        self.symbols = kept
        Config.TRADING_CONFIG['symbols'] = kept
        logger.info("Updated active symbols", symbols=kept)
