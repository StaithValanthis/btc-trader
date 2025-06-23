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
        # will hold only symbols successfully initialized
        self.symbols: list[str] = []
        self.md_service    = BybitMarketData()
        self.candle_svcs   = {}  # symbol → CandleService
        self.trade_svcs    = {}  # symbol → TradeService
        self.running = False
        self._refresh_task = None

    async def run(self):
        # 1) Pre‐run checks & initial backfill
        await StartupChecker.run_checks()

        # 2) Start MD websocket
        self.running = True
        await self.md_service.run()

        # 3) Determine initial symbol‐set
        if not Config.BYBIT_CONFIG.get('testnet', True):
            # mainnet: fetch top-30 by volume
            to_trade = await asyncio.to_thread(fetch_top_symbols, 30)
        else:
            # testnet: use the static list in config
            to_trade = Config.TRADING_CONFIG.get('symbols', [])

        await self._update_symbols(to_trade)

        # 4) Kick off refresh loop
        self._refresh_task = asyncio.create_task(self._symbol_refresh_loop())

        # 5) Keep running
        while self.running:
            await asyncio.sleep(1)

    async def stop(self):
        self.running = False
        if self._refresh_task:
            self._refresh_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._refresh_task

        await self.md_service.stop()
        for svc in self.candle_svcs.values():
            await svc.stop()
        for svc in self.trade_svcs.values():
            svc.stop()
        await Database.close()
        logger.info("TradingBot stopped")

    async def _symbol_refresh_loop(self, interval_s: int = 3600):
        while self.running:
            try:
                if not Config.BYBIT_CONFIG.get('testnet', True):
                    new_syms = await asyncio.to_thread(fetch_top_symbols, 30)
                else:
                    new_syms = Config.TRADING_CONFIG.get('symbols', [])
                await self._update_symbols(new_syms)
            except Exception as e:
                logger.warning("Failed to refresh symbols", error=str(e))
            await asyncio.sleep(interval_s)

    async def _update_symbols(self, new_symbols: list[str]):
        """
        Ensure we only run services for symbols that successfully init;
        tear down any extras.
        """
        old_set = set(self.symbols)
        new_set = set(new_symbols)

        # STOP services for symbols no longer desired
        for sym in old_set - new_set:
            logger.info("Removing symbol", symbol=sym)
            cs = self.candle_svcs.pop(sym, None)
            if cs:
                await cs.stop()
            ts = self.trade_svcs.pop(sym, None)
            if ts:
                ts.stop()

        # ADD services for brand-new symbols
        for sym in new_set - old_set:
            logger.info("Adding symbol", symbol=sym)
            cs = CandleService(sym)
            await cs.start()

            ts = TradeService(sym)
            try:
                ok = await ts.initialize()
            except Exception as e:
                ok = False
                logger.warning("TradeService init failed; skipping symbol", symbol=sym, error=str(e))

            if not ok:
                # if trade init fails, roll back candle svc too
                await cs.stop()
                continue

            # both candle + trade are up
            self.candle_svcs[sym] = cs
            self.trade_svcs[sym]  = ts

        # update our “active” list to those with a trade svc
        self.symbols = list(self.trade_svcs.keys())
        Config.TRADING_CONFIG['symbols'] = self.symbols
        logger.info("Updated active symbols", symbols=self.symbols)
