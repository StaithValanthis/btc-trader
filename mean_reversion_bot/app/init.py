# File: app/init.py

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
        self.symbols: list[str] = []
        self.md_service    = BybitMarketData()
        self.candle_svcs   = {}  # symbol → CandleService
        self.trade_svcs    = {}  # symbol → TradeService
        self.running = False
        self._refresh_task = None

    async def run(self):
        # 1) Pre‐run checks & initial backfill
        await StartupChecker.run_checks()

        # 2) Initialize DB & MD websocket
        self.running = True
        await self.md_service.run()

        # 3) Pick your symbols
        if Config.BYBIT_CONFIG.get("testnet", False):
            # testnet: leave your defaults intact
            initial = Config.TRADING_CONFIG["symbols"]
            logger.info("Testnet mode: using default symbols", symbols=initial)
        else:
            # mainnet: fetch the top 30 by volume
            initial = await asyncio.to_thread(fetch_top_symbols, 30)
            logger.info("Mainnet mode: fetched top symbols by volume", symbols=initial)

        await self._update_symbols(initial)

        # 4) Kick off the symbol‐refresh loop (every hour), only on mainnet
        if not Config.BYBIT_CONFIG.get("testnet", False):
            self._refresh_task = asyncio.create_task(self._symbol_refresh_loop())
        else:
            logger.info("Testnet mode: skipping symbol‐refresh loop")

        # 5) Keep the bot alive
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
        """Every `interval_s` seconds, re‐fetch top symbols and reconcile."""
        while self.running:
            try:
                new_syms = await asyncio.to_thread(fetch_top_symbols, 30)
                logger.info("Hourly refresh: fetched top symbols by volume", symbols=new_syms)
                await self._update_symbols(new_syms)
            except Exception as e:
                logger.warning("Failed to refresh symbols", error=str(e))
            await asyncio.sleep(interval_s)

    async def _update_symbols(self, new_symbols: list[str]):
        old = set(self.symbols)
        new = set(new_symbols)

        to_add    = new - old
        to_remove = old - new

        for sym in to_remove:
            logger.info("Removing symbol", symbol=sym)
            cs = self.candle_svcs.pop(sym, None)
            if cs:
                await cs.stop()
            ts = self.trade_svcs.pop(sym, None)
            if ts:
                ts.stop()

        for sym in to_add:
            logger.info("Adding symbol", symbol=sym)
            cs = CandleService(sym)
            await cs.start()
            self.candle_svcs[sym] = cs

            ts = TradeService(sym)
            await ts.initialize()
            self.trade_svcs[sym] = ts

        self.symbols = new_symbols
        Config.TRADING_CONFIG['symbols'] = new_symbols
        logger.info("Updated active symbols", symbols=new_symbols)


# no change to this helper — only called from the mainnet branches above
async def _fetch_top_30() -> list[str]:
    return await asyncio.to_thread(fetch_top_symbols, 30)
