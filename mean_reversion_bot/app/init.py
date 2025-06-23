import asyncio
from structlog import get_logger
from app.core import Database
from app.core.bybit_client import BybitMarketData
from app.services.candle_service import CandleService
from app.services.trade_service import TradeService
from app.debug.startup_check import StartupChecker
from app.core.config import Config

logger = get_logger(__name__)

class TradingBot:
    def __init__(self):
        self.symbols = Config.TRADING_CONFIG['symbols']
        self.md_service    = BybitMarketData()
        self.candle_svcs   = [CandleService(s) for s in self.symbols]
        self.trade_svcs    = [TradeService(s) for s in self.symbols]
        self.running = False

    async def run(self):
        # Pre-run checks
        await StartupChecker.run_checks()

        # Init DB & backfill done in StartupChecker
        self.running = True

        # Start market-data websocket
        await self.md_service.run()

        # Start candle aggregators
        for cs in self.candle_svcs:
            await cs.start()

        # Init & start trade services
        for ts in self.trade_svcs:
            await ts.initialize()

        # Keep alive
        while self.running:
            await asyncio.sleep(1)

    async def stop(self):
        self.running = False
        await self.md_service.stop()
        for cs in self.candle_svcs:
            await cs.stop()
        for ts in self.trade_svcs:
            ts.stop()
        await Database.close()
        logger.info("TradingBot stopped")
