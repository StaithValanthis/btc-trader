# File: app/init.py
import asyncio
from app.utils.logger import logger
from app.core import Database, BybitMarketData, Config
from app.services import CandleService
from app.services.mm_service import MMService
from app.debug.startup_check import StartupChecker

class TradingBot:
    def __init__(self):
        self.bybit = BybitMarketData()
        # Use MMService which now includes ML-based tuning
        self.mm_service = MMService(risk_aversion=0.1, k=1.0, T=1.0)
        self.candle_service = CandleService(interval_seconds=3600)
        self.running = False

    async def run(self):
        try:
            await StartupChecker.run_checks()
            logger.info("Initializing database...")
            await Database.initialize()
            await self.candle_service.start()
            self.running = True
            logger.info("Starting main tasks...")
            await asyncio.gather(
                self.bybit.run(),
                self._monitor_system(),
                self.mm_service.run(update_interval=60)
            )
        except Exception as e:
            logger.error("Bot failed to start", error=str(e))
            raise
        finally:
            await self.stop()

    async def _monitor_system(self):
        while self.running:
            try:
                ws_status = {
                    "connected": self.bybit.ws is not None,
                    "last_message": self.bybit.last_message_time,
                    "reconnect_attempts": self.bybit.reconnect_attempts
                }
                db_status = await Database.fetch("SELECT 1")
                logger.info("System status", ws_status=ws_status, db_status=db_status)
            except Exception as e:
                logger.error("Monitoring error", error=str(e))
            await asyncio.sleep(30)

    async def stop(self):
        self.running = False
        await self.bybit.stop()
        await self.candle_service.stop()
        await self.mm_service.stop()
        await Database.close()
        logger.info("Trading bot fully stopped")

if __name__ == "__main__":
    from app import TradingBot
    asyncio.run(TradingBot().run())
