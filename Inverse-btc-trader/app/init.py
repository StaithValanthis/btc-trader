# File: app/init.py

import asyncio
from app.utils.logger import logger
from app.core import Database, BybitMarketData, Config
from app.services import TradeService
from app.debug.startup_check import StartupChecker

class TradingBot:
    def __init__(self):
        self.bybit = BybitMarketData()
        self.trade_service = TradeService()
        self.running = False

    async def run(self):
        """Initialize and run the bot."""
        try:
            # Minimal startup checks (DB, Bybit)
            await StartupChecker.run_checks()

            logger.info("Initializing database...")
            await Database.initialize()
            await self.trade_service.initialize()

            self.running = True
            logger.info("Starting main tasks...")

            # Start Bybit data feed + monitor tasks
            await asyncio.gather(
                self.bybit.run(),
                self._monitor_system()
            )
        except Exception as e:
            logger.error("Bot failed to start", error=str(e))
            raise
        finally:
            await self.stop()

    async def _monitor_system(self):
        """Simple health monitoring."""
        while self.running:
            try:
                ws_status = {
                    "connected": self.bybit.ws is not None,
                    "last_message": self.bybit.last_message_time,
                    "reconnect_attempts": self.bybit.reconnect_attempts
                }
                db_status = await Database.fetch("SELECT 1")

                logger.info("System status",
                            ws_status=ws_status,
                            db_status=db_status)

                await asyncio.sleep(30)  # Check health every 30s
            except Exception as e:
                logger.error("Monitoring error", error=str(e))
                await asyncio.sleep(10)

    async def stop(self):
        """Stop the bot gracefully."""
        self.running = False
        await self.bybit.stop()
        await self.trade_service.stop()
        await Database.close()
        logger.info("Trading bot fully stopped")
