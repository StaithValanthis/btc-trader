# File: app/init.py

import asyncio
from app.utils.logger import logger
from app.core import Database, BybitMarketData, Config
from app.services import TradeService, CandleService
from app.debug.startup_check import StartupChecker

class TradingBot:
    def __init__(self):
        self.bybit = BybitMarketData()
        self.trade_service = TradeService()
        self.candle_service = CandleService(interval_seconds=60)  # Aggregates 1-minute candles
        self.running = False

    async def run(self):
        try:
            # Run startup checks (includes backfilling candles if necessary)
            await StartupChecker.run_checks()

            logger.info("Initializing database...")
            await Database.initialize()
            await self.trade_service.initialize()

            # Start live candle aggregation.
            await self.candle_service.start()

            self.running = True
            logger.info("Starting main tasks...")

            await asyncio.gather(
                self.bybit.run(),
                self._monitor_system(),
                self._trading_loop()
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

    async def _trading_loop(self):
        while self.running:
            await self.trade_service.run_trading_logic()
            await asyncio.sleep(60)  # Run trading logic every 1 minute

    async def stop(self):
        self.running = False
        await self.bybit.stop()
        await self.candle_service.stop()
        await self.trade_service.stop()
        await Database.close()
        logger.info("Trading bot fully stopped")

if __name__ == "__main__":
    from app import TradingBot
    asyncio.run(TradingBot().run())
