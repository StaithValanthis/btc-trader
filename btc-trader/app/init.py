from app.utils.logger import logger
from app.core.database import Database
from app.core.bybit_client import BybitMarketData
from app.services.trade_service import TradeService
from app.strategies.lstm_strategy import LSTMStrategy
from app.debug.startup_check import StartupChecker
import asyncio
import nest_asyncio
import time

nest_asyncio.apply()

class TradingBot:
    def __init__(self):
        self.bybit = BybitMarketData()
        self.trade_service = TradeService()
        self.strategy = LSTMStrategy(self.trade_service)
        self.running = False

    async def run(self):
        try:
            # Run automated startup checks
            await StartupChecker.run_checks()
            
            logger.info("Initializing database...")
            await Database.initialize()
            
            logger.info("Starting WebSocket connection...")
            await self.bybit.run()
            
            self.running = True
            logger.info("Starting trading strategy...")
            
            await asyncio.gather(
                self.strategy.run(),
                self._health_check()
            )
        except Exception as e:
            logger.error("Bot failed to start", error=str(e))
            raise
        finally:
            await self.stop()

    async def _health_check(self):
        """Periodic system health monitoring"""
        while self.running:
            try:
                # Check WebSocket connection
                if not self.bybit.ws:
                    logger.warning("WebSocket disconnected - attempting reconnect")
                    await self.bybit.run()
                
                # Check database connection
                await Database.fetch("SELECT 1")
                
                await asyncio.sleep(30)
            except Exception as e:
                logger.error("Health check failed", error=str(e))

    async def stop(self):
        self.running = False
        await self.bybit.stop()
        await Database.close()
        logger.info("Trading bot stopped")

async def main():
    bot = TradingBot()
    await bot.run()

if __name__ == "__main__":
    asyncio.run(main())