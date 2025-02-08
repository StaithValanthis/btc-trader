from app.utils.logger import logger
from app.core import Database, BybitMarketData
from app.services import TradeService
from app.strategies import LSTMStrategy
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
            # Run comprehensive startup checks
            await StartupChecker.run_checks()
            
            logger.info("Initializing database...")
            await Database.initialize()
            
            self.running = True
            logger.info("Starting main components...")
            
            await asyncio.gather(
                self.bybit.run(),
                self.strategy.run(),
                self._monitor_system()
            )
        except Exception as e:
            logger.error("Bot failed to start", error=str(e))
            raise
        finally:
            await self.stop()

    async def _monitor_system(self):
        """Continuous system health monitoring"""
        while self.running:
            try:
                # Check WebSocket status
                ws_status = {
                    "connected": self.bybit.ws is not None,
                    "last_message": self.bybit.last_message_time,
                    "reconnect_attempts": self.bybit.reconnect_attempts
                }
                
                # Check database status
                db_status = await Database.fetch("SELECT 1")
                
                # Check strategy status
                strategy_status = {
                    "data_ready": self.strategy.data_ready,
                    "model_loaded": self.strategy.model_loaded,
                    "warmup_progress": self.strategy.warmup_start_time
                }
                
                logger.info("System status", 
                           ws_status=ws_status,
                           db_status=db_status,
                           strategy_status=strategy_status)
                
                await asyncio.sleep(60)
                
            except Exception as e:
                logger.error("Monitoring error", error=str(e))
                await asyncio.sleep(10)

    async def stop(self):
        self.running = False
        await self.bybit.stop()
        await self.trade_service.stop()
        await Database.close()
        logger.info("Trading bot fully stopped")

async def main():
    bot = TradingBot()
    await bot.run()

if __name__ == "__main__":
    asyncio.run(main())