import asyncio
import nest_asyncio
import time
from app.utils.logger import logger
from app.core.database import Database
from app.core.bybit_client import BybitMarketData
from app.services.trade_service import TradeService
from app.strategies.lstm_strategy import LSTMStrategy
from app.debug.startup_check import StartupChecker

nest_asyncio.apply()

class TradingBot:
    """Main trading bot class that wires together components."""
    def __init__(self):
        self.trade_service = TradeService()
        self.strategy = LSTMStrategy(self.trade_service)
        self.bybit = BybitMarketData(strategy=self.strategy)
        self.running = False

    async def run(self):
        try:
            logger.info("Performing startup checks...")
            await StartupChecker.run_checks()

            logger.info("Initializing database...")
            await Database.initialize()

            self.running = True
            logger.info("Starting main components...")

            # Run all components concurrently
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
        """Continuously monitor system health."""
        while self.running:
            try:
                ws_status = {
                    "connected": self.bybit.ws is not None,
                    "last_message": self.bybit.last_message_time.isoformat() if self.bybit.last_message_time else None,
                    "reconnect_attempts": self.bybit.reconnect_attempts
                }
                db_status = "OK" if await Database.fetch("SELECT 1") else "Error"
                strategy_status = {
                    "data_ready": self.strategy.data_ready,
                    "model_loaded": self.strategy.model_loaded,
                    "warmup_progress": self.strategy.warmup_start_time
                }
                logger.info("System status", ws_status=ws_status, db_status=db_status, strategy_status=strategy_status)
                await asyncio.sleep(60)
            except Exception as e:
                logger.error("Monitoring error", error=str(e))
                await asyncio.sleep(10)

    async def stop(self):
        """Stop all components gracefully."""
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
