# app/__init__.py
import asyncio
import signal
from structlog import get_logger
from app.core import Database, BybitMarketData
from app.services.trade_service import TradeService
from app.strategies.lstm_strategy import LSTMStrategy

logger = get_logger(__name__)

class TradingBot:
    def __init__(self):
        self.trade_service = TradeService()
        self.strategy = LSTMStrategy(self.trade_service)
        self.bybit = BybitMarketData(strategy=self.strategy)  # Pass strategy here
        self.running = False

    async def run(self):
        """Main application loop"""
        self.running = True
        signal.signal(signal.SIGINT, self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)
        
        try:
            # Initialize components
            await Database.initialize()
            await self.trade_service.initialize()
            await self.market_data.initialize()

            # Start main services
            await asyncio.gather(
                self.market_data.run(),
                self.strategy.run()
            )

        except Exception as e:
            logger.critical("Fatal error", error=str(e))
        finally:
            await self.stop()

    def _shutdown(self, signum, frame):
        """Signal handler for graceful shutdown"""
        logger.info(f"Received shutdown signal {signum}")
        self.running = False

    async def stop(self):
        """Orderly shutdown sequence"""
        logger.info("Initiating shutdown...")
        tasks = [
            self.market_data.stop(),
            self.trade_service.stop(),
            Database.close()
        ]
        await asyncio.gather(*tasks)
        logger.info("Shutdown complete")

async def main():
    bot = TradingBot()
    await bot.run()

if __name__ == "__main__":
    asyncio.run(main())