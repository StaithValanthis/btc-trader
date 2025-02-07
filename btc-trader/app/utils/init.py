from app.utils.logger import logger
from app.core.database import Database
from app.core.bybit_client import BybitMarketData
from app.services.trade_service import TradeService
from app.strategies.lstm_strategy import LSTMStrategy
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
            logger.info("Initializing database...")
            await Database.initialize()
            logger.info("Database initialized")
            
            self.running = True
            logger.info("Starting WebSocket and data analysis...")
            
            async def data_analysis():
                while self.running:
                    try:
                        # Progress tracking
                        if not self.strategy.data_ready:
                            count = await Database.fetch("SELECT COUNT(*) FROM market_data")
                            elapsed = time.time() - (self.strategy.warmup_start_time or time.time())
                            logger.info(
                                "Initializing...",
                                data_collected=f"{count[0]['count']}/{Config.MODEL_CONFIG['min_training_samples']}",
                                time_elapsed=f"{elapsed:.1f}s"
                            )
                        
                        # Fetch and analyze data
                        data = await self.trade_service.get_market_data(100)
                        await self.strategy.analyze(data)
                        await asyncio.sleep(60)
                    except Exception as e:
                        logger.error("Data analysis error", error=str(e))
                        await asyncio.sleep(10)
            
            await asyncio.gather(
                self.bybit.run(),
                self.strategy.run(),
                data_analysis()
            )
        except Exception as e:
            logger.error("Bot failed", error=str(e))
            raise
        finally:
            await self.stop()

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