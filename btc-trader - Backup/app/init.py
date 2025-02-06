from app.utils.logger import logger
from app.core.database import Database
from app.core.bybit_client import BybitMarketData
from app.services.trade_service import TradeService
from app.strategies.sma_crossover import SMACrossover
import asyncio
import nest_asyncio

nest_asyncio.apply()

class TradingBot:
    def __init__(self):
        self.bybit = BybitMarketData()
        self.strategy = SMACrossover()
        self.running = False

    async def run(self):
        try:
            await Database.initialize()
            self.running = True
            
            async def data_analysis():
                while self.running:
                    try:
                        data = await TradeService().get_market_data(100)
                        await self.strategy.analyze(data)
                        await asyncio.sleep(60)
                    except Exception as e:
                        logger.error("Data analysis error", error=str(e))
                        await asyncio.sleep(10)
            
            await asyncio.gather(
                self.bybit.run(),
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