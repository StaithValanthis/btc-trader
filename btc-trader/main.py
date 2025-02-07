import asyncio
import nest_asyncio
from app.core import Database, BybitMarketData
from app.services import TradeService
from app.strategies import LSTMStrategy
from app.utils.logger import configure_logger

configure_logger()

async def main():
    await Database.initialize()
    
    market_data = BybitMarketData()
    trade_service = TradeService()
    strategy = LSTMStrategy(trade_service)
    
    await asyncio.gather(
        market_data.run(),
        strategy.run()
    )

if __name__ == "__main__":
    nest_asyncio.apply()
    asyncio.run(main())