import asyncio
import nest_asyncio
from app.core.database import Database
from app.core.bybit_client import BybitMarketData
from app.strategies.sma_crossover import SMACrossover
from app.services.trade_service import TradeService
from app.utils.logger import configure_logger

configure_logger()

async def main():
    # Initialize database
    await Database.initialize()
    
    # Create services
    market_data = BybitMarketData()
    trade_service = TradeService()
    strategy = SMACrossover(trade_service)
    
    # Run components
    await asyncio.gather(
        market_data.run(),
        strategy.run(),
        trade_service.monitor_positions()
    )

if __name__ == "__main__":
    nest_asyncio.apply()
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Bot stopped by user")