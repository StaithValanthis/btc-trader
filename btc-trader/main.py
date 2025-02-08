import asyncio
import nest_asyncio
from app.core import Database
from app import TradingBot

nest_asyncio.apply()

async def main():
    try:
        # Initialize database
        await Database.initialize()
        
        # Start trading bot
        bot = TradingBot()
        await bot.run()
    except Exception as e:
        print(f"Application failed: {str(e)}")
    finally:
        await Database.close()

if __name__ == "__main__":
    asyncio.run(main())