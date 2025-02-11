import asyncio
import nest_asyncio
from app.core.database import Database
from app.init import TradingBot
from app.utils.logger import logger

nest_asyncio.apply()

async def main():
    try:
        logger.info("Starting trading bot...")
        # Initialize the database (if not already done by bot, this is safe)
        await Database.initialize()

        bot = TradingBot()
        await bot.run()
    except Exception as e:
        logger.error("Application failed", error=str(e))
    finally:
        await Database.close()

if __name__ == "__main__":
    asyncio.run(main())
