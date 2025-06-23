# File: mean_reversion_bot/main.py

import asyncio
from structlog import get_logger

from app.core.config import Config
from app.init import TradingBot

logger = get_logger(__name__)

async def main():
    # log the raw configuration
    logger.info("Loaded configuration", config=Config.__dict__)

    # -- Remove any manual overwrite of Config.TRADING_CONFIG['symbols'] here! --
    # The TradingBot.run() method will:
    #   - fetch top-30 when testnet=False (i.e. mainnet)
    #   - use the static symbols from TRADING_CONFIG when testnet=True

    bot = TradingBot()
    try:
        await bot.run()
    except KeyboardInterrupt:
        await bot.stop()

if __name__ == "__main__":
    asyncio.run(main())
