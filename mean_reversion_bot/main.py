# File: mean_reversion_bot/main.py

import asyncio
from structlog import get_logger

from app.core.config import Config
from app.utils.symbols import fetch_top_symbols
from app.init import TradingBot

logger = get_logger(__name__)

async def main():
    # ── Dynamically fetch top 30 USDT-perps by 24 h volume ──
    try:
        top30 = fetch_top_symbols(30)
        Config.TRADING_CONFIG['symbols'] = top30
        logger.info("Overwrote TRADING_CONFIG symbols with top30 by volume")
    except Exception as e:
        logger.warning("Could not fetch top symbols; using defaults", error=str(e))

    bot = TradingBot()
    await bot.run()

if __name__ == "__main__":
    asyncio.run(main())
