# File: v2-Inverse-btc-trader/main.py

import asyncio
from app import TradingBot

async def main() -> None:
    """
    Entry point for the trading bot. Initializes and runs the TradingBot.
    """
    bot = TradingBot()
    await bot.run()

if __name__ == "__main__":
    asyncio.run(main())
