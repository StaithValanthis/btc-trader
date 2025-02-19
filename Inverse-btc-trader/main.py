# main.py
import asyncio
from app import TradingBot

async def main():
    bot = TradingBot()
    await bot.run()

if __name__ == "__main__":
    asyncio.run(main())
