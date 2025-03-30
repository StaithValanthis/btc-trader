# main.py
import asyncio
import time

from app import TradingBot

async def main():
    # Optional: Wait 15 seconds so Docker Compose can wire up networks
    time.sleep(15)
    bot = TradingBot()
    await bot.run()

if __name__ == "__main__":
    asyncio.run(main())
