import asyncio
from app.init import TradingBot

async def main():
    bot = TradingBot()
    await bot.run()

if __name__ == "__main__":
    asyncio.run(main())
