import asyncio
import time
from app import TradingBot
from app.core.config import Config

async def main():
    # Wait so Docker Compose can wire up networks
    time.sleep(15)
    symbols = Config.TRADING_CONFIG['symbols']
    bots = [TradingBot(symbol) for symbol in symbols]
    await asyncio.gather(*(bot.run() for bot in bots))

if __name__ == "__main__":
    asyncio.run(main())
