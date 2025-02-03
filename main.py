from trading_bot import BitcoinTrader
import asyncio

if __name__ == "__main__":
    trader = BitcoinTrader()
    asyncio.run(trader.run())