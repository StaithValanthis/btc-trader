#!/usr/bin/env python3
import asyncio

from app.core.config import Config
from app.utils.logger import logger
from app.services.trade_service import TradeService

# helper that you should already have, or copy from your utils:
from app.utils.symbols import fetch_top_symbols

# make sure this class actually exists in app/core/bybit_client.py
from app.core.bybit_client import BybitClient


async def main():
    is_testnet = Config.BYBIT_CONFIG["testnet"]

    # 1) choose symbols
    if is_testnet:
        symbols = Config.TRADING_CONFIG["symbols"]
        logger.info("Testnet mode: using default symbols", symbols=symbols)
    else:
        symbols = await fetch_top_symbols(
            category=Config.BYBIT_CONFIG["category"],
            limit=30
        )
        logger.info("Mainnet mode: fetched top symbols by volume", symbols=symbols)

    # 2) open one shared WebSocket
    async with BybitClient() as ws:
        services = []

        # 3) initialize each TradeService
        for sym in symbols:
            svc = TradeService(sym)
            await svc.initialize()
            if svc.session and svc.tick_size is not None:
                services.append(svc)
            else:
                logger.warning("TradeService init failed; skipping symbol", symbol=sym)

        if not services:
            logger.error("No valid symbols to trade—exiting")
            return

        # 4) run them all in parallel on the same WS
        tasks = [asyncio.create_task(svc.run(ws)) for svc in services]
        await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
