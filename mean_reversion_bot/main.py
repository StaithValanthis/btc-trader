#!/usr/bin/env python3
import asyncio

from app.core.config import Config
from app.utils.logger import logger
from app.services.trade_service import TradeService
from app.core.bybit_ws import BybitWebSocket
from app.utils.symbols import fetch_top_symbols

async def main():
    is_testnet = Config.BYBIT_CONFIG["testnet"]

    if is_testnet:
        symbols = Config.TRADING_CONFIG["symbols"]
        logger.info("Testnet mode: using default symbols", symbols=symbols)
    else:
        symbols = await fetch_top_symbols(
            n=30
        )
        logger.info("Mainnet mode: fetched top symbols by volume", symbols=symbols)

    async with BybitWebSocket() as ws:
        services = []

        for sym in symbols:
            svc = TradeService(sym)
            await svc.initialize()
            if svc.session and svc.tick_size is not None:
                await svc.run(ws)
                services.append(svc)
            else:
                logger.warning("TradeService init failed; skipping symbol", symbol=sym)

        if not services:
            logger.error("No valid symbols to trade—exiting")
            return

        while True:
            await asyncio.sleep(10)

if __name__ == "__main__":
    asyncio.run(main())
