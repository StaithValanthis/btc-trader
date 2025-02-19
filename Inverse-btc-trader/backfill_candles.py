# File: backfill_candles.py

import asyncio
import time
from datetime import datetime, timezone
import pandas as pd
from structlog import get_logger

# We'll import Database from your existing code
# Adjust the import path to match your project structure
from app.core.database import Database
from app.core.config import Config
from pybit.unified_trading import HTTP

logger = get_logger(__name__)

async def backfill_bybit_kline(
    symbol="BTCUSD",
    interval=1,  # 1-minute bars
    days_to_fetch=1,
    start_time_ms=None
):
    """
    Fetch historical kline from Bybit's v5 API and insert into the `candles` table.
    - symbol: e.g., "BTCUSD"
    - interval: candlestick interval in minutes (1,3,5,15,30,60, etc.)
    - days_to_fetch: how many days of data to fetch
    - start_time_ms: the starting UTC timestamp in milliseconds for your backfill
    """
    await Database.initialize()

    # Create a Bybit session
    session = HTTP(
        testnet=Config.BYBIT_CONFIG['testnet'],
        api_key=Config.BYBIT_CONFIG['api_key'],
        api_secret=Config.BYBIT_CONFIG['api_secret'],
    )

    # Convert days to total minutes
    total_minutes = days_to_fetch * 24 * 60
    bars_per_fetch = 200  # Bybit typically returns up to 200 bars per request

    # If not provided, default to some date in the past
    if start_time_ms is None:
        # Example: go back ~1 day from now
        now_ms = int(time.time() * 1000)
        start_time_ms = now_ms - (days_to_fetch * 24 * 60 * 60 * 1000)

    current_start = start_time_ms

    # We'll do multiple fetches if total_minutes > bars_per_fetch * interval
    max_fetches = (total_minutes // (bars_per_fetch * interval)) + 1

    inserted_count = 0

    logger.info("Starting backfill", symbol=symbol, interval=interval, days=days_to_fetch)

    for _ in range(max_fetches):
        try:
            resp = session.get_kline(
                category="inverse",  # for BTCUSD inverse
                symbol=symbol,
                interval=str(interval),
                start=current_start,
                limit=bars_per_fetch
            )
            if resp["retCode"] != 0:
                logger.error("Bybit error", ret_code=resp["retCode"], message=resp.get("retMsg"))
                break

            data_list = resp["result"].get("list", [])
            if not data_list:
                logger.info("No more kline data returned by Bybit.")
                break

            # Each bar might look like: [openTime, open, high, low, close, volume, turnover]
            # We'll parse each and insert into `candles`.
            batch_inserts = []
            for bar in data_list:
                # bar[0] = open time in ms
                bar_open_time_ms = int(bar[0])
                dt = datetime.utcfromtimestamp(bar_open_time_ms / 1000).replace(tzinfo=timezone.utc)

                o_price = float(bar[1])
                h_price = float(bar[2])
                l_price = float(bar[3])
                c_price = float(bar[4])
                vol     = float(bar[5])  # volume in BTC if it's an inverse contract

                batch_inserts.append((dt, o_price, h_price, l_price, c_price, vol))

            # Insert them
            for row in batch_inserts:
                dt, o, h, l, c, v = row
                query = '''
                    INSERT INTO candles (time, open, high, low, close, volume)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    ON CONFLICT (time) DO NOTHING
                '''
                await Database.execute(query, dt, o, h, l, c, v)
                inserted_count += 1

            # Move start_time forward by bars_per_fetch * interval minutes
            current_start += bars_per_fetch * interval * 60 * 1000

        except Exception as e:
            logger.error("Backfill error", error=str(e))
            break

    logger.info("Backfill complete", inserted=inserted_count)
    await Database.close()

if __name__ == "__main__":
    # Example usage:
    # python -m backfill_candles
    # or python backfill_candles.py
    #
    # Will fetch 1 day of 1-min data from Bybit starting ~24h ago
    asyncio.run(
        backfill_bybit_kline(
            symbol="BTCUSD",
            interval=1,
            days_to_fetch=2,  # fetch 2 days worth
            start_time_ms=None
        )
    )
