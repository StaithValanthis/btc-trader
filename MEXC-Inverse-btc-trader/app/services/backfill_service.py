# File: backfill_candles.py

import asyncio
import time
from datetime import datetime, timezone
import pandas as pd
from structlog import get_logger

from app.core.database import Database
from app.core.config import Config
from app.core.mexc_client import MexcClient

logger = get_logger(__name__)

async def backfill_mexc_kline(
    symbol="BTC_USD",
    interval="1m",  # MEXC interval format (e.g., "1m", "5m", "15m", etc.)
    days_to_fetch=1,
    start_time_ms=None
):
    """
    Fetch historical candlestick (kline) data from MEXC API and insert into the 'candles' table.

    Parameters:
      - symbol: Trading pair, e.g. "BTCUSD".
      - interval: Candlestick interval.
      - days_to_fetch: How many days worth of data to backfill.
      - start_time_ms: Starting timestamp in milliseconds (UTC). If None, defaults to now minus days_to_fetch.
    """
    await Database.initialize()

    # Create a MEXC client instance
    mexc_client = MexcClient(
        api_key=Config.MEXC_CONFIG['api_key'],
        api_secret=Config.MEXC_CONFIG['api_secret'],
        testnet=Config.MEXC_CONFIG['testnet']
    )

    # Calculate total minutes to fetch and determine the number of bars per request.
    total_minutes = days_to_fetch * 24 * 60
    bars_per_fetch = 200  # Adjust based on MEXC API limits

    if start_time_ms is None:
        now_ms = int(time.time() * 1000)
        start_time_ms = now_ms - (days_to_fetch * 24 * 60 * 60 * 1000)

    current_start = start_time_ms
    inserted_count = 0

    logger.info("Starting MEXC backfill", symbol=symbol, interval=interval, days=days_to_fetch)

    # Calculate maximum fetch iterations.
    max_fetches = (total_minutes // bars_per_fetch) + 1

    for _ in range(max_fetches):
        try:
            # Call the MEXC kline endpoint. This example uses the _get method directly.
            # Ensure the endpoint and parameter names match MEXC's API.
            resp = await asyncio.to_thread(
                mexc_client._get,
                "/api/v1/contract/kline",
                {
                    "symbol": symbol,
                    "interval": interval,
                    "startTime": current_start,
                    "limit": bars_per_fetch
                }
            )
            # MEXC API is assumed to return a JSON with "code": 0 for success and "data": list of candles.
            if resp.get("code", -1) != 0:
                logger.error("MEXC API error", code=resp.get("code"), message=resp.get("msg"))
                break

            data_list = resp.get("data", [])
            if not data_list:
                logger.info("No more kline data returned from MEXC.")
                break

            # Process each candle.
            # Assuming each candle is formatted as:
            # [openTime (ms), open, high, low, close, volume]
            batch_inserts = []
            for bar in data_list:
                bar_time_ms = int(bar[0])
                dt = datetime.utcfromtimestamp(bar_time_ms / 1000).replace(tzinfo=timezone.utc)
                o_price = float(bar[1])
                h_price = float(bar[2])
                l_price = float(bar[3])
                c_price = float(bar[4])
                vol     = float(bar[5])
                batch_inserts.append((dt, o_price, h_price, l_price, c_price, vol))

            # Insert candles into the database.
            for row in batch_inserts:
                dt, o, h, l, c, v = row
                query = '''
                    INSERT INTO candles (time, open, high, low, close, volume)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    ON CONFLICT (time) DO NOTHING
                '''
                await Database.execute(query, dt, o, h, l, c, v)
                inserted_count += 1

            # Advance the start time by the number of bars fetched (each bar is assumed to be 1 minute).
            current_start += bars_per_fetch * 60 * 1000

        except Exception as e:
            logger.error("Backfill error", error=str(e))
            break

    logger.info("Backfill complete", inserted=inserted_count)
    await Database.close()

if __name__ == "__main__":
    # Example usage:
    # python backfill_candles.py
    asyncio.run(
        backfill_mexc_kline(
            symbol="BTCUSD",
            interval="1m",
            days_to_fetch=1,
            start_time_ms=None
        )
    )
