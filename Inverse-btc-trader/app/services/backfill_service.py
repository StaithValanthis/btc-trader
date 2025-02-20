# File: app/services/backfill_service.py

import asyncio
import time
from datetime import datetime, timezone
from structlog import get_logger
from pybit.unified_trading import HTTP

from app.core.database import Database
from app.core.config import Config

logger = get_logger(__name__)

async def backfill_bybit_kline(
    symbol="BTCUSD",
    interval=1,
    days_to_fetch=1,
    start_time_ms=None
):
    """
    Fetch Bybit inverse Kline (candle) data (OHLC) and insert it into the `candles` table.
    
    - symbol: e.g. "BTCUSD" for the inverse contract.
    - interval: candlestick interval in minutes (e.g., 1, 3, 5, 15, etc.).
    - days_to_fetch: number of days of historical data to fetch.
    - start_time_ms: starting UTC timestamp in milliseconds (if None, defaults to days_to_fetch ago).
    
    Note: This function assumes that the database pool is already initialized.
    """
    session = HTTP(
        testnet=Config.BYBIT_CONFIG['testnet'],
        api_key=Config.BYBIT_CONFIG['api_key'],
        api_secret=Config.BYBIT_CONFIG['api_secret']
    )

    # If no start time is provided, default to days_to_fetch ago.
    if start_time_ms is None:
        now_ms = int(time.time() * 1000)
        start_time_ms = now_ms - (days_to_fetch * 24 * 60 * 60 * 1000)

    total_minutes = days_to_fetch * 24 * 60
    bars_per_fetch = 200
    inserted_count = 0
    current_start = start_time_ms

    fetches_needed = (total_minutes // (bars_per_fetch * interval)) + 1

    logger.info(f"Starting candle backfill for {symbol}: interval={interval} minute(s), "
                f"{days_to_fetch} day(s) starting at {start_time_ms}")

    for _ in range(fetches_needed):
        resp = await asyncio.to_thread(
            session.get_kline,
            category="inverse",
            symbol=symbol,
            interval=str(interval),
            start=current_start,
            limit=bars_per_fetch
        )
        if resp.get("retCode", 0) != 0:
            logger.error(f"Bybit API error (retCode: {resp.get('retCode')}) - {resp.get('retMsg')}")
            break

        kline_data = resp.get("result", {}).get("list", [])
        if not kline_data:
            logger.info("No more data returned from Bybit. Stopping backfill.")
            break

        for bar in kline_data:
            try:
                # bar[0] is the open time in milliseconds.
                bar_time_ms = int(bar[0])
                dt = datetime.utcfromtimestamp(bar_time_ms / 1000).replace(tzinfo=timezone.utc)
                o_price = float(bar[1])
                h_price = float(bar[2])
                l_price = float(bar[3])
                c_price = float(bar[4])
                volume  = float(bar[5])

                query = '''
                    INSERT INTO candles (time, open, high, low, close, volume)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    ON CONFLICT (time) DO NOTHING
                '''
                await Database.execute(query, dt, o_price, h_price, l_price, c_price, volume)
                inserted_count += 1
            except Exception as e:
                logger.error("Failed to insert candle", 
                             error=str(e),
                             time=dt,
                             open=o_price,
                             high=h_price,
                             low=l_price,
                             close=c_price,
                             volume=volume)

        current_start += bars_per_fetch * interval * 60 * 1000

    logger.info(f"Bybit candle backfill complete for {symbol}. Inserted {inserted_count} records.")


async def maybe_backfill_candles(
    min_rows=2000,        # Require at least 2000 rows in the candles table.
    symbol="BTCUSD",
    interval=1,
    days_to_fetch=21,     # Fetch 21 days of data.
    start_time_ms=None
):
    """
    Check the `candles` table row count. If fewer than `min_rows` rows exist,
    automatically backfill historical candles.
    """
    await Database.initialize()
    row_count = await Database.fetchval("SELECT COUNT(*) FROM candles")
    logger.info(f"Candle row count: {row_count}")

    if row_count < min_rows:
        logger.warning(f"Candles table has fewer than {min_rows} rows; initiating backfill for {days_to_fetch} day(s).")
        await backfill_bybit_kline(
            symbol=symbol,
            interval=interval,
            days_to_fetch=days_to_fetch,
            start_time_ms=start_time_ms
        )
    else:
        logger.info("Candles table has sufficient data; backfill not required.")


if __name__ == "__main__":
    asyncio.run(backfill_bybit_kline())
