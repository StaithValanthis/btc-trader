# Updated file: app/services/backfill_service.py

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
    days_to_fetch=365,  # Updated to fetch 365 days of data.
    start_time_ms=None
):
    """
    Fetch Bybit inverse Kline (candle) data and insert it into the `candles` table using a rolling update approach.
    """
    session = HTTP(
        testnet=Config.BYBIT_CONFIG['testnet'],
        api_key=Config.BYBIT_CONFIG['api_key'],
        api_secret=Config.BYBIT_CONFIG['api_secret']
    )

    # Determine start_time_ms based on the latest candle in the DB if not provided
    if start_time_ms is None:
        latest = await Database.fetchval("SELECT EXTRACT(EPOCH FROM MAX(time)) * 1000 FROM candles")
        if latest is None:
            now_ms = int(time.time() * 1000)
            start_time_ms = now_ms - (days_to_fetch * 24 * 60 * 60 * 1000)
        else:
            start_time_ms = int(latest) + 1  # Start just after the latest candle

    total_minutes = days_to_fetch * 24 * 60
    bars_per_fetch = 200  # Bybit returns up to 200 bars per request
    inserted_count = 0
    current_start = start_time_ms

    fetches_needed = (total_minutes // (bars_per_fetch * interval)) + 1

    logger.info(f"Starting rolling candle backfill for {symbol}: interval={interval} minute(s), {days_to_fetch} day(s) from {start_time_ms}")

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
    min_rows=1000,
    symbol="BTCUSD",
    interval=1,
    days_to_fetch=365,
    start_time_ms=None
):
    """
    Check the `candles` table row count. If fewer than `min_rows` rows exist, automatically backfill historical candles.
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