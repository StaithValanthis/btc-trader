# File: v2-Inverse-btc-trader/app/services/backfill_service.py

import asyncio
import time
from datetime import datetime, timezone
from structlog import get_logger
from pybit.unified_trading import HTTP
from app.core.database import Database
from app.core.config import Config

logger = get_logger(__name__)

async def backfill_bybit_kline(
    symbol: str = "BTCUSD",
    interval: int = 1,
    days_to_fetch: int = 365,
    start_time_ms: int = None
) -> None:
    """
    Fetch Bybit inverse Kline (candle) data and insert into `candles` using batched inserts.

    Args:
        symbol (str): E.g. "BTCUSD"
        interval (int): Candlestick interval in minutes
        days_to_fetch (int): Number of days to fetch
        start_time_ms (int): Start time in milliseconds, defaults to the last candle + 1
    """

    session = HTTP(
        testnet=Config.BYBIT_CONFIG['testnet'],
        api_key=Config.BYBIT_CONFIG['api_key'],
        api_secret=Config.BYBIT_CONFIG['api_secret']
    )

    if start_time_ms is None:
        latest = await Database.fetchval("SELECT EXTRACT(EPOCH FROM MAX(time)) * 1000 FROM candles")
        if latest is None:
            now_ms = int(time.time() * 1000)
            start_time_ms = now_ms - (days_to_fetch * 24 * 60 * 60 * 1000)
        else:
            start_time_ms = int(latest) + 1

    total_minutes = days_to_fetch * 24 * 60
    bars_per_fetch = 200
    inserted_count = 0
    current_start = start_time_ms
    fetches_needed = (total_minutes // (bars_per_fetch * interval)) + 1

    logger.info(f"Starting candle backfill for {symbol}, interval={interval} min, {days_to_fetch} day(s)")

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

        batch_inserts = []
        for bar in kline_data:
            try:
                bar_time_ms = int(bar[0])
                dt = datetime.utcfromtimestamp(bar_time_ms / 1000).replace(tzinfo=timezone.utc)
                o_price = float(bar[1])
                h_price = float(bar[2])
                l_price = float(bar[3])
                c_price = float(bar[4])
                volume  = float(bar[5])
                batch_inserts.append((dt, o_price, h_price, l_price, c_price, volume))
            except Exception as e:
                logger.error("Failed to parse candle", error=str(e))

        if batch_inserts:
            values_str = ",".join(
                f"('{row[0].isoformat()}',{row[1]},{row[2]},{row[3]},{row[4]},{row[5]})"
                for row in batch_inserts
            )
            insert_query = f"""
                INSERT INTO candles (time, open, high, low, close, volume)
                VALUES {values_str}
                ON CONFLICT (time) DO NOTHING
            """
            await Database.execute(insert_query)
            inserted_count += len(batch_inserts)

        current_start += bars_per_fetch * interval * 60 * 1000

    logger.info(f"Bybit candle backfill complete. Inserted {inserted_count} records for {symbol}.")

async def maybe_backfill_candles(
    min_rows: int = 1000,
    symbol: str = "BTCUSD",
    interval: int = 1,
    days_to_fetch: int = 365,
    start_time_ms: int = None
) -> None:
    """
    Check if `candles` table has < min_rows; if so, backfill historical candles.

    Args:
        min_rows (int): Threshold to decide whether to backfill
        symbol (str): Symbol to backfill
        interval (int): Candlestick interval in minutes
        days_to_fetch (int): Number of days to fetch if backfilling
        start_time_ms (int): If specified, start from this time in ms
    """
    await Database.initialize()
    row_count = await Database.fetchval("SELECT COUNT(*) FROM candles")
    logger.info(f"Candle row count: {row_count}")

    if row_count < min_rows:
        logger.warning(f"Candles table has < {min_rows} rows; initiating backfill for {days_to_fetch} days.")
        await backfill_bybit_kline(
            symbol=symbol,
            interval=interval,
            days_to_fetch=days_to_fetch,
            start_time_ms=start_time_ms
        )
    else:
        logger.info("Candles table already has sufficient data; backfill not required.")

if __name__ == "__main__":
    asyncio.run(backfill_bybit_kline())
