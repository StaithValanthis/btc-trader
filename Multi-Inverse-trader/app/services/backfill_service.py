import asyncio
import time
from datetime import datetime, timezone
from structlog import get_logger
from pybit.unified_trading import HTTP
from app.core.database import Database
from app.core.config import Config

logger = get_logger(__name__)

async def backfill_bybit_kline(
    symbol,
    interval=1,
    days_to_fetch=365,
    start_time_ms=None
):
    """
    Fetch Bybit inverse Kline (candle) data for the given symbol and insert it into the candles table.
    """
    session = HTTP(
        testnet=Config.BYBIT_CONFIG['testnet'],
        api_key=Config.BYBIT_CONFIG['api_key'],
        api_secret=Config.BYBIT_CONFIG['api_secret']
    )

    # Determine the start time based on the latest candle for this symbol.
    if start_time_ms is None:
        latest = await Database.fetchval(
            "SELECT EXTRACT(EPOCH FROM MAX(time)) * 1000 FROM candles WHERE symbol=$1",
            symbol
        )
        if latest is None:
            now_ms = int(time.time() * 1000)
            start_time_ms = now_ms - (days_to_fetch * 24 * 60 * 60 * 1000)
        else:
            start_time_ms = int(latest) + 1

    total_minutes = days_to_fetch * 24 * 60
    bars_per_fetch = 200  # Bybit returns up to 200 bars per request
    inserted_count = 0
    current_start = start_time_ms
    fetches_needed = (total_minutes // (bars_per_fetch * interval)) + 1

    logger.info(
        f"Starting rolling candle backfill for {symbol}: interval={interval} minute(s), "
        f"{days_to_fetch} day(s) from {start_time_ms}"
    )

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
            logger.error(
                f"Bybit API error for {symbol} (retCode: {resp.get('retCode')}) - {resp.get('retMsg')}"
            )
            break

        kline_data = resp.get("result", {}).get("list", [])
        if not kline_data:
            logger.info(f"No more data returned from Bybit for {symbol}. Stopping backfill.")
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
                    INSERT INTO candles (symbol, time, open, high, low, close, volume)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    ON CONFLICT (symbol, time) DO NOTHING
                '''
                await Database.execute(query, symbol, dt, o_price, h_price, l_price, c_price, volume)
                inserted_count += 1
            except Exception as e:
                logger.error("Failed to insert candle", error=str(e))
        current_start += bars_per_fetch * interval * 60 * 1000

    logger.info(f"Bybit candle backfill complete for {symbol}. Inserted {inserted_count} records.")
    return inserted_count

async def maybe_backfill_candles(
    symbol,
    min_rows=1000,
    interval=1,
    days_to_fetch=365,
    start_time_ms=None
):
    """
    Check the candle count for the given symbol and backfill if the count is below min_rows.
    """
    await Database.initialize()
    row_count = await Database.fetchval("SELECT COUNT(*) FROM candles WHERE symbol=$1", symbol)
    logger.info(f"Candle row count for {symbol}: {row_count}")

    if row_count < min_rows:
        logger.warning(
            f"Candles table for {symbol} has fewer than {min_rows} rows; initiating backfill for {days_to_fetch} day(s)."
        )
        await backfill_bybit_kline(
            symbol=symbol,
            interval=interval,
            days_to_fetch=days_to_fetch,
            start_time_ms=start_time_ms
        )
    else:
        logger.info(f"Candles table for {symbol} has sufficient data; backfill not required.")
