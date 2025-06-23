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
    symbol: str,
    interval: int = 1,
    days_to_fetch: int = 365,
    start_time_ms: int = None
):
    """
    Fetch Bybit linear Kline (candle) data for `symbol`
    and insert into the `candles` table.
    """
    session = HTTP(
        testnet=Config.BYBIT_CONFIG['testnet'],
        api_key=Config.BYBIT_CONFIG['api_key'],
        api_secret=Config.BYBIT_CONFIG['api_secret']
    )

    # Determine start_time based on latest candle if not provided
    if start_time_ms is None:
        latest = await Database.fetchval(
            "SELECT EXTRACT(EPOCH FROM MAX(time)) * 1000 FROM candles WHERE symbol = $1",
            symbol
        )
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
    category = Config.BYBIT_CONFIG['category']  # 'linear'

    logger.info(
        "Backfilling candles",
        symbol=symbol,
        interval=interval,
        days=days_to_fetch,
        category=category
    )

    for _ in range(fetches_needed):
        resp = await asyncio.to_thread(
            session.get_kline,
            category=category,
            symbol=symbol,
            interval=str(interval),
            start=current_start,
            limit=bars_per_fetch
        )
        if resp.get("retCode", 0) != 0:
            logger.error(
                "Bybit API error",
                symbol=symbol,
                retCode=resp.get("retCode"),
                message=resp.get("retMsg")
            )
            break

        kline_data = resp.get("result", {}).get("list", [])
        if not kline_data:
            logger.info("No more data for symbol", symbol=symbol)
            break

        for bar in kline_data:
            try:
                ts_ms = int(bar[0])
                dt = datetime.utcfromtimestamp(ts_ms / 1000).replace(tzinfo=timezone.utc)
                o = float(bar[1])
                h = float(bar[2])
                l = float(bar[3])
                c = float(bar[4])
                v = float(bar[5])
                await Database.execute(
                    '''
                    INSERT INTO candles (symbol, time, open, high, low, close, volume)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    ON CONFLICT (symbol, time) DO NOTHING
                    ''',
                    symbol, dt, o, h, l, c, v
                )
                inserted_count += 1
            except Exception as e:
                logger.error(
                    "Failed to insert candle",
                    symbol=symbol,
                    error=str(e)
                )

        current_start += bars_per_fetch * interval * 60 * 1000

    logger.info(
        "Completed backfill",
        symbol=symbol,
        inserted=inserted_count
    )


async def maybe_backfill_candles(
    min_rows: int = 1000,
    interval: int = 1,
    days_to_fetch: int = 365
):
    """
    For each symbol in Config.TRADING_CONFIG['symbols'], check its candle row count.
    If below min_rows, backfill up to `days_to_fetch`.
    """
    await Database.initialize()
    symbols = Config.TRADING_CONFIG['symbols']

    for symbol in symbols:
        row_count = await Database.fetchval(
            "SELECT COUNT(*) FROM candles WHERE symbol = $1",
            symbol
        )
        logger.info("Candle row count", symbol=symbol, count=row_count)

        if row_count < min_rows:
            logger.warning(
                "Insufficient candles; initiating backfill",
                symbol=symbol,
                have=row_count,
                need=min_rows
            )
            await backfill_bybit_kline(
                symbol=symbol,
                interval=interval,
                days_to_fetch=days_to_fetch
            )
        else:
            logger.info("Sufficient candle data", symbol=symbol, count=row_count)

    await Database.close()
