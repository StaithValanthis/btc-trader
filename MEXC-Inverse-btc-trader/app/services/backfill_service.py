import asyncio
import time
from datetime import datetime, timezone
from structlog import get_logger
from pybit.unified_trading import HTTP

from app.core.database import Database
from app.core.config import Config

logger = get_logger(__name__)

async def backfill_bybit_kline(
    symbol=Config.BYBIT_CONFIG['symbol'],  # Use Bybit symbol (e.g. "BTCUSD")
    interval=1,  # 1-minute bars
    days_to_fetch=365,  # Fetch 365 days of data
    start_time_ms=None
):
    """
    Fetch historical kline data from Bybit and insert it into the `candles` table.
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
        try:
            resp = await asyncio.to_thread(
                session.get_kline,
                category="inverse",  # For inverse contracts
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
                    logger.error("Failed to insert candle", error=str(e),
                                 time=dt, open=o_price, high=h_price, low=l_price, close=c_price, volume=volume)

            current_start += bars_per_fetch * interval * 60 * 1000
        except Exception as e:
            logger.error("Error during backfill", error=str(e))
            break

    logger.info(f"Bybit candle backfill complete for {symbol}. Inserted {inserted_count} records.")
