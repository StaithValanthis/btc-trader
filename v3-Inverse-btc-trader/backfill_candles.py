# File: v2-Inverse-btc-trader/backfill_candles.py

import asyncio
import time
from datetime import datetime, timezone
from structlog import get_logger
from app.core.database import Database
from app.core.config import Config
from pybit.unified_trading import HTTP

logger = get_logger(__name__)

async def backfill_bybit_kline(
    symbol: str = "BTCUSD",
    interval: int = 1,  # in minutes
    days_to_fetch: int = 1,
    start_time_ms: int = None
) -> None:
    """
    Fetch historical kline from Bybit's v5 API and insert into the `candles` table.
    Uses a single multi-row insert per batch to improve performance.

    Args:
        symbol (str): Symbol to backfill, e.g. "BTCUSD"
        interval (int): Candlestick interval in minutes
        days_to_fetch (int): How many days of data to fetch
        start_time_ms (int): The starting UTC timestamp in ms for your backfill
    """
    await Database.initialize()

    session = HTTP(
        testnet=Config.BYBIT_CONFIG['testnet'],
        api_key=Config.BYBIT_CONFIG['api_key'],
        api_secret=Config.BYBIT_CONFIG['api_secret'],
    )

    total_minutes = days_to_fetch * 24 * 60
    bars_per_fetch = 200

    if start_time_ms is None:
        now_ms = int(time.time() * 1000)
        start_time_ms = now_ms - (days_to_fetch * 24 * 60 * 60 * 1000)

    current_start = start_time_ms
    max_fetches = (total_minutes // (bars_per_fetch * interval)) + 1
    inserted_count = 0

    logger.info("Starting backfill", symbol=symbol, interval=interval, days=days_to_fetch)

    for _ in range(max_fetches):
        try:
            resp = session.get_kline(
                category="inverse",
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

            # Build one multi-row insert
            batch_inserts = []
            for bar in data_list:
                bar_open_time_ms = int(bar[0])
                dt = datetime.utcfromtimestamp(bar_open_time_ms / 1000).replace(tzinfo=timezone.utc)
                o_price = float(bar[1])
                h_price = float(bar[2])
                l_price = float(bar[3])
                c_price = float(bar[4])
                vol     = float(bar[5])
                batch_inserts.append((dt, o_price, h_price, l_price, c_price, vol))

            if batch_inserts:
                values_str = ",".join(
                    f"('{row[0].isoformat()}',{row[1]},{row[2]},{row[3]},{row[4]},{row[5]})"
                    for row in batch_inserts
                )
                query = f"""
                    INSERT INTO candles (time, open, high, low, close, volume)
                    VALUES {values_str}
                    ON CONFLICT (time) DO NOTHING
                """
                await Database.execute(query)
                inserted_count += len(batch_inserts)

            current_start += bars_per_fetch * interval * 60 * 1000

        except Exception as e:
            logger.error("Backfill error", error=str(e))
            break

    logger.info("Backfill complete", inserted=inserted_count)
    await Database.close()

if __name__ == "__main__":
    asyncio.run(
        backfill_bybit_kline(
            symbol="BTCUSD",
            interval=1,
            days_to_fetch=2,
            start_time_ms=None
        )
    )
