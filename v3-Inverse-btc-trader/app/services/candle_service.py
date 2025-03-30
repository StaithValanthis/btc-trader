# File: app/services/candle_service.py

import asyncio
from datetime import datetime, timezone, timedelta
from structlog import get_logger
import pandas as pd

from app.core.database import Database

logger = get_logger(__name__)

class CandleService:
    """
    Aggregates trades from `market_data` into 1-minute OHLCV candles in `candles`.
    If no trades occur in a minute, carry forward the previous candle's values (volume=0).
    """

    def __init__(self, interval_seconds=60):
        self.interval_seconds = interval_seconds
        self.running = False

    async def start(self):
        """Begin candle aggregation in the background."""
        self.running = True
        asyncio.create_task(self._run_aggregator())

    async def stop(self):
        """Stop candle aggregation."""
        self.running = False
        logger.info("Candle service stopped")

    async def _run_aggregator(self):
        """
        Main loop that every `interval_seconds`:
          1. Finds trades from the last interval
          2. Aggregates into OHLCV
          3. Inserts into `candles` table
          4. If no trades occur, we carry forward the previous candle (volume=0)
        """
        while self.running:
            try:
                # We'll produce a candle for the previous interval
                end_time = datetime.now(timezone.utc).replace(second=0, microsecond=0)
                start_time = end_time - timedelta(seconds=self.interval_seconds)

                # Candle covers [start_time, end_time)
                rows = await Database.fetch('''
                    SELECT time, price, volume
                    FROM market_data
                    WHERE time >= $1 AND time < $2
                    ORDER BY time ASC
                ''', start_time, end_time)

                candle_time = start_time

                if rows:
                    # We have trades. Aggregate them into OHLCV.
                    df = pd.DataFrame(rows, columns=["time", "price", "volume"])
                    o_price = df["price"].iloc[0]
                    h_price = df["price"].max()
                    l_price = df["price"].min()
                    c_price = df["price"].iloc[-1]
                    total_vol = df["volume"].sum()

                    await self._insert_candle(
                        candle_time, o_price, h_price, l_price, c_price, total_vol
                    )
                    logger.debug(
                        "Inserted new 1-minute candle (with trades)",
                        candle_time=candle_time.isoformat(),
                        open=o_price, high=h_price, low=l_price,
                        close=c_price, vol=total_vol
                    )

                else:
                    # No trades in this interval. Carry forward the last candle's OHLC as a flat candle.
                    last_candle = await Database.fetchrow('''
                        SELECT open, high, low, close
                        FROM candles
                        ORDER BY time DESC
                        LIMIT 1
                    ''')
                    if last_candle:
                        # Use the last candle's close for open, high, low, close. volume=0
                        ohlc = dict(last_candle)
                        o_price = ohlc["close"]
                        h_price = ohlc["close"]
                        l_price = ohlc["close"]
                        c_price = ohlc["close"]
                        total_vol = 0.0

                        await self._insert_candle(
                            candle_time, o_price, h_price, l_price, c_price, total_vol
                        )
                        logger.debug(
                            "Inserted new 1-minute candle (no trades), carried forward",
                            candle_time=candle_time.isoformat(),
                            close=c_price
                        )
                    else:
                        # There is no previous candle yet, meaning the table is empty.
                        # Just log it and continue.
                        logger.debug(
                            "No trades and no previous candle exists yet",
                            interval=(start_time, end_time)
                        )
            except Exception as e:
                logger.error("Candle aggregation error", error=str(e))

            # Sleep for the next interval
            await asyncio.sleep(self.interval_seconds)

    async def _insert_candle(self, candle_time, open_p, high_p, low_p, close_p, volume):
        """
        Helper method to insert (or do-nothing if conflict) a row into the `candles` table.
        """
        query = '''
            INSERT INTO candles (time, open, high, low, close, volume)
            VALUES ($1, $2, $3, $4, $5, $6)
            ON CONFLICT (time) DO NOTHING
        '''
        await Database.execute(query, candle_time, open_p, high_p, low_p, close_p, volume)