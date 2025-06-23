import asyncio
from datetime import datetime, timezone, timedelta
import pandas as pd
from structlog import get_logger

from app.core.database import Database

logger = get_logger(__name__)

class CandleService:
    """
    Aggregates `market_data` into 1-min OHLCV per symbol.
    """
    def __init__(self, symbol: str, interval_seconds: int = 60):
        self.symbol = symbol
        self.interval_seconds = interval_seconds
        self.running = False

    async def start(self):
        self.running = True
        asyncio.create_task(self._run_aggregator())

    async def stop(self):
        self.running = False
        logger.info(f"CandleService stopped for {self.symbol}")

    async def _run_aggregator(self):
        while self.running:
            try:
                end_time = datetime.now(timezone.utc).replace(second=0, microsecond=0)
                start_time = end_time - timedelta(seconds=self.interval_seconds)

                rows = await Database.fetch('''
                    SELECT time, price, volume
                    FROM market_data
                    WHERE symbol=$1 AND time >= $2 AND time < $3
                    ORDER BY time ASC
                ''', self.symbol, start_time, end_time)

                candle_time = start_time
                if rows:
                    df = pd.DataFrame(rows, columns=["time","price","volume"])
                    o = df["price"].iloc[0]
                    h = df["price"].max()
                    l = df["price"].min()
                    c = df["price"].iloc[-1]
                    v = df["volume"].sum()
                else:
                    last = await Database.fetchrow('''
                        SELECT close FROM candles
                        WHERE symbol=$1
                        ORDER BY time DESC
                        LIMIT 1
                    ''', self.symbol)
                    if last:
                        o = h = l = c = last["close"]
                        v = 0.0
                    else:
                        # no previous candle -> skip
                        await asyncio.sleep(self.interval_seconds)
                        continue

                await Database.execute('''
                    INSERT INTO candles (symbol, time, open, high, low, close, volume)
                    VALUES ($1,$2,$3,$4,$5,$6,$7)
                    ON CONFLICT DO NOTHING
                ''', self.symbol, candle_time, o, h, l, c, v)

            except Exception as e:
                logger.error("Candle agg error", error=str(e), symbol=self.symbol)

            await asyncio.sleep(self.interval_seconds)
