import asyncio
from datetime import datetime, timezone, timedelta
from structlog import get_logger
import pandas as pd
from app.core.database import Database

logger = get_logger(__name__)

class CandleService:
    """
    Aggregates trade data into 1-minute OHLCV candles for a given symbol.
    If no trades occur, it carries forward the last candle.
    """
    def __init__(self, symbol, interval_seconds=60):
        self.symbol = symbol
        self.interval_seconds = interval_seconds
        self.running = False

    async def start(self):
        self.running = True
        asyncio.create_task(self._run_aggregator())

    async def stop(self):
        self.running = False
        logger.info(f"Candle service for {self.symbol} stopped")

    async def _run_aggregator(self):
        while self.running:
            try:
                # Produce a candle for the previous interval
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
                    df = pd.DataFrame(rows, columns=["time", "price", "volume"])
                    o_price = df["price"].iloc[0]
                    h_price = df["price"].max()
                    l_price = df["price"].min()
                    c_price = df["price"].iloc[-1]
                    total_vol = df["volume"].sum()
                    await self._insert_candle(candle_time, o_price, h_price, l_price, c_price, total_vol)
                    logger.debug("Inserted new 1-minute candle (with trades)",
                        symbol=self.symbol, candle_time=candle_time.isoformat(),
                        open=o_price, high=h_price, low=l_price, close=c_price, vol=total_vol)
                else:
                    last_candle = await Database.fetchrow('''
                        SELECT open, high, low, close
                        FROM candles
                        WHERE symbol=$1
                        ORDER BY time DESC
                        LIMIT 1
                    ''', self.symbol)
                    if last_candle:
                        ohlc = dict(last_candle)
                        o_price = ohlc["close"]
                        h_price = ohlc["close"]
                        l_price = ohlc["close"]
                        c_price = ohlc["close"]
                        total_vol = 0.0
                        await self._insert_candle(candle_time, o_price, h_price, l_price, c_price, total_vol)
                        logger.debug("Inserted new 1-minute candle (no trades), carried forward",
                            symbol=self.symbol, candle_time=candle_time.isoformat(), close=c_price)
                    else:
                        logger.debug("No trades and no previous candle exists yet",
                                     symbol=self.symbol, interval=(start_time, end_time))
            except Exception as e:
                logger.error("Candle aggregation error", error=str(e))
            await asyncio.sleep(self.interval_seconds)

    async def _insert_candle(self, candle_time, open_p, high_p, low_p, close_p, volume):
        query = '''
            INSERT INTO candles (symbol, time, open, high, low, close, volume)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            ON CONFLICT (symbol, time) DO NOTHING
        '''
        await Database.execute(query, self.symbol, candle_time, open_p, high_p, low_p, close_p, volume)
