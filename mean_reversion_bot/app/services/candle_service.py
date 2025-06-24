import asyncio, pandas as pd
from datetime import datetime,timezone,timedelta
from structlog import get_logger
from app.core.database import db_fetch, db_execute

logger = get_logger(__name__)

async def aggregate_one_min(pool, symbol):
    now = datetime.now(timezone.utc).replace(second=0,microsecond=0)
    since = now - timedelta(minutes=1)
    rows = await db_fetch(pool, """
        SELECT time,price,volume FROM market_data
        WHERE symbol=$1 AND time >= $2 AND time < $3
        ORDER BY time
    """, symbol, since, now)

    if rows:
        df = pd.DataFrame(rows,columns=["time","price","volume"])
        o = df.price.iloc[0]
        h = df.price.max()
        l = df.price.min()
        c = df.price.iloc[-1]
        v = df.volume.sum()
    else:
        last = await pool.fetchrow("""
            SELECT close FROM candles
            WHERE symbol=$1
            ORDER BY time DESC LIMIT 1
        """,symbol)
        if not last: return
        o=h=l=c=last["close"]; v=0.0

    await db_execute(pool, """
        INSERT INTO candles(symbol,time,open,high,low,close,volume)
        VALUES($1,$2,$3,$4,$5,$6,$7)
        ON CONFLICT DO NOTHING
    """, symbol, since, o,h,l,c,v)

async def run_candle_aggregator(pool, symbol):
    while True:
        try:
            await aggregate_one_min(pool, symbol)
        except Exception as e:
            logger.error("Candle agg error",symbol=symbol,error=str(e))
        await asyncio.sleep(1)
