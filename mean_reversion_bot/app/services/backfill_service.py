import asyncio, time
from datetime import datetime,timezone
from structlog import get_logger
from pybit.unified_trading import HTTP
from pybit.exceptions import InvalidRequestError
from app.core.config import Config
from app.core.database import db_execute, init_db

logger = get_logger(__name__)

async def fetch_and_insert_candles(pool, symbol, interval, days):
    session = HTTP(**Config["BYBIT"])
    now_ms = int(time.time()*1000)
    start_ms = now_ms - days*24*3600*1000
    limit=200
    next_start = start_ms

    while True:
        try:
            r = await asyncio.to_thread(
                session.get_kline,
                category = Config["BYBIT"]["category"],
                symbol   = symbol,
                interval = str(interval),
                start    = next_start,
                limit    = limit
            )
        except InvalidRequestError:
            logger.warning("Invalid symbol, stop backfill",symbol=symbol)
            return

        data = r.get("result",{}).get("list",[])
        if not data: break

        for bar in data:
            ts = int(bar[0])
            dt = datetime.fromtimestamp(ts/1000, tz=timezone.utc)
            _o,_h,_l,_c,_v = map(float,bar[1:6])
            await db_execute(pool, """
                INSERT INTO candles(symbol,time,open,high,low,close,volume)
                VALUES($1,$2,$3,$4,$5,$6,$7)
                ON CONFLICT DO NOTHING
            """, symbol, dt, _o,_h,_l,_c,_v)
        next_start += limit*interval*60*1000

async def maybe_backfill(min_rows,interval,init_days,inc_days,pool):
    for s in Config["TRADING"]["symbols"]:
        cnt = await pool.fetchval("SELECT COUNT(*) FROM candles WHERE symbol=$1",s)
        days = init_days if cnt < min_rows else inc_days
        logger.info("Backfill",symbol=s,rows=cnt,days=days)
        await fetch_and_insert_candles(pool,s,interval,days)
