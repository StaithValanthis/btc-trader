import asyncio
import asyncpg
from structlog import get_logger
from app.core.config import Config

logger = get_logger(__name__)

async def init_db():
    cfg = Config["DB"]
    for i in range(5):
        try:
            logger.info("Connecting to DB", attempt=i+1)
            pool = await asyncpg.create_pool(
                **cfg, min_size=5, max_size=20, timeout=30
            )
            async with pool.acquire() as conn:
                await conn.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;")
                # market_data
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS market_data (
                      symbol TEXT NOT NULL, time TIMESTAMPTZ NOT NULL,
                      trade_id TEXT NOT NULL, price DOUBLE PRECISION NOT NULL,
                      volume DOUBLE PRECISION NOT NULL,
                      PRIMARY KEY(symbol,time,trade_id)
                    );
                    SELECT create_hypertable('market_data','time',if_not_exists=>TRUE);
                """)
                # candles
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS candles (
                      symbol TEXT NOT NULL, time TIMESTAMPTZ NOT NULL,
                      open,high,low,close,volume DOUBLE PRECISION NOT NULL,
                      PRIMARY KEY(symbol,time)
                    );
                    SELECT create_hypertable('candles','time',if_not_exists=>TRUE);
                """)
                # positions, fills...
            logger.info("DB initialized")
            return pool
        except Exception as e:
            logger.error("DB init failed", error=str(e))
            await asyncio.sleep(2**i)
    raise RuntimeError("Could not initialize DB")

# wrappers
async def db_execute(pool, query, *args):
    async with pool.acquire() as conn:
        return await conn.execute(query,*args)

async def db_fetch(pool, query, *args):
    async with pool.acquire() as conn:
        return await conn.fetch(query,*args)

async def db_fetchval(pool, query, *args):
    async with pool.acquire() as conn:
        return await conn.fetchval(query,*args)
