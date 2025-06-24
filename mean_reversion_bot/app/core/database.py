import asyncio
import asyncpg
from structlog import get_logger
from app.core.config import Config

logger = get_logger(__name__)

class Database:
    _pool = None
    _max_retries = 5
    _base_delay = 2

    @classmethod
    async def initialize(cls):
        for attempt in range(cls._max_retries):
            try:
                logger.info(f"DB connect attempt {attempt+1}/{cls._max_retries}")
                cls._pool = await asyncpg.create_pool(
                    **Config.DB_CONFIG, min_size=5, max_size=20, timeout=30
                )
                async with cls._pool.acquire() as conn:
                    # Enable TimescaleDB
                    await conn.execute('CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;')

                    # market_data hypertable
                    await conn.execute('''
                        CREATE TABLE IF NOT EXISTS market_data (
                            symbol TEXT NOT NULL,
                            time TIMESTAMPTZ NOT NULL,
                            trade_id TEXT NOT NULL,
                            price DOUBLE PRECISION NOT NULL,
                            volume DOUBLE PRECISION NOT NULL,
                            PRIMARY KEY (symbol, time, trade_id)
                        );
                    ''')
                    await conn.execute('''
                        SELECT create_hypertable(
                            'market_data', 'time',
                            if_not_exists => TRUE,
                            chunk_time_interval => INTERVAL '1 day'
                        );
                    ''')

                    # candles hypertable
                    await conn.execute('''
                        CREATE TABLE IF NOT EXISTS candles (
                            symbol TEXT NOT NULL,
                            time TIMESTAMPTZ NOT NULL,
                            open DOUBLE PRECISION NOT NULL,
                            high DOUBLE PRECISION NOT NULL,
                            low DOUBLE PRECISION NOT NULL,
                            close DOUBLE PRECISION NOT NULL,
                            volume DOUBLE PRECISION NOT NULL,
                            PRIMARY KEY (symbol, time)
                        );
                    ''')
                    await conn.execute('''
                        SELECT create_hypertable(
                            'candles', 'time',
                            if_not_exists => TRUE,
                            chunk_time_interval => INTERVAL '1 day'
                        );
                    ''')

                    # positions (track open position per symbol)
                    await conn.execute('''
                        CREATE TABLE IF NOT EXISTS positions (
                            symbol   TEXT         PRIMARY KEY,
                            side     TEXT         NOT NULL,    -- 'long' or 'short'
                            size     DOUBLE PRECISION NOT NULL,
                            entry_ts TIMESTAMPTZ  NOT NULL,
                            entry_px DOUBLE PRECISION NOT NULL
                        );
                    ''')

                    # fills (record individual fills)
                    await conn.execute('''
                        CREATE TABLE IF NOT EXISTS fills (
                            id     SERIAL        PRIMARY KEY,
                            symbol TEXT          NOT NULL,
                            side   TEXT          NOT NULL,
                            qty    DOUBLE PRECISION NOT NULL,
                            price  DOUBLE PRECISION NOT NULL,
                            ts     TIMESTAMPTZ   NOT NULL DEFAULT NOW()
                        );
                    ''')

                logger.info("Database initialized")
                return
            except Exception as e:
                logger.error("DB init failed", error=str(e))
                await asyncio.sleep(cls._base_delay ** attempt)

    @classmethod
    async def close(cls):
        if cls._pool:
            await cls._pool.close()
            logger.info("DB pool closed")

    @classmethod
    async def execute(cls, query, *args):
        if not cls._pool:
            raise RuntimeError("DB not initialized")
        async with cls._pool.acquire() as conn:
            return await conn.execute(query, *args)

    @classmethod
    async def fetch(cls, query, *args):
        if not cls._pool:
            raise RuntimeError("DB not initialized")
        async with cls._pool.acquire() as conn:
            return await conn.fetch(query, *args)

    @classmethod
    async def fetchval(cls, query, *args):
        if not cls._pool:
            raise RuntimeError("DB not initialized")
        async with cls._pool.acquire() as conn:
            return await conn.fetchval(query, *args)

    @classmethod
    async def fetchrow(cls, query, *args):
        if not cls._pool:
            raise RuntimeError("DB not initialized")
        async with cls._pool.acquire() as conn:
            return await conn.fetchrow(query, *args)
