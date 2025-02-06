import asyncpg
from structlog import get_logger

logger = get_logger(__name__)

class Database:
    _pool = None

    @classmethod
    async def get_pool(cls):
        if not cls._pool:
            cls._pool = await asyncpg.create_pool(
                user='postgres',
                password='postgres',
                database='trading_bot',
                host='postgres',
                min_size=5,
                max_size=20
            )
        return cls._pool

    @classmethod
    async def fetch(cls, query, *args):
        pool = await cls.get_pool()
        async with pool.acquire() as conn:
            return await conn.fetch(query, *args)

    @classmethod
    async def execute(cls, query, *args):
        pool = await cls.get_pool()
        async with pool.acquire() as conn:
            return await conn.execute(query, *args)

    @classmethod
    async def initialize(cls):
        pool = await cls.get_pool()
        async with pool.acquire() as conn:
            await conn.execute('CREATE EXTENSION IF NOT EXISTS timescaledb;')
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS market_data (
                    time TIMESTAMPTZ NOT NULL,
                    price DOUBLE PRECISION NOT NULL,
                    features JSONB NOT NULL
                );
            ''')
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    time TIMESTAMPTZ NOT NULL,
                    price DOUBLE PRECISION NOT NULL,
                    signal TEXT NOT NULL,
                    profit_loss DOUBLE PRECISION
                );
            ''')
            await conn.execute("SELECT create_hypertable('market_data', 'time', if_not_exists => TRUE);")
            await conn.execute("SELECT create_hypertable('trades', 'time', if_not_exists => TRUE);")

    @classmethod
    async def close(cls):
        if cls._pool:
            await cls._pool.close()