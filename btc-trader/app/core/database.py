import asyncpg
from structlog import get_logger
from app.core.config import Config

logger = get_logger(__name__)

class Database:
    _pool = None

    @classmethod
    async def get_pool(cls):
        if not cls._pool:
            cls._pool = await asyncpg.create_pool(
                **Config.DB_CONFIG,
                min_size=5,
                max_size=20
            )
        return cls._pool

    @classmethod
    async def execute(cls, query, *args):
        pool = await cls.get_pool()
        async with pool.acquire() as conn:
            return await conn.execute(query, *args)

    @classmethod
    async def fetch(cls, query, *args):
        pool = await cls.get_pool()
        async with pool.acquire() as conn:
            return await conn.fetch(query, *args)

    @classmethod
    async def initialize(cls):
        try:
            pool = await cls.get_pool()
            async with pool.acquire() as conn:
                # Create TimescaleDB extension
                await conn.execute('CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;')
                
                # Create market_data table
                await conn.execute('''
                    CREATE TABLE IF NOT EXISTS market_data (
                        time TIMESTAMPTZ NOT NULL PRIMARY KEY,
                        price DOUBLE PRECISION NOT NULL,
                        volume DOUBLE PRECISION NOT NULL
                    );
                ''')
                
                # Convert to hypertable
                await conn.execute('''
                    SELECT create_hypertable(
                        'market_data', 
                        'time',
                        if_not_exists => TRUE
                    );
                ''')
                
                # Add data retention policy (2 months)
                await conn.execute('''
                    SELECT add_retention_policy(
                        'market_data', 
                        INTERVAL '2 months'
                    );
                ''')
                
                # Enable compression for market_data
                await conn.execute('''
                    ALTER TABLE market_data SET (
                        timescaledb.compress,
                        timescaledb.compress_orderby = 'time DESC'
                    );
                ''')
                
                # Add compression policy (compress data older than 7 days)
                await conn.execute('''
                    SELECT add_compression_policy(
                        'market_data', 
                        INTERVAL '7 days'
                    );
                ''')
                
                # Create trades table
                await conn.execute('''
                    CREATE TABLE IF NOT EXISTS trades (
                        time TIMESTAMPTZ NOT NULL PRIMARY KEY,
                        side TEXT NOT NULL,
                        price DOUBLE PRECISION NOT NULL,
                        qty DOUBLE PRECISION NOT NULL
                    );
                ''')
                
                # Add data retention policy for trades (6 months)
                await conn.execute('''
                    SELECT add_retention_policy(
                        'trades', 
                        INTERVAL '6 months'
                    );
                ''')
                
                logger.info("Database initialized with retention and compression policies")
        except Exception as e:
            logger.error("Database initialization failed", error=str(e))
            raise

    @classmethod
    async def close(cls):
        if cls._pool:
            await cls._pool.close()
            logger.info("Database connection closed")