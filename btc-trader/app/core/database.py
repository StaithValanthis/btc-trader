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
                
                # Convert market_data to hypertable
                await conn.execute('''
                    SELECT create_hypertable(
                        'market_data', 
                        'time',
                        if_not_exists => TRUE
                    );
                ''')
                
                # Check if retention policy already exists for market_data
                retention_policy_exists = await conn.fetchval('''
                    SELECT EXISTS (
                        SELECT 1
                        FROM timescaledb_information.jobs
                        WHERE proc_name = 'policy_retention'
                        AND hypertable_name = 'market_data'
                    );
                ''')
                
                # Add retention policy for market_data if it doesn't exist
                if not retention_policy_exists:
                    await conn.execute('''
                        SELECT add_retention_policy(
                            'market_data', 
                            INTERVAL '2 months'
                        );
                    ''')
                    logger.info("Retention policy added for market_data")
                else:
                    logger.info("Retention policy already exists for market_data")
                
                # Enable compression for market_data
                await conn.execute('''
                    ALTER TABLE market_data SET (
                        timescaledb.compress,
                        timescaledb.compress_orderby = 'time DESC'
                    );
                ''')
                
                # Check if compression policy already exists for market_data
                compression_policy_exists = await conn.fetchval('''
                    SELECT EXISTS (
                        SELECT 1
                        FROM timescaledb_information.jobs
                        WHERE proc_name = 'policy_compression'
                        AND hypertable_name = 'market_data'
                    );
                ''')
                
                # Add compression policy for market_data if it doesn't exist
                if not compression_policy_exists:
                    await conn.execute('''
                        SELECT add_compression_policy(
                            'market_data', 
                            INTERVAL '7 days'
                        );
                    ''')
                    logger.info("Compression policy added for market_data")
                else:
                    logger.info("Compression policy already exists for market_data")
                
                # Create trades table
                await conn.execute('''
                    CREATE TABLE IF NOT EXISTS trades (
                        time TIMESTAMPTZ NOT NULL PRIMARY KEY,
                        side TEXT NOT NULL,
                        price DOUBLE PRECISION NOT NULL,
                        qty DOUBLE PRECISION NOT NULL
                    );
                ''')
                
                # Convert trades to hypertable
                await conn.execute('''
                    SELECT create_hypertable(
                        'trades', 
                        'time',
                        if_not_exists => TRUE
                    );
                ''')
                
                # Check if retention policy already exists for trades
                trades_retention_policy_exists = await conn.fetchval('''
                    SELECT EXISTS (
                        SELECT 1
                        FROM timescaledb_information.jobs
                        WHERE proc_name = 'policy_retention'
                        AND hypertable_name = 'trades'
                    );
                ''')
                
                # Add retention policy for trades if it doesn't exist
                if not trades_retention_policy_exists:
                    await conn.execute('''
                        SELECT add_retention_policy(
                            'trades', 
                            INTERVAL '6 months'
                        );
                    ''')
                    logger.info("Retention policy added for trades")
                else:
                    logger.info("Retention policy already exists for trades")
                    
                logger.info("Database initialized with retention and compression policies")
        except Exception as e:
            logger.error("Database initialization failed", error=str(e))
            raise

    @classmethod
    async def close(cls):
        if cls._pool:
            await cls._pool.close()
            logger.info("Database connection closed")