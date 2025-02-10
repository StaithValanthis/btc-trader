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
        """
        Initialize database connection pool and create the market_data table 
        with PRIMARY KEY(time, trade_id). Timescale requires 'time' in any 
        unique/primary key on a hypertable.
        """
        for attempt in range(cls._max_retries):
            try:
                logger.info(f"Attempting to create DB pool (attempt {attempt+1}/{cls._max_retries})...")
                cls._pool = await asyncpg.create_pool(
                    **Config.DB_CONFIG,
                    min_size=5,
                    max_size=20,
                    timeout=30,
                    command_timeout=60
                )

                async with cls._pool.acquire() as conn:
                    # 1) Create TimescaleDB extension
                    await conn.execute('CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;')

                    # 2) Create the market_data table with a primary key on (time, trade_id)
                    await conn.execute('''
                        CREATE TABLE IF NOT EXISTS market_data (
                            time TIMESTAMPTZ NOT NULL,
                            trade_id TEXT NOT NULL,
                            price DOUBLE PRECISION NOT NULL,
                            volume DOUBLE PRECISION NOT NULL,
                            PRIMARY KEY (time, trade_id)
                        );
                    ''')

                    # 3) Convert it to a hypertable on 'time'
                    await conn.execute('''
                        SELECT create_hypertable(
                            'market_data',
                            'time',
                            if_not_exists => TRUE
                        );
                    ''')

                logger.info("Database initialized")
                return

            except Exception as e:
                # If creation fails, wait and retry
                if attempt == cls._max_retries - 1:
                    logger.error("Max DB connection attempts reached, giving up", error=str(e))
                    raise
                else:
                    wait_time = cls._base_delay ** attempt
                    logger.warning(
                        f"Database connection failed (attempt {attempt+1}/{cls._max_retries}), "
                        f"retrying in {wait_time} seconds...",
                        error=str(e)
                    )
                    await asyncio.sleep(wait_time)

    @classmethod
    async def close(cls):
        """Close the database connection pool"""
        if cls._pool:
            await cls._pool.close()
            cls._pool = None
            logger.info("Database connection pool closed")

    @classmethod
    async def execute(cls, query, *args):
        """Execute a SQL command"""
        if not cls._pool:
            raise RuntimeError("Database pool is not initialized. Call Database.initialize() first.")
        async with cls._pool.acquire() as conn:
            return await conn.execute(query, *args)

    @classmethod
    async def fetch(cls, query, *args):
        """Fetch multiple rows"""
        if not cls._pool:
            raise RuntimeError("Database pool is not initialized. Call Database.initialize() first.")
        async with cls._pool.acquire() as conn:
            return await conn.fetch(query, *args)

    @classmethod
    async def fetchrow(cls, query, *args):
        """Fetch a single row"""
        if not cls._pool:
            raise RuntimeError("Database pool is not initialized. Call Database.initialize() first.")
        async with cls._pool.acquire() as conn:
            return await conn.fetchrow(query, *args)

    @classmethod
    async def fetchval(cls, query, *args):
        """Fetch a single value"""
        if not cls._pool:
            raise RuntimeError("Database pool is not initialized. Call Database.initialize() first.")
        async with cls._pool.acquire() as conn:
            return await conn.fetchval(query, *args)
