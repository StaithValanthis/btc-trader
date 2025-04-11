# File: app/core/database.py

import asyncio
import asyncpg
from structlog import get_logger
from app.core.config import Config

logger = get_logger(__name__)

class Database:
    _pool = None

    _max_retries = 10
    _base_delay = 3

    @classmethod
    async def initialize(cls):
        """Initialize the database connection pool and create tables."""
        for attempt in range(cls._max_retries):
            try:
                logger.info(f"Database connection attempt {attempt+1}/{cls._max_retries}")
                cls._pool = await asyncpg.create_pool(
                    **Config.DB_CONFIG,
                    min_size=5,
                    max_size=20,
                    timeout=30
                )

                async with cls._pool.acquire() as conn:
                    # Create TimescaleDB extension if not present
                    await conn.execute('CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;')

                    # market_data table
                    await conn.execute('''
                        CREATE TABLE IF NOT EXISTS market_data (
                            time TIMESTAMPTZ NOT NULL,
                            trade_id TEXT NOT NULL,
                            price DOUBLE PRECISION NOT NULL,
                            volume DOUBLE PRECISION NOT NULL,
                            PRIMARY KEY (time, trade_id)
                        );
                    ''')
                    await conn.execute('''
                        SELECT create_hypertable(
                            'market_data',
                            'time',
                            if_not_exists => TRUE,
                            chunk_time_interval => INTERVAL '1 day'
                        );
                    ''')

                    # trades table
                    await conn.execute('''
                        CREATE TABLE IF NOT EXISTS trades (
                            time TIMESTAMPTZ NOT NULL,
                            side TEXT NOT NULL,
                            price DOUBLE PRECISION NOT NULL,
                            quantity DOUBLE PRECISION NOT NULL,
                            PRIMARY KEY (time, side, price)
                        );
                    ''')

                    # candles table
                    await conn.execute('''
                        CREATE TABLE IF NOT EXISTS candles (
                            time TIMESTAMPTZ NOT NULL,
                            open DOUBLE PRECISION NOT NULL,
                            high DOUBLE PRECISION NOT NULL,
                            low DOUBLE PRECISION NOT NULL,
                            close DOUBLE PRECISION NOT NULL,
                            volume DOUBLE PRECISION NOT NULL,
                            PRIMARY KEY (time)
                        );
                    ''')
                    await conn.execute('''
                        SELECT create_hypertable(
                            'candles',
                            'time',
                            if_not_exists => TRUE,
                            chunk_time_interval => INTERVAL '1 day'
                        );
                    ''')

                    # -----------------------------------------------------------
                    # REMOVE add_retention_policy call. We do not want to re-add
                    # the policy every time. If it already exists, Timescale tries
                    # to run it, leading to deadlock during backfill.
                    #
                    # Instead, do it once manually:
                    #   SELECT add_retention_policy('candles', INTERVAL '30 days');
                    #
                    # Or handle it with a separate init/migration script.
                    # -----------------------------------------------------------

                logger.info("Database initialized successfully")
                return
            except Exception as e:
                logger.error("Database connection failed", error=str(e))
                backoff = cls._base_delay ** attempt
                await asyncio.sleep(backoff)

        logger.error("All DB connection attempts failed. Check logs for DNS or Postgres issues.")

    @classmethod
    async def close(cls):
        """Close the database connection pool."""
        if cls._pool:
            await cls._pool.close()
            logger.info("Database connection pool closed")

    @classmethod
    async def execute(cls, query, *args):
        """Execute a SQL command and return status."""
        if not cls._pool:
            raise RuntimeError("Database not initialized. Call Database.initialize() first.")
        async with cls._pool.acquire() as conn:
            return await conn.execute(query, *args)

    @classmethod
    async def fetch(cls, query, *args):
        """Fetch multiple rows."""
        if not cls._pool:
            raise RuntimeError("Database not initialized. Call Database.initialize() first.")
        async with cls._pool.acquire() as conn:
            return await conn.fetch(query, *args)

    @classmethod
    async def fetchrow(cls, query, *args):
        """Fetch a single row."""
        if not cls._pool:
            raise RuntimeError("Database not initialized. Call Database.initialize() first.")
        async with cls._pool.acquire() as conn:
            return await conn.fetchrow(query, *args)

    @classmethod
    async def fetchval(cls, query, *args):
        """Fetch a single value."""
        if not cls._pool:
            raise RuntimeError("Database not initialized. Call Database.initialize() first.")
        async with cls._pool.acquire() as conn:
            return await conn.fetchval(query, *args)