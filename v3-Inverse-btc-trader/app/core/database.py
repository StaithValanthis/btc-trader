# File: v2-Inverse-btc-trader/app/core/database.py

import asyncio
import asyncpg
from structlog import get_logger
from app.core.config import Config

logger = get_logger(__name__)

class Database:
    """
    Manages the asyncpg connection pool and provides high-level DB operations.
    Creates TimescaleDB hypertables for market_data, trades, candles.
    """

    _pool = None
    _max_retries = 5
    _base_delay = 2

    @classmethod
    async def initialize(cls) -> None:
        """
        Initialize the database connection pool and create tables.
        Includes retry logic and TimescaleDB hypertable creation.
        """
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

                logger.info("Database initialized successfully")
                return
            except Exception as e:
                logger.error("Database connection failed", attempt=attempt+1, error=str(e))
                await asyncio.sleep(cls._base_delay ** attempt)

    @classmethod
    async def close(cls) -> None:
        """
        Close the database connection pool.
        """
        if cls._pool:
            await cls._pool.close()
            logger.info("Database connection pool closed")

    @classmethod
    async def execute(cls, query: str, *args) -> str:
        """
        Execute a SQL command and return status.
        """
        if not cls._pool:
            raise RuntimeError("Database not initialized. Call Database.initialize() first.")
        async with cls._pool.acquire() as conn:
            return await conn.execute(query, *args)

    @classmethod
    async def fetch(cls, query: str, *args):
        """
        Fetch multiple rows.
        """
        if not cls._pool:
            raise RuntimeError("Database not initialized. Call Database.initialize() first.")
        async with cls._pool.acquire() as conn:
            return await conn.fetch(query, *args)

    @classmethod
    async def fetchrow(cls, query: str, *args):
        """
        Fetch a single row.
        """
        if not cls._pool:
            raise RuntimeError("Database not initialized. Call Database.initialize() first.")
        async with cls._pool.acquire() as conn:
            return await conn.fetchrow(query, *args)

    @classmethod
    async def fetchval(cls, query: str, *args):
        """
        Fetch a single value.
        """
        if not cls._pool:
            raise RuntimeError("Database not initialized. Call Database.initialize() first.")
        async with cls._pool.acquire() as conn:
            return await conn.fetchval(query, *args)
