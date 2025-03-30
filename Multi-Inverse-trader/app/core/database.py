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
                    # Create TimescaleDB extension if not already present
                    await conn.execute('CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;')
                    
                    # --- Market Data Table ---
                    await conn.execute('''
                        CREATE TABLE IF NOT EXISTS market_data (
                            symbol VARCHAR NOT NULL,
                            time TIMESTAMPTZ NOT NULL,
                            trade_id TEXT NOT NULL,
                            price DOUBLE PRECISION NOT NULL,
                            volume DOUBLE PRECISION NOT NULL,
                            PRIMARY KEY (symbol, time, trade_id)
                        );
                    ''')
                    # Ensure the column exists (if table existed previously)
                    await conn.execute(
                        "ALTER TABLE market_data ADD COLUMN IF NOT EXISTS symbol VARCHAR NOT NULL DEFAULT 'BTCUSD';"
                    )
                    await conn.execute('''
                        CREATE UNIQUE INDEX IF NOT EXISTS market_data_symbol_time_tradeid_idx
                        ON market_data(symbol, time, trade_id);
                    ''')
                    await conn.execute('''
                        SELECT create_hypertable(
                            'market_data',
                            'time',
                            if_not_exists => TRUE,
                            chunk_time_interval => INTERVAL '1 day'
                        );
                    ''')

                    # --- Trades Table ---
                    await conn.execute('''
                        CREATE TABLE IF NOT EXISTS trades (
                            symbol VARCHAR NOT NULL,
                            time TIMESTAMPTZ NOT NULL,
                            side TEXT NOT NULL,
                            price DOUBLE PRECISION NOT NULL,
                            quantity DOUBLE PRECISION NOT NULL,
                            PRIMARY KEY (symbol, time, side, price)
                        );
                    ''')
                    await conn.execute(
                        "ALTER TABLE trades ADD COLUMN IF NOT EXISTS symbol VARCHAR NOT NULL DEFAULT 'BTCUSD';"
                    )
                    await conn.execute('''
                        CREATE UNIQUE INDEX IF NOT EXISTS trades_symbol_time_side_price_idx
                        ON trades(symbol, time, side, price);
                    ''')

                    # --- Candles Table ---
                    await conn.execute('''
                        CREATE TABLE IF NOT EXISTS candles (
                            symbol VARCHAR NOT NULL,
                            time TIMESTAMPTZ NOT NULL,
                            open DOUBLE PRECISION NOT NULL,
                            high DOUBLE PRECISION NOT NULL,
                            low DOUBLE PRECISION NOT NULL,
                            close DOUBLE PRECISION NOT NULL,
                            volume DOUBLE PRECISION NOT NULL,
                            PRIMARY KEY (symbol, time)
                        );
                    ''')
                    await conn.execute(
                        "ALTER TABLE candles ADD COLUMN IF NOT EXISTS symbol VARCHAR NOT NULL DEFAULT 'BTCUSD';"
                    )
                    # Attempt to add the desired primary key; if one already exists, catch the error and skip.
                    await conn.execute('''
                    DO $$
                    BEGIN
                        BEGIN
                            ALTER TABLE candles ADD CONSTRAINT candles_symbol_time_pkey PRIMARY KEY (symbol, time);
                        EXCEPTION WHEN SQLSTATE '42P16' THEN
                            RAISE NOTICE 'Primary key already exists, skipping';
                        END;
                    END $$;
                    ''')
                    await conn.execute('''
                        CREATE UNIQUE INDEX IF NOT EXISTS candles_symbol_time_idx
                        ON candles(symbol, time);
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
                logger.error("Database connection failed", error=str(e))
                backoff = cls._base_delay ** attempt
                await asyncio.sleep(backoff)
        logger.error("All DB connection attempts failed. Check logs for DNS or Postgres issues.")

    @classmethod
    async def close(cls):
        if cls._pool:
            await cls._pool.close()
            logger.info("Database connection pool closed")

    @classmethod
    async def execute(cls, query, *args):
        if not cls._pool:
            raise RuntimeError("Database not initialized. Call Database.initialize() first.")
        async with cls._pool.acquire() as conn:
            return await conn.execute(query, *args)

    @classmethod
    async def fetch(cls, query, *args):
        if not cls._pool:
            raise RuntimeError("Database not initialized. Call Database.initialize() first.")
        async with cls._pool.acquire() as conn:
            return await conn.fetch(query, *args)

    @classmethod
    async def fetchrow(cls, query, *args):
        if not cls._pool:
            raise RuntimeError("Database not initialized. Call Database.initialize() first.")
        async with cls._pool.acquire() as conn:
            return await conn.fetchrow(query, *args)

    @classmethod
    async def fetchval(cls, query, *args):
        if not cls._pool:
            raise RuntimeError("Database not initialized. Call Database.initialize() first.")
        async with cls._pool.acquire() as conn:
            return await conn.fetchval(query, *args)
