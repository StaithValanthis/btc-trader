# app/core/database.py
# app/core/database.py
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
        Initialize database connection pool and create tables with TimescaleDB optimizations
        Initialize database connection pool and create tables with TimescaleDB optimizations
        """
        for attempt in range(cls._max_retries):
            try:
                logger.info(f"Database connection attempt {attempt+1}/{cls._max_retries}")
                logger.info(f"Database connection attempt {attempt+1}/{cls._max_retries}")
                cls._pool = await asyncpg.create_pool(
                    **Config.DB_CONFIG,
                    min_size=5,
                    max_size=20,
                    timeout=30,
                    command_timeout=60
                )

                async with cls._pool.acquire() as conn:
                    # Create TimescaleDB extension
                    # Create TimescaleDB extension
                    await conn.execute('CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;')

                    # Market data table
                    # Market data table
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

                    # Trades table
                    await conn.execute('''
                        CREATE TABLE IF NOT EXISTS trades (
                            time TIMESTAMPTZ NOT NULL,
                            side TEXT NOT NULL,
                            price DOUBLE PRECISION NOT NULL,
                            quantity DOUBLE PRECISION NOT NULL,
                            PRIMARY KEY (time, side, price)
                        );
                    ''')
                    await conn.execute('''
                        SELECT create_hypertable(
                            'trades',
                            'time',
                            if_not_exists => TRUE,
                            chunk_time_interval => INTERVAL '1 day'
                            if_not_exists => TRUE,
                            chunk_time_interval => INTERVAL '1 day'
                        );
                    ''')

                    # Configure compression
                    await cls._configure_compression(conn)

                logger.info("Database initialized successfully")
                    # Configure compression
                    await cls._configure_compression(conn)

                logger.info("Database initialized successfully")
                return

            except Exception as e:
                if attempt == cls._max_retries - 1:
                    logger.error("Database initialization failed", error=str(e))
                    logger.error("Database initialization failed", error=str(e))
                    raise
                wait_time = cls._base_delay ** attempt
                logger.warning("Retrying database connection", wait_seconds=wait_time)
                await asyncio.sleep(wait_time)

    @classmethod
    async def _configure_compression(cls, conn):
        """Configure TimescaleDB compression policies safely"""
        try:
            # Market data compression
            await conn.execute('''
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM timescaledb_information.compression_settings 
                        WHERE hypertable_name = 'market_data'
                    ) THEN
                        ALTER TABLE market_data SET (
                            timescaledb.compress,
                            timescaledb.compress_orderby = 'time DESC',
                            timescaledb.compress_segmentby = 'trade_id'
                        );
                    END IF;
                    
                    PERFORM add_compression_policy(
                        'market_data', 
                        INTERVAL '7 days',
                        if_not_exists => true
                    );
                END $$;
            ''')

            # Trades compression
            await conn.execute('''
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM timescaledb_information.compression_settings 
                        WHERE hypertable_name = 'trades'
                    ) THEN
                        ALTER TABLE trades SET (
                            timescaledb.compress,
                            timescaledb.compress_orderby = 'time DESC'
                        );
                    END IF;
                    
                    PERFORM add_compression_policy(
                        'trades',
                        INTERVAL '7 days',
                        if_not_exists => true
                    );
                END $$;
            ''')
            logger.info("Compression configured safely")
        except Exception as e:
            logger.error("Compression configuration error", error=str(e))
            raise
        
                wait_time = cls._base_delay ** attempt
                logger.warning("Retrying database connection", wait_seconds=wait_time)
                await asyncio.sleep(wait_time)

    @classmethod
    async def _configure_compression(cls, conn):
        """Configure TimescaleDB compression policies safely"""
        try:
            # Market data compression
            await conn.execute('''
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM timescaledb_information.compression_settings 
                        WHERE hypertable_name = 'market_data'
                    ) THEN
                        ALTER TABLE market_data SET (
                            timescaledb.compress,
                            timescaledb.compress_orderby = 'time DESC',
                            timescaledb.compress_segmentby = 'trade_id'
                        );
                    END IF;
                    
                    PERFORM add_compression_policy(
                        'market_data', 
                        INTERVAL '7 days',
                        if_not_exists => true
                    );
                END $$;
            ''')

            # Trades compression
            await conn.execute('''
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM timescaledb_information.compression_settings 
                        WHERE hypertable_name = 'trades'
                    ) THEN
                        ALTER TABLE trades SET (
                            timescaledb.compress,
                            timescaledb.compress_orderby = 'time DESC'
                        );
                    END IF;
                    
                    PERFORM add_compression_policy(
                        'trades',
                        INTERVAL '7 days',
                        if_not_exists => true
                    );
                END $$;
            ''')
            logger.info("Compression configured safely")
        except Exception as e:
            logger.error("Compression configuration error", error=str(e))
            raise
        
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
            raise RuntimeError("Database not initialized. Call Database.initialize() first.")
            raise RuntimeError("Database not initialized. Call Database.initialize() first.")
        async with cls._pool.acquire() as conn:
            return await conn.execute(query, *args)

    @classmethod
    async def fetch(cls, query, *args):
        """Fetch multiple rows"""
        if not cls._pool:
            raise RuntimeError("Database not initialized. Call Database.initialize() first.")
            raise RuntimeError("Database not initialized. Call Database.initialize() first.")
        async with cls._pool.acquire() as conn:
            return await conn.fetch(query, *args)

    @classmethod
    async def fetchrow(cls, query, *args):
        """Fetch a single row"""
        if not cls._pool:
            raise RuntimeError("Database not initialized. Call Database.initialize() first.")
            raise RuntimeError("Database not initialized. Call Database.initialize() first.")
        async with cls._pool.acquire() as conn:
            return await conn.fetchrow(query, *args)

    @classmethod
    async def fetchval(cls, query, *args):
        """Fetch a single value"""
        if not cls._pool:
            raise RuntimeError("Database not initialized. Call Database.initialize() first.")
            raise RuntimeError("Database not initialized. Call Database.initialize() first.")
        async with cls._pool.acquire() as conn:
            return await conn.fetchval(query, *args)