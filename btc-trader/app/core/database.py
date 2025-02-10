# app/core/database.py
import asyncpg
from structlog import get_logger
from app.core.config import Config

logger = get_logger(__name__)

class Database:
    _pool = None

    @classmethod
    async def initialize(cls):
        """Initialize database connection pool and schema"""
        try:
            cls._pool = await asyncpg.create_pool(
                **Config.DB_CONFIG,
                min_size=5,
                max_size=20,
                timeout=30,
                command_timeout=60
            )
            
            async with cls._pool.acquire() as conn:
                # Create TimescaleDB extension
                await conn.execute('CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;')
                
                # Create market_data table
                await conn.execute('''
                    CREATE TABLE IF NOT EXISTS market_data (
                        time TIMESTAMPTZ NOT NULL,
                        price DOUBLE PRECISION NOT NULL,
                        volume DOUBLE PRECISION NOT NULL,
                        UNIQUE (time, price, volume)
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
                
            logger.info("Database initialized")
            
        except Exception as e:
            logger.error("Database initialization failed", error=str(e))
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
        async with cls._pool.acquire() as conn:
            return await conn.execute(query, *args)

    @classmethod
    async def fetch(cls, query, *args):
        """Fetch multiple rows"""
        async with cls._pool.acquire() as conn:
            return await conn.fetch(query, *args)

    @classmethod
    async def fetchrow(cls, query, *args):
        """Fetch a single row"""
        async with cls._pool.acquire() as conn:
            return await conn.fetchrow(query, *args)

    @classmethod
    async def fetchval(cls, query, *args):
        """Fetch a single value"""
        async with cls._pool.acquire() as conn:
            return await conn.fetchval(query, *args)