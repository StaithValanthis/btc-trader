import asyncpg
import os
from datetime import datetime, timezone
from structlog import get_logger
from app.core.config import Config

logger = get_logger(__name__)

class Database:
    _pool = None

    @classmethod
    async def get_pool(cls):
        if not cls._pool:
            try:
                cls._pool = await asyncpg.create_pool(
                    **Config.DB_CONFIG,
                    min_size=5,
                    max_size=20,
                    timeout=30
                )
                logger.info("Database pool created successfully")
            except Exception as e:
                logger.error("Failed to create database pool", error=str(e))
                raise
        return cls._pool

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
            logger.info("Database initialized successfully")

    @classmethod
    async def close(cls):
        if cls._pool:
            await cls._pool.close()
            logger.info("Database pool closed")

    @classmethod
    async def execute(cls, query, *args):
        pool = await cls.get_pool()
        try:
            return await pool.execute(query, *args)
        except Exception as e:
            logger.error("Database query failed", query=query, error=str(e))
            raise