# File: app/debug/startup_check.py
import asyncio
from structlog import get_logger
from app.core import Config, Database
from pybit.unified_trading import HTTP

logger = get_logger(__name__)

class StartupChecker:
    @classmethod
    async def run_checks(cls):
        """Minimal startup checks: environment, DB connectivity, Bybit API."""
        logger.info("Running startup checks...")
        cls._check_env_vars()

        if Database._pool is None:
            await Database.initialize()

        await cls._check_db_connection()
        await cls._check_bybit_connection()

        logger.info("All startup checks passed")

    @classmethod
    def _check_env_vars(cls):
        # Basic check for required keys
        if not Config.BYBIT_CONFIG['api_key'] or not Config.BYBIT_CONFIG['api_secret']:
            raise EnvironmentError("BYBIT_API_KEY or BYBIT_API_SECRET not set.")

    @classmethod
    async def _check_db_connection(cls):
        """Test database connectivity."""
        try:
            val = await Database.fetchval("SELECT 1")
            assert val == 1
            logger.info("Database connection successful")
        except Exception as e:
            raise ConnectionError(f"Database connection failed: {str(e)}")

    @classmethod
    async def _check_bybit_connection(cls):
        """Test Bybit API connectivity via get_server_time."""
        try:
            session = HTTP(
                testnet=Config.BYBIT_CONFIG['testnet'],
                api_key=Config.BYBIT_CONFIG['api_key'],
                api_secret=Config.BYBIT_CONFIG['api_secret']
            )
            await asyncio.to_thread(session.get_server_time)
            logger.info("Bybit API connection successful")
        except Exception as e:
            raise ConnectionError(f"Bybit API connection failed: {str(e)}")
