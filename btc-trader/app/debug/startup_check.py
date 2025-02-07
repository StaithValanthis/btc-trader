from structlog import get_logger
from app.core.config import Config
from app.core.database import Database
from pybit.unified_trading import HTTP
import asyncio

logger = get_logger(__name__)

class StartupChecker:
    @classmethod
    async def run_checks(cls):
        """Run all automated startup checks"""
        logger.info("Running startup checks...")
        await cls._check_env_vars()
        await cls._check_db_connection()
        await cls._check_bybit_connection()
        logger.info("All startup checks passed")

    @classmethod
    async def _check_env_vars(cls):
        """Verify required environment variables"""
        required = [
            'BYBIT_API_KEY', 'BYBIT_API_SECRET',
            'DB_HOST', 'DB_NAME', 'DB_USER'
        ]
        missing = [var for var in required if not getattr(Config, var, None)]
        if missing:
            raise EnvironmentError(f"Missing environment variables: {missing}")

    @classmethod
    async def _check_db_connection(cls):
        """Test database connectivity"""
        try:
            await Database.get_pool()
            await Database.fetch("SELECT 1")
        except Exception as e:
            raise ConnectionError(f"Database connection failed: {str(e)}")

    @classmethod
    async def _check_bybit_connection(cls):
        """Test Bybit API connectivity"""
        try:
            session = HTTP(
                testnet=Config.BYBIT_CONFIG['testnet'],
                api_key=Config.BYBIT_CONFIG['api_key'],
                api_secret=Config.BYBIT_CONFIG['api_secret']
            )
            await asyncio.to_thread(
                session.get_server_time
            )
        except Exception as e:
            raise ConnectionError(f"Bybit API connection failed: {str(e)}")