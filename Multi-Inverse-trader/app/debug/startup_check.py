import asyncio
from structlog import get_logger
from app.core.config import Config
from app.core.database import Database
from pybit.unified_trading import HTTP
from app.services.backfill_service import maybe_backfill_candles

logger = get_logger(__name__)

class StartupChecker:
    @classmethod
    async def run_checks(cls, symbol):
        logger.info(f"Running startup checks for {symbol}...")
        cls._check_env_vars()
        if Database._pool is None:
            await Database.initialize()
        await cls._check_db_connection()
        await cls._check_bybit_connection(symbol)
        await maybe_backfill_candles(symbol=symbol, min_rows=2000, interval=1, days_to_fetch=10)
        logger.info(f"All startup checks passed for {symbol}")

    @classmethod
    def _check_env_vars(cls):
        if not Config.BYBIT_CONFIG['api_key'] or not Config.BYBIT_CONFIG['api_secret']:
            raise EnvironmentError("BYBIT_API_KEY or BYBIT_API_SECRET not set.")

    @classmethod
    async def _check_db_connection(cls):
        try:
            val = await Database.fetchval("SELECT 1")
            assert val == 1
            logger.info("Database connection successful")
        except Exception as e:
            raise ConnectionError(f"Database connection failed: {e}")

    @classmethod
    async def _check_bybit_connection(cls, symbol):
        try:
            session = HTTP(
                testnet=Config.BYBIT_CONFIG['testnet'],
                api_key=Config.BYBIT_CONFIG['api_key'],
                api_secret=Config.BYBIT_CONFIG['api_secret']
            )
            await asyncio.to_thread(session.get_server_time)
            logger.info(f"Bybit API connection successful for {symbol}")
        except Exception as e:
            raise ConnectionError(f"Bybit API connection failed: {e}")
