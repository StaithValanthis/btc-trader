# File: app/debug/startup_check.py

from datetime import datetime, timezone 
from structlog import get_logger
from app.core import Config, Database
from pybit.unified_trading import HTTP
import asyncio

logger = get_logger(__name__)

class StartupChecker:
    @classmethod
    async def run_checks(cls):
        """Run all automated startup checks, including DB init if needed."""
        logger.info("Running comprehensive startup checks...")

        # 1) ENV var checks
        await cls._check_env_vars()

        # 2) Ensure DB is initialized
        if Database._pool is None:
            logger.info("Database pool not initialized; calling Database.initialize()")
            await Database.initialize()

        # 3) DB checks
        await cls._check_db_connection()
        await cls._check_db_schema()

        # 4) Bybit connectivity
        await cls._check_bybit_connection()

        # 5) Data flow test
        await cls._check_data_flow()

        logger.info("All startup checks passed")

    @classmethod
    async def _check_env_vars(cls):
        required = [
            ('BYBIT_API_KEY', Config.BYBIT_CONFIG['api_key']),
            ('BYBIT_API_SECRET', Config.BYBIT_CONFIG['api_secret']),
            ('DB_HOST', Config.DB_CONFIG['host']),
            ('DB_NAME', Config.DB_CONFIG['database']),
            ('DB_USER', Config.DB_CONFIG['user'])
        ]
        missing = [name for name, value in required if not value]
        if missing:
            raise EnvironmentError(f"Missing configuration values for: {missing}")

    @classmethod
    async def _check_db_connection(cls):
        """Test database connectivity."""
        try:
            await Database.fetchval("SELECT 1")
            logger.info("Database connection successful")
        except Exception as e:
            raise ConnectionError(f"Database connection failed: {str(e)}")

    @classmethod
    async def _check_db_schema(cls):
        """Verify database schema."""
        try:
            columns = await Database.fetch('''
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = 'market_data'
            ''')
            logger.info("Market_data table schema", columns=columns)

            hypertables = await Database.fetch('''
                SELECT * FROM timescaledb_information.hypertables 
                WHERE hypertable_name = 'market_data'
            ''')
            logger.info("Hypertable status", hypertable=hypertables[0] if hypertables else None)

        except Exception as e:
            logger.error("Database schema check failed", error=str(e))
            raise

    @classmethod
    async def _check_bybit_connection(cls):
        """Test Bybit API connectivity."""
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

    @classmethod
    async def _check_data_flow(cls):
        """Verify data pipeline (insert + delete)."""
        try:
            test_time = datetime.now(timezone.utc)
            test_trade_id = "startup_test"  # dummy trade_id

            # Insert with dummy trade_id
            await Database.execute('''
                INSERT INTO market_data (time, trade_id, price, volume)
                VALUES ($1, $2, $3, $4)
            ''', test_time, test_trade_id, 50000.0, 1.0)

            # Now delete that row
            await Database.execute('''
                DELETE FROM market_data 
                WHERE time = $1 AND trade_id = $2
            ''', test_time, test_trade_id)

            logger.info("Data flow check successful")

        except Exception as e:
            logger.error("Data flow check failed", error=str(e))
            raise
