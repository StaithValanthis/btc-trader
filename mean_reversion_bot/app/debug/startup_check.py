# File: app/debug/startup_check.py

import asyncio
from structlog import get_logger
from app.core.config import Config
from app.core.database import Database
from pybit.unified_trading import HTTP
from app.services.backfill_service import maybe_backfill_candles

logger = get_logger(__name__)

class StartupChecker:
    @classmethod
    async def run_checks(cls):
        logger.info("Running startup checks...")
        cls._check_env_vars()

        # 1) Database
        if Database._pool is None:
            await Database.initialize()
        await cls._check_db_connection()

        # 2) Bybit connectivity
        session = HTTP(
            testnet=Config.BYBIT_CONFIG['testnet'],
            api_key=Config.BYBIT_CONFIG['api_key'],
            api_secret=Config.BYBIT_CONFIG['api_secret'],
        )
        await cls._check_bybit_connection(session)

        # 3) Prune out any invalid symbols
        cls._validate_symbols(session)

        # 4) Backfill candle history for all remaining symbols
        #    → signature is maybe_backfill_candles(min_rows, interval, days)
        #    → here we supply those three positionally
        await maybe_backfill_candles(
            1000,   # min_rows
            1,      # interval (1-minute candles)
            7      # days of history
        )

        logger.info("All startup checks passed")

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
    async def _check_bybit_connection(cls, session: HTTP):
        try:
            # throws if invalid creds or endpoint
            await asyncio.to_thread(session.get_server_time)
            logger.info("Bybit API connection successful")
        except Exception as e:
            raise ConnectionError(f"Bybit API connection failed: {e}")

    @classmethod
    def _validate_symbols(cls, session: HTTP):
        """
        Ping instruments-info for each symbol in our config and drop any that fail.
        """
        valid = []
        for s in Config.TRADING_CONFIG['symbols']:
            try:
                resp = session.get_instruments_info(
                    category=Config.BYBIT_CONFIG.get('category', 'LinearPerpetual'),
                    symbol=s
                )
                if resp.get('retCode', 0) == 0:
                    valid.append(s)
                else:
                    logger.warning("Skipping invalid symbol", symbol=s, error=resp.get('retMsg'))
            except Exception as e:
                logger.warning("Skipping invalid symbol", symbol=s, error=str(e))

        dropped = set(Config.TRADING_CONFIG['symbols']) - set(valid)
        if dropped:
            logger.info("Filtered out invalid symbols", dropped=list(dropped))
        Config.TRADING_CONFIG['symbols'] = valid
