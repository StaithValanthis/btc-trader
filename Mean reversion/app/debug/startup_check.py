import asyncio
from structlog import get_logger
from app.core.config import Config
from app.core.database import Database
from pybit.unified_trading import HTTP
from app.services.backfill_service import maybe_backfill_candles

logger = get_logger(__name__)

class StartupChecker:
    @classmethod
    async def _check_env_vars(cls):
        if not Config.BYBIT_CONFIG['api_key'] or not Config.BYBIT_CONFIG['api_secret']:
            raise EnvironmentError("BYBIT_API_KEY or BYBIT_API_SECRET not set.")

    @classmethod
    async def _check_db_connection(cls):
        try:
            val = await Database.fetchval("SELECT 1")
            assert val == 1
            logger.info("DB connection OK")
        except Exception as e:
            raise ConnectionError(f"DB error: {e}")

    @classmethod
    async def _check_bybit_connection(cls):
        session = HTTP(**Config.BYBIT_CONFIG)
        await asyncio.to_thread(session.get_server_time)
        logger.info("Bybit API OK")

    @classmethod
    async def _determine_optimal_params(cls):
        from scripts.backtester import fetch_candles, simulate
        df = await fetch_candles(limit=2000)
        pr = Config.PARAM_RANGES
        best = {'sharpe': -float('inf')}
        for w in pr['BB_WINDOW']:
            for d in pr['BB_DEV']:
                for rl in pr['RSI_LONG']:
                    for rs in pr['RSI_SHORT']:
                        m = simulate(df, w, d, rl, rs)
                        if m['sharpe'] > best['sharpe']:
                            best = {'window':w,'dev':d,'rsi_long':rl,'rsi_short':rs,'sharpe':m['sharpe']}
        return best

    @classmethod
    async def run_checks(cls):
        logger.info("Startup checks")
        await cls._check_env_vars()
        if Database._pool is None:
            await Database.initialize()
        await cls._check_db_connection()
        await cls._check_bybit_connection()
        await maybe_backfill_candles(min_rows=2000, symbol="BTCUSD", interval=1, days_to_fetch=30)
        try:
            opt = await cls._determine_optimal_params()
            tc = Config.TRADING_CONFIG
            tc['BB_WINDOW']  = opt['window']
            tc['BB_DEV']     = opt['dev']
            tc['RSI_LONG']   = opt['rsi_long']
            tc['RSI_SHORT']  = opt['rsi_short']
            logger.info("Optimized params", **opt)
        except Exception as e:
            logger.warning("Opt failed; using defaults", error=str(e))
        logger.info("Startup complete")
