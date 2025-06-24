import asyncio, time
from structlog import get_logger
from app.core.database import init_db, db_fetchval
from app.core.config import Config
from pybit.unified_trading import HTTP

logger = get_logger(__name__)

def check_env():
    if not Config["BYBIT"]["api_key"] or not Config["BYBIT"]["api_secret"]:
        raise EnvironmentError("BYBIT_API_KEY / SECRET unset")

async def test_db(pool):
    val = await db_fetchval(pool, "SELECT 1")
    assert val == 1
    logger.info("DB test OK")

async def test_bybit():
    sess = HTTP(
        testnet   = Config["BYBIT"]["testnet"],
        api_key   = Config["BYBIT"]["api_key"],
        api_secret= Config["BYBIT"]["api_secret"],
    )
    await asyncio.to_thread(sess.get_server_time)
    logger.info("Bybit API OK")

async def validate_symbols():
    http = HTTP(
        testnet   = Config["BYBIT"]["testnet"],
        api_key   = Config["BYBIT"]["api_key"],
        api_secret= Config["BYBIT"]["api_secret"],
    )
    valid=[]
    for s in Config["TRADING"]["symbols"]:
        r = http.get_instruments_info(category=Config["BYBIT"]["category"],symbol=s)
        if r.get("retCode",0)==0:
            valid.append(s)
        else:
            logger.warning("Bad symbol",symbol=s)
    Config["TRADING"]["symbols"]=valid

async def run_startup():
    check_env()
    pool = await init_db()
    await test_db(pool)
    await test_bybit()
    await validate_symbols()
    # backfill omitted here
    logger.info("Startup checks passed")
    return pool
