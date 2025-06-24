import aiohttp
from structlog import get_logger
from app.core.config import Config
from pybit.unified_trading import HTTP
import asyncio

logger = get_logger(__name__)

async def fetch_top_symbols(n: int = 30) -> list[str]:
    """
    Async: Fetch the top `n` symbols on Bybit by 24 h volume.
    """
    base = "https://api-testnet.bybit.com" if Config.BYBIT_CONFIG["testnet"] else "https://api.bybit.com"
    url = f"{base}/v5/market/tickers"
    params = {"category": Config.BYBIT_CONFIG["category"]}

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, timeout=10) as resp:
                data = await resp.json()
    except Exception as e:
        logger.error("Failed to fetch tickers", error=str(e))
        return []

    if data.get("retCode", -1) != 0:
        logger.error(
            "Bybit API error fetching tickers",
            retCode=data.get("retCode"),
            retMsg=data.get("retMsg"),
        )
        return []

    tickers = data["result"]["list"]
    tickers.sort(key=lambda t: float(t.get("volume24h", 0.0)), reverse=True)
    top = [t["symbol"] for t in tickers[:n]]
    logger.info("Fetched top symbols by volume", symbols=top)
    return top

async def filter_tradable_symbols(symbols: list[str], leverage: int) -> list[str]:
    """
    Filters out symbols that are not enabled for perpetual trading with at least the given leverage.
    """
    session = HTTP(
        testnet=Config.BYBIT_CONFIG['testnet'],
        api_key=Config.BYBIT_CONFIG['api_key'],
        api_secret=Config.BYBIT_CONFIG['api_secret'],
    )

    async def check(sym):
        try:
            resp = await asyncio.to_thread(
                session.get_instruments_info,
                category=Config.BYBIT_CONFIG['category'],
                symbol=sym
            )
            lst = resp.get('result', {}).get('list', [])
            if lst:
                item = lst[0]
                # Only keep if status is 'Trading' and contractType is 'LinearPerpetual'
                if (
                    item.get('status') == 'Trading'
                    and item.get('contractType', '') == 'LinearPerpetual'
                ):
                    lev = float(item.get('leverageFilter', {}).get('maxLeverage', 0))
                    if lev >= leverage:
                        return sym
        except Exception as e:
            logger.warning("Error in symbol filter", symbol=sym, error=str(e))
        return None

    results = await asyncio.gather(*[check(s) for s in symbols])
    tradable = [r for r in results if r]
    logger.info("Filtered tradable symbols", symbols=tradable)
    return tradable
