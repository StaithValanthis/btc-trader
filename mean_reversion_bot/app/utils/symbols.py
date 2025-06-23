import aiohttp
from structlog import get_logger
from app.core.config import Config

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
