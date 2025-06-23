import requests
from structlog import get_logger
from app.core.config import Config

logger = get_logger(__name__)

def fetch_top_symbols(n: int = 30) -> list[str]:
    """
    Fetch the top `n` symbols on Bybit by 24 h volume.
    Uses live or testnet endpoint based on Config.BYBIT_CONFIG['testnet'].
    """
    base = "https://api-testnet.bybit.com" if Config.BYBIT_CONFIG["testnet"] else "https://api.bybit.com"
    url = f"{base}/v5/market/tickers"
    params = {"category": Config.BYBIT_CONFIG["category"]}

    try:
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
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
    # Sort descending by 24 h volume (field is "volume24h")
    tickers.sort(key=lambda t: float(t.get("volume24h", 0.0)), reverse=True)
    top = [t["symbol"] for t in tickers[:n]]

    logger.info("Fetched top symbols by volume", symbols=top)
    return top
