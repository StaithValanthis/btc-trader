import aiohttp, asyncio
from structlog import get_logger
from app.core.config import Config

logger = get_logger(__name__)

async def fetch_top_symbols(n=30):
    base = "api-testnet.bybit.com" if Config["BYBIT"]["testnet"] else "api.bybit.com"
    url = f"https://{base}/v5/market/tickers"
    params={"category":Config["BYBIT"]["category"]}
    try:
        async with aiohttp.ClientSession() as s:
            async with s.get(url,params=params,timeout=10) as resp:
                data = await resp.json()
    except Exception as e:
        logger.error("Ticker fetch failed",error=str(e))
        return []
    if data.get("retCode")!=0:
        logger.error("Ticker API error",**data)
        return []
    lst = data["result"]["list"]
    lst.sort(key=lambda x: float(x.get("volume24h",0)),reverse=True)
    top = [t["symbol"] for t in lst[:n]]
    logger.info("Top symbols",symbols=top)
    return top

async def filter_tradable(symbols,lev):
    from pybit.unified_trading import HTTP
    http = HTTP(**Config["BYBIT"])
    async def ok(s):
        try:
            r= await asyncio.to_thread(http.get_instruments_info,category=Config["BYBIT"]["category"],symbol=s)
            i = r["result"]["list"][0]
            return i["status"]=="Trading" and float(i["leverageFilter"]["maxLeverage"])>=lev
        except:
            return False
    res = await asyncio.gather(*(ok(s) for s in symbols))
    out=[s for s,f in zip(symbols,res) if f]
    logger.info("Tradable",symbols=out)
    return out
