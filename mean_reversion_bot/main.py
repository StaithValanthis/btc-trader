import asyncio
from app.core.config import Config
from app.debug.startup_check import run_startup
from app.core.bybit_ws    import start_bybit_ws, ws_url
from app.services.candle_service import run_candle_aggregator
from app.services.trade_service  import on_candle_msg

async def main():
    pool = await run_startup()
    symbols = Config["TRADING"]["symbols"]
    from app.utils.symbols import fetch_top_symbols, filter_tradable
    if not Config["BYBIT"]["testnet"]:
        syms = await fetch_top_symbols(30)
        symbols = await filter_tradable(syms,Config["TRADING"]["leverage"])

    # prepare per-symbol state and handlers
    state_map = {}
    for sym in symbols:
        http = HTTP(**Config["BYBIT"])
        inst = await asyncio.to_thread(http.get_instruments_info,category=Config["BYBIT"]["category"],symbol=sym)
        pf = inst["result"]["list"][0]["priceFilter"]
        lf = inst["result"]["list"][0]["lotSizeFilter"]
        state_map[sym] = {
            "session": http,
            "symbol": sym,
            "cfg":     Config["TRADING"],
            "tick_size": float(pf["tickSize"]),
            "lot_size":  float(lf["qtyStep"]),
            "current_pos": None,
            "entry_price": None,
            "buf": pd.DataFrame()
        }
        asyncio.create_task(run_candle_aggregator(pool,sym))

    async def handler(msg):
        topic = msg.get("topic","")
        for sym,st in state_map.items():
            if topic==f"candle.1.{sym}":
                await on_candle_msg(st,msg["data"])

    # subscribe to all candle streams
    topics = [f"candle.1.{s}" for s in symbols]
    await start_bybit_ws(topics, handler)

if __name__=="__main__":
    asyncio.run(main())
