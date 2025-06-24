#!/usr/bin/env python3
import os, json, asyncio
from app.core.config import Config
from scripts.backtester import main as backtest_main

PARAM_FILE=os.path.join(os.path.dirname(__file__),"../best_params_by_symbol.json")

async def update(symbols):
    try:
        with open(PARAM_FILE) as f: store=json.load(f)
    except: store={}
    for s in symbols:
        os.system(f"python3 scripts/backtester.py --symbol {s} --limit 10000")
        with open("best_params.json") as f: best=json.load(f)
        store[s]=best
    store={k:v for k,v in store.items() if k in symbols}
    with open(PARAM_FILE,"w") as f: json.dump(store,f,indent=2)
    print("Updated",PARAM_FILE)

if __name__=="__main__":
    syms=Config["TRADING"]["symbols"]
    asyncio.run(update(syms))
