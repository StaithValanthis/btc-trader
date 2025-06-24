#!/usr/bin/env python3
import os, json, asyncio
from app.core.config import Config
from scripts.backtester import main as backtest_main

PARAMS_FILE = os.path.join(os.path.dirname(__file__), "../best_params_by_symbol.json")

async def update_best_params(symbols):
    # Load previous params
    try:
        with open(PARAMS_FILE, "r") as f:
            best_params = json.load(f)
    except Exception:
        best_params = {}

    for symbol in symbols:
        # Run a backtest for this symbol (or load last known best for this period)
        # Here we assume backtester.py supports --symbol and writes best to a tmp file
        os.system(f"python3 scripts/backtester.py --symbol {symbol} --limit 10000")
        with open("best_params.json", "r") as f:
            best = json.load(f)
        best_params[symbol] = best

    # Remove entries for symbols no longer present
    best_params = {sym: best_params[sym] for sym in symbols if sym in best_params}
    with open(PARAMS_FILE, "w") as f:
        json.dump(best_params, f, indent=2)
    print(f"Updated {PARAMS_FILE} for {len(symbols)} symbols.")

if __name__ == "__main__":
    # You may want to get symbols from Config or dynamically from DB/your symbol refresh logic
    symbols = Config.TRADING_CONFIG["symbols"]
    asyncio.run(update_best_params(symbols))
