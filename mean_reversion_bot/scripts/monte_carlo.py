#!/usr/bin/env python3
"""
Monte Carlo sampler for BB+RSI+SL mean reversion.
"""
import random
import pandas as pd
import asyncio
from scripts.backtester import fetch_candles, simulate
from app.core.config import Config

async def main():
    df = await fetch_candles(limit=5000)
    mc     = Config.PARAM_RANGES['MC_TRIALS']
    ranges = Config.PARAM_RANGES
    sls    = ranges['SL_PCT']

    results = []
    for _ in range(mc):
        w  = random.randint(*ranges['MC_RANGE']['window'])
        d  = round(random.uniform(*ranges['MC_RANGE']['dev']), 2)
        rl = random.randint(*ranges['MC_RANGE']['rsi_long'])
        rs = random.randint(*ranges['MC_RANGE']['rsi_short'])
        sl = random.choice(sls)
        m  = simulate(df, w, d, rl, rs, sl)
        results.append({
            'window':    w,
            'dev':       d,
            'rsi_long':  rl,
            'rsi_short': rs,
            'sl_pct':    sl,
            'pnl':       m['pnl'],
            'sharpe':    m['sharpe']
        })

    df_out = pd.DataFrame(results).sort_values(['sharpe','pnl'], ascending=False).head(10)
    print(df_out)

if __name__ == '__main__':
    asyncio.run(main())
