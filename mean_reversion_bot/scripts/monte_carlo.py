#!/usr/bin/env python3
import random, asyncio, pandas as pd
from scripts.backtester import fetch, compute
from app.core.config import Config

async def main():
    df=await fetch(None,5000)
    mc=Config["PARAM_RANGES"]["MC_TRIALS"]
    rng=Config["PARAM_RANGES"]["MC_RANGE"]
    sls=Config["PARAM_RANGES"]["SL_PCT"]
    results=[]
    for _ in range(mc):
        w=random.randint(*rng["window"])
        d=round(random.uniform(*rng["dev"]),2)
        rl=random.randint(*rng["rsi_long"])
        rs=random.randint(*rng["rsi_short"])
        sl=random.choice(sls)
        m=compute(df,(w,d,rl,rs,sl))
        results.append({"window":w,"dev":d,"rsi_long":rl,"rsi_short":rs,"sl_pct":sl,**m})
    out=pd.DataFrame(results).sort_values(["sharpe","pnl"],ascending=False).head(10)
    print(out)

if __name__=="__main__":
    asyncio.run(main())
