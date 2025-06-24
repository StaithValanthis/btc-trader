#!/usr/bin/env python3
import argparse, asyncio, itertools, json
import pandas as pd, numpy as np
from app.core.database import init_db, db_fetch
from app.core.config import Config

def compute(df,params):
    window,dev,rl,rs,sl = params
    m = df["close"].rolling(window).mean()
    std = df["close"].rolling(window).std()
    upper,lower = m+dev*std, m-dev*std
    delta=df["close"].diff()
    gain,loss = delta.clip(lower=0), -delta.clip(upper=0)
    avg_g,avg_l = gain.rolling(rl).mean(), loss.rolling(rl).mean()
    rsi=100-(100/(1+avg_g/avg_l))
    pos,entry,stop,rets=0,0,0,[]
    for t in df.index:
        price=df.at[t,"close"]
        if pos==0:
            if price<lower.at[t] and rsi.at[t]<rl:
                pos,entry,stop=1,price,price*(1-sl)
            elif price>upper.at[t] and rsi.at[t]>rs:
                pos,entry,stop=-1,price,price*(1+sl)
        elif pos==1:
            if price>=m.at[t]:
                rets.append((price-entry)/entry); pos=0
            elif price<=stop:
                rets.append((stop-entry)/entry); pos=0
        elif pos==-1:
            if price<=m.at[t]:
                rets.append((entry-price)/entry); pos=0
            elif price>=stop:
                rets.append((entry-stop)/entry); pos=0
    if not rets: return {"pnl":0,"sharpe":0}
    arr=np.array(rets)
    sr = (arr.mean()/arr.std()*np.sqrt(len(arr))) if arr.std()>0 else 0
    return {"pnl":arr.sum(),"sharpe":sr}

async def fetch(symbol,limit):
    pool = await init_db()
    q="SELECT time,open,high,low,close FROM candles"
    args=[]
    if symbol:
        q+=" WHERE symbol=$1"; args.append(symbol)
    q+=" ORDER BY time"
    if limit: q+=f" LIMIT {limit}"
    rows=await db_fetch(pool,q,*args)
    df=pd.DataFrame(rows,columns=["time","open","high","low","close"])
    df.time=pd.to_datetime(df.time); df.set_index("time",inplace=True)
    return df

async def main():
    p=argparse.ArgumentParser()
    p.add_argument("--symbol",type=str)
    p.add_argument("--limit",type=int,default=None)
    args=p.parse_args()
    df=await fetch(args.symbol,args.limit)
    pr=Config["PARAM_RANGES"]
    combos=itertools.product(pr["BB_WINDOW"],pr["BB_DEV"],pr["RSI_LONG"],pr["RSI_SHORT"],pr["SL_PCT"])
    results=[]
    for c in combos:
        m=compute(df,c)
        results.append(dict(zip(["window","dev","rsi_long","rsi_short","sl_pct"],c),**m))
    out=pd.DataFrame(results).sort_values(["sharpe","pnl"],ascending=False).head(10)
    print(out)
    if not out.empty:
        best=out.iloc[0].to_dict()
        with open("best_params.json","w") as f: json.dump(best,f,indent=2)
        print("Wrote best_params.json")

if __name__=="__main__":
    asyncio.run(main())
