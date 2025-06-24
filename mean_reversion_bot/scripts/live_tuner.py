#!/usr/bin/env python3
import time, json, redis, asyncio
from app.core.config import Config

r = redis.Redis(host="localhost",port=6379,db=0,decode_responses=True)

async def monitor():
    while True:
        raw = r.lindex("mr_perf",0)
        if raw:
            data=json.loads(raw)
            if data["sharpe"]<0.5:
                cur=Config["TRADING"]["SL_PCT"]
                new=max(0.001,cur-0.001)
                Config["TRADING"]["SL_PCT"]=new
                print(f"[TUNER] SL_PCT→{new}")
        await asyncio.sleep(3600)

if __name__=="__main__":
    asyncio.run(monitor())
