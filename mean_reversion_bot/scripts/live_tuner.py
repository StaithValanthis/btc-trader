#!/usr/bin/env python3
"""
Live tuning of BB_DEV & SL_PCT via Redis.
"""
import time
import json
import redis
import asyncio
from app.core.config import Config

r = redis.Redis(host=Config.DB_CONFIG['host'], port=6379, db=0)

def push_performance(window, dev, rl, rs, sl, pnl, sharpe):
    entry = json.dumps({
        'time':      time.time(),
        'window':    window,
        'dev':       dev,
        'rsi_long':  rl,
        'rsi_short': rs,
        'sl_pct':    sl,
        'pnl':       pnl,
        'sharpe':    sharpe
    })
    r.lpush('mr_perf', entry)
    r.ltrim('mr_perf', 0, 100)

async def monitor():
    """
    Every hour: adjust SL_PCT if Sharpe dips below threshold.
    """
    while True:
        raw = r.lindex('mr_perf', 0)
        if raw:
            data = json.loads(raw)
            if data['sharpe'] < 0.5:
                cur_sl = Config.TRADING_CONFIG.get('SL_PCT', 0.01)
                new_sl = max(0.001, cur_sl - 0.001)
                Config.TRADING_CONFIG['SL_PCT'] = new_sl
                print(f"[TUNER] Adjusted SL_PCT → {new_sl}")
        await asyncio.sleep(3600)

if __name__ == '__main__':
    asyncio.run(monitor())
