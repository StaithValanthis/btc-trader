#!/usr/bin/env python3
"""
Grid backtester for BB+RSI+SL mean reversion.
"""
import asyncio
import itertools
import pandas as pd
import numpy as np
from app.core.database import Database
from app.core.config import Config

async def fetch_candles(limit=None):
    await Database.initialize()
    q = "SELECT time, open, high, low, close FROM candles ORDER BY time ASC"
    if limit:
        q += f" LIMIT {limit}"
    rows = await Database.fetch(q)
    await Database.close()
    df = pd.DataFrame(rows, columns=["time","open","high","low","close"])
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    return df

def simulate(df, bb_window, bb_dev, rsi_long, rsi_short, sl_pct):
    """
    Return dict with 'pnl' and 'sharpe', applying an SL at sl_pct.
    """
    # Bollinger Bands
    mid   = df['close'].rolling(bb_window).mean()
    std   = df['close'].rolling(bb_window).std()
    upper = mid + bb_dev * std
    lower = mid - bb_dev * std

    # RSI
    delta    = df['close'].diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.rolling(rsi_long).mean()
    avg_loss = loss.rolling(rsi_long).mean()
    rs       = avg_gain / avg_loss
    rsi      = 100 - (100 / (1 + rs))

    position    = 0
    entry_price = 0
    stop_price  = 0
    returns     = []

    for t in df.index:
        price = df.at[t, 'close']
        # Entry
        if position == 0:
            if price < lower.at[t] and rsi.at[t] < rsi_long:
                position    = 1
                entry_price = price
                stop_price  = entry_price * (1 - sl_pct)
            elif price > upper.at[t] and rsi.at[t] > rsi_short:
                position    = -1
                entry_price = price
                stop_price  = entry_price * (1 + sl_pct)
        # Exit on mid-band or SL
        elif position == 1:
            if price >= mid.at[t]:
                returns.append((price - entry_price) / entry_price)
                position = 0
            elif price <= stop_price:
                returns.append((stop_price - entry_price) / entry_price)
                position = 0
        elif position == -1:
            if price <= mid.at[t]:
                returns.append((entry_price - price) / entry_price)
                position = 0
            elif price >= stop_price:
                returns.append((entry_price - stop_price) / entry_price)
                position = 0

    if not returns:
        return {'pnl': 0, 'sharpe': 0}

    arr    = np.array(returns)
    pnl    = arr.sum()
    sharpe = (arr.mean() / arr.std() * np.sqrt(len(arr))) if arr.std() > 0 else 0
    return {'pnl': pnl, 'sharpe': sharpe}

async def main():
    df = await fetch_candles(limit=Config.PARAM_RANGES['BB_WINDOW'][-1] * 100)
    pr = Config.PARAM_RANGES

    results = []
    for bb_w, bb_d, rl, rs, sl in itertools.product(
        pr['BB_WINDOW'], pr['BB_DEV'],
        pr['RSI_LONG'],  pr['RSI_SHORT'],
        pr['SL_PCT']
    ):
        m = simulate(df, bb_w, bb_d, rl, rs, sl)
        results.append({
            'window':    bb_w,
            'dev':       bb_d,
            'rsi_long':  rl,
            'rsi_short': rs,
            'sl_pct':    sl,
            'pnl':       m['pnl'],
            'sharpe':    m['sharpe']
        })

    out = pd.DataFrame(results).sort_values(['sharpe','pnl'], ascending=False).head(10)
    print(out)

if __name__ == '__main__':
    asyncio.run(main())
