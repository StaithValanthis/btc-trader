import asyncio, pandas as pd, ta
from structlog import get_logger
from pybit.unified_trading import HTTP
from app.core.config import Config

logger = get_logger(__name__)

def compute_indicators(df, window, dev, rsi_long):
    bb = ta.volatility.BollingerBands(df.close, window, window_dev=dev)
    df = df.assign(
        mid=bb.bollinger_mavg(),
        upper=bb.bollinger_hband(),
        lower=bb.bollinger_lband()
    )
    rsi = ta.momentum.RSIIndicator(df.close, window=rsi_long)
    df["rsi"] = rsi.rsi()
    return df

async def position_signal(df, cfg):
    cur = df.iloc[-1]
    if cur.close < cur.lower and cur.rsi < cfg["RSI_LONG"]:
        return "long"
    if cur.close > cur.upper and cur.rsi > cfg["RSI_SHORT"]:
        return "short"
    return None

async def place_order(session, side, symbol, qty):
    return await asyncio.to_thread(
        session.place_active_order,
        category=Config["BYBIT"]["category"],
        symbol=symbol,
        side="Buy" if side=="long" else "Sell",
        orderType="Market",
        qty=str(qty)
    )

async def on_candle_msg(state, msg):
    """
    state holds: session, symbol, cfg, tick_size, lot_size,
                  current_pos, entry_price, candle_buffer
    """
    df = pd.DataFrame([{
        "time":   pd.to_datetime(msg["startTime"],unit="ms"),
        "open":   float(msg["open"]),
        "high":   float(msg["high"]),
        "low":    float(msg["low"]),
        "close":  float(msg["close"]),
        "volume": float(msg["volume"])
    }])
    state["buf"]=pd.concat([state["buf"],df]).iloc[-state["cfg"]["BB_WINDOW"]*3:]
    if len(state["buf"]) < state["cfg"]["BB_WINDOW"]:
        return

    state["buf"] = compute_indicators(
        state["buf"], state["cfg"]["BB_WINDOW"], state["cfg"]["BB_DEV"], state["cfg"]["RSI_LONG"]
    )
    sig = await position_signal(state["buf"], state["cfg"])
    price = state["buf"].close.iloc[-1]

    # exit logic
    if state["current_pos"]=="long":
        stop = state["entry_price"]*(1-state["cfg"]["SL_PCT"])
        if price>=state["buf"].mid.iloc[-1] or price<=stop:
            await place_order(state["session"],"short",state["symbol"],None)
            state.update(current_pos=None,entry_price=None)
    elif state["current_pos"]=="short":
        stop = state["entry_price"]*(1+state["cfg"]["SL_PCT"])
        if price<=state["buf"].mid.iloc[-1] or price>=stop:
            await place_order(state["session"],"long",state["symbol"],None)
            state.update(current_pos=None,entry_price=None)
    elif sig:
        # size calc
        bal = await asyncio.to_thread(
            state["session"].get_wallet_balance, accountType="UNIFIED"
        )
        usdt = next((float(c["usdValue"]) for c in bal["result"]["list"][0]["coin"] if c["coin"]=="USDT"),0)
        risk = usdt*state["cfg"]["risk_pct"]
        stop_px = price*(1-state["cfg"]["SL_PCT"]) if sig=="long" else price*(1+state["cfg"]["SL_PCT"])
        dist = abs(price-stop_px)
        size = int((risk*state["cfg"]["leverage"]/(dist/price))//state["lot_size"])*state["lot_size"]
        if size>=state["lot_size"]:
            resp = await place_order(state["session"],sig,state["symbol"],size)
            if resp.get("retCode")==0:
                state.update(current_pos=sig,entry_price=price)
                logger.info("Entered",symbol=state["symbol"],side=sig,price=price,qty=size)
            else:
                logger.error("Entry failed",response=resp)
