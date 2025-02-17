import numpy as np
import pandas as pd
from config import Config

def compute_rsi(prices, period=Config.RSI_PERIOD):
    if len(prices) < period + 1:
        return None
    series = pd.Series(list(prices))
    delta = series.diff().dropna()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean().iloc[-1]
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean().iloc[-1]
    if avg_loss == 0:
        return 100
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def compute_atr(prices, period=Config.ATR_PERIOD):
    if len(prices) < period + 1:
        return None
    arr = np.array(list(prices)[- (period+1):])
    return np.mean(np.abs(np.diff(arr)))

def compute_bb(prices, window=Config.BB_WINDOW, num_std=Config.BB_STD_MULTIPLIER):
    if len(prices) < window:
        return None, None, None
    series = pd.Series(list(prices)[-window:])
    ma = series.mean()
    std = series.std()
    return ma, ma + num_std * std, ma - num_std * std

class EMAUpdater:
    def __init__(self, period=Config.EMA_PERIOD):
        self.period = period
        self.alpha = 2 / (period + 1)
        self.current_ema = None

    def update(self, price):
        if self.current_ema is None:
            self.current_ema = price
        else:
            self.current_ema = (price - self.current_ema) * self.alpha + self.current_ema
        return self.current_ema
