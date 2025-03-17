# File: v2-Inverse-btc-trader/app/ml/feature_engineering.py

"""
Shared feature engineering functions to avoid duplication across training/inference modules.
"""

import pandas as pd
import ta

def compute_15min_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute 15-min aggregated features for a given 1-min DataFrame.
    This includes RSI, MACD, Bollinger, ATR, etc.

    Args:
        df (pd.DataFrame): 1-min raw data with columns [open, high, low, close, volume]

    Returns:
        pd.DataFrame: 15-min DataFrame with engineered features
    """
    df_15 = df.resample("15min").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum"
    }).ffill().dropna()

    df_15["returns"] = df_15["close"].pct_change()
    df_15["rsi"] = ta.momentum.rsi(df_15["close"], window=14)
    macd = ta.trend.MACD(df_15["close"], window_slow=26, window_fast=12, window_sign=9)
    df_15["macd"] = macd.macd()
    df_15["macd_diff"] = macd.macd_diff()

    boll = ta.volatility.BollingerBands(df_15["close"], window=20, window_dev=2)
    df_15["bb_width"] = boll.bollinger_hband() - boll.bollinger_lband()

    atr_indicator = ta.volatility.AverageTrueRange(
        high=df_15["high"], low=df_15["low"], close=df_15["close"], window=14
    )
    df_15["atr"] = atr_indicator.average_true_range()

    df_15["volatility_ratio"] = (df_15["high"] - df_15["low"]) / df_15["close"]

    # future_return for label generation if needed
    df_15["future_return"] = (df_15["close"].shift(-5) / df_15["close"]) - 1
    df_15["future_return_smooth"] = df_15["future_return"].rolling(window=3, min_periods=1).mean()

    df_15.dropna(inplace=True)
    return df_15
