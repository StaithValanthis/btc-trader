# File: app/ml/data_preprocessor.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from structlog import get_logger

logger = get_logger(__name__)

class DataPreprocessor:
    def __init__(self, lookback=60, prediction_window=5):
        self.lookback = lookback
        self.prediction_window = prediction_window
        self.scaler = MinMaxScaler(feature_range=(0, 1))

        # For 1m base: open_1m, close_1m, etc.
        self.base_1m = ['open_1m', 'high_1m', 'low_1m', 'close_1m', 'volume_1m']
        # For 5m base
        self.base_5m = ['open_5m', 'high_5m', 'low_5m', 'close_5m', 'volume_5m']

        # Indicators for 1m
        self.indicators_1m = [
            'rsi_1m', 'macd_1m', 'signal_1m', 'upper_band_1m', 'lower_band_1m',
            'ema9_1m', 'ema21_1m', 'ema55_1m'
        ]
        # Indicators for 5m
        self.indicators_5m = [
            'ema9_5m', 'ema21_5m', 'ema55_5m', 'rsi_5m', 'macd_5m', 'signal_5m'
        ]

        # Full list of required columns
        self.required_columns = self.base_1m + self.base_5m + self.indicators_1m + self.indicators_5m

    def merge_timeframes(self, df1m, df5m):
        """
        Merge 1m & 5m data on their time index (both named 'bucket').
        We'll do a left join on the 1m index, then forward-fill 5m.
        """
        df = df1m.join(df5m, how='left')  # left join to keep all 1m rows
        # Forward fill the 5m columns so intermediate 1m rows have the same 5m data
        df.ffill(inplace=True)
        return df

    def create_features_for_1m(self, df):
        """
        Calculate indicators for the 1-minute columns only.
        Expect columns: open_1m, close_1m, etc.
        """
        if df.empty:
            return
        # Basic checks
        needed_1m = ['open_1m', 'high_1m', 'low_1m', 'close_1m', 'volume_1m']
        if not all(c in df.columns for c in needed_1m):
            logger.warning("Missing 1m columns for indicator creation.")
            return
        
        # RSI for 1m close
        df['rsi_1m'] = self._calculate_rsi(df['close_1m'], 14)
        # MACD for 1m
        macd, signal = self._calculate_macd(df['close_1m'])
        df['macd_1m'] = macd
        df['signal_1m'] = signal
        # Bollinger (or simpler) - just example
        df['upper_band_1m'], df['lower_band_1m'] = self._calculate_bollinger_bands(df['close_1m'])
        # EMAs
        df['ema9_1m'] = df['close_1m'].ewm(span=9, adjust=False).mean()
        df['ema21_1m'] = df['close_1m'].ewm(span=21, adjust=False).mean()
        df['ema55_1m'] = df['close_1m'].ewm(span=55, adjust=False).mean()

        # Drop partial NaNs from these calculations
        df.dropna(subset=self.indicators_1m, inplace=True)

    # In DataPreprocessor
    def create_features_for_1m(self, df):
        """
        Create features specifically for 1-minute data.
        """
        try:
            if df.empty:
                logger.warning("Input DataFrame is empty")
                return df

            # Ensure required base columns exist
            if not all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume']):
                raise ValueError("Missing required price columns")

            # Calculate indicators for 1-minute data
            df = df.sort_index(ascending=True)

            # RSI
            df['rsi'] = self._calculate_rsi(df['close'], 14)

            # MACD
            df['macd'], df['signal'] = self._calculate_macd(df['close'])

            # Bollinger Bands
            df['upper_band'], df['lower_band'] = self._calculate_bollinger_bands(df['close'])

            # ATR
            df['atr'] = self._calculate_atr(df['high'], df['low'], df['close'])

            # VWAP
            df['vwap'] = self._calculate_vwap(df['close'], df['volume'])

            # Ensure all required columns are present
            missing = set(self.required_columns) - set(df.columns)
            if missing:
                raise ValueError(f"Missing calculated columns: {missing}")

            return df.dropna()

        except Exception as e:
            logger.error("Feature creation for 1-minute data failed", error=str(e))
            return pd.DataFrame()

    # -------------------------------------------------
    # Helper methods for RSI, MACD, Bollinger, etc.
    # -------------------------------------------------
    def _calculate_rsi(self, series, period=14):
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()

        rs = avg_gain / (avg_loss + 1e-9)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_macd(self, series, fastperiod=12, slowperiod=26, signalperiod=9):
        ema_fast = series.ewm(span=fastperiod, adjust=False).mean()
        ema_slow = series.ewm(span=slowperiod, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=signalperiod, adjust=False).mean()
        return macd, signal

    def _calculate_bollinger_bands(self, series, window=20, num_std=2):
        sma = series.rolling(window).mean()
        std = series.rolling(window).std()
        upper = sma + (num_std * std)
        lower = sma - (num_std * std)
        return upper, lower
