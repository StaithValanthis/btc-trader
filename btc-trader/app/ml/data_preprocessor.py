# File: app/ml/data_preprocessor.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from structlog import get_logger

logger = get_logger(__name__)

class DataPreprocessor:
    def __init__(self, lookback=60, prediction_window=30):
        self.lookback = lookback
        self.prediction_window = prediction_window
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
        # Define expected base column names for 1-minute data
        self.base_columns = ['open', 'high', 'low', 'close', 'volume']
        # Define names for the indicators that will be calculated
        self.indicator_columns = [
            'rsi', 'macd', 'signal', 'upper_band', 'lower_band', 'atr', 'vwap'
        ]
        # The full list of columns that should exist after feature creation
        self.required_columns = self.base_columns + self.indicator_columns

    def merge_timeframes(self, df1m: pd.DataFrame, df5m: pd.DataFrame) -> pd.DataFrame:
        """
        Merge 1m & 5m data on their time index (both named 'bucket').
        A left join on the 1m data is performed, and the 5m data is forward-filled.
        """
        df = df1m.join(df5m, how='left')
        df.ffill(inplace=True)
        return df

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features with robust error handling."""
        try:
            if df.empty:
                logger.warning("Input DataFrame is empty")
                return df

            # Ensure required base columns exist
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                raise ValueError(f"Missing required columns: {set(required_cols) - set(df.columns)}")

            # Calculate indicators with error handling
            try:
                df['rsi'] = self._calculate_rsi(df['close'], period=14)
                df['macd'], df['signal'] = self._calculate_macd(df['close'])
                df['upper_band'], df['lower_band'] = self._calculate_bollinger_bands(df['close'])
                df['atr'] = self._calculate_atr(df['high'], df['low'], df['close'])
                df['vwap'] = self._calculate_vwap(df['close'], df['volume'])
            except Exception as e:
                logger.error("Indicator calculation failed", error=str(e))
                return pd.DataFrame()

            # Forward fill missing values instead of dropping
            df.ffill(inplace=True)
            

            # Verify all required columns are present
            missing = set(self.required_columns) - set(df.columns)
            if missing:
                logger.error("Missing calculated columns", columns=missing)
                return pd.DataFrame()                

            return df            

        except Exception as e:
            logger.error("Feature creation failed", error=str(e))
            return pd.DataFrame()

    # -------------------------------------------------
    # Helper methods for indicator calculations
    # -------------------------------------------------
    def _calculate_rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate the Relative Strength Index (RSI) for a given series.
        """
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(period, min_periods=period).mean()
        avg_loss = loss.rolling(period, min_periods=period).mean()
        rs = avg_gain / (avg_loss + 1e-9)  # avoid division by zero
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_macd(self, series: pd.Series, fastperiod: int = 12, slowperiod: int = 26, signalperiod: int = 9):
        """
        Calculate the MACD and its signal line.
        """
        ema_fast = series.ewm(span=fastperiod, adjust=False).mean()
        ema_slow = series.ewm(span=slowperiod, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=signalperiod, adjust=False).mean()
        return macd, signal

    def _calculate_bollinger_bands(self, series: pd.Series, window: int = 20, num_std: float = 2):
        """
        Calculate Bollinger Bands for the given series.
        """
        sma = series.rolling(window=window).mean()
        std = series.rolling(window=window).std()
        upper = sma + (num_std * std)
        lower = sma - (num_std * std)
        return upper, lower

    def _calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate the Average True Range (ATR) for given high, low, and close series.
        """
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period, min_periods=period).mean()
        return atr

    def _calculate_vwap(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Calculate the Volume Weighted Average Price (VWAP) for given close and volume series.
        This implementation computes the cumulative VWAP.
        """
        cum_vol = volume.cumsum()
        cum_vol_price = (close * volume).cumsum()
        vwap = cum_vol_price / cum_vol
        return vwap

    def prepare_training_data(self, df):
        """
        Prepare training data with the correct lookback window.
        """
        if df.empty:
            logger.warning("Input DataFrame is empty")
            return np.array([]), np.array([])

        # Ensure required columns exist
        missing_cols = set(self.required_columns) - set(df.columns)
        if missing_cols:
            logger.error("Missing required columns", missing=missing_cols)
            return np.array([]), np.array([])

        # Extract features and scale
        features = df[self.required_columns]
        scaled_features = self.scaler.fit_transform(features)

        # Generate sequences with the correct lookback window
        X, y = [], []
        for i in range(len(scaled_features) - self.lookback - self.prediction_window + 1):
            X_seq = scaled_features[i:i + self.lookback]
            y_value = scaled_features[i + self.lookback + self.prediction_window - 1][self.required_columns.index('close')]
            X.append(X_seq)
            y.append(y_value)

        return np.array(X), np.array(y)
        
    def prepare_prediction_data(self, df: pd.DataFrame):    
        """
        Prepare prediction data with the correct lookback window.
        """
        if df.empty:
            logger.warning("Input DataFrame is empty")
            return np.array([]), np.array([])

        # Ensure required columns exist
        missing_cols = set(self.required_columns) - set(df.columns)
        if missing_cols:
            logger.error("Missing required columns", missing=missing_cols)
            return np.array([]), np.array([])

        # Extract features and scale
        features = df[self.required_columns]
        scaled_features = self.scaler.transform(features)

        # Use the last `lookback` samples for prediction
        X = scaled_features[-self.lookback:]
        return np.array([X]), np.array([])  # Return X as a batch of 1 sample
