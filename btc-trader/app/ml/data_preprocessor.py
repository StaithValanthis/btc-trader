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
        """
        Given a DataFrame with base columns (open, high, low, close, volume),
        calculate technical indicators and return a DataFrame with the new features.
        
        Indicators calculated:
          - RSI (Relative Strength Index)
          - MACD and its Signal line
          - Bollinger Bands (upper and lower)
          - ATR (Average True Range)
          - VWAP (Volume Weighted Average Price)
        
        Raises a ValueError if any required base columns are missing.
        """
        try:
            if df.empty:
                logger.warning("Input DataFrame is empty")
                return df

            # Check for required base columns
            if not all(col in df.columns for col in self.base_columns):
                raise ValueError(f"Missing required base columns: {set(self.base_columns) - set(df.columns)}")

            # Ensure data is sorted by time (assumes index is datetime)
            df = df.sort_index(ascending=True)

            # Calculate RSI based on closing prices
            df['rsi'] = self._calculate_rsi(df['close'], period=14)

            # Calculate MACD and signal line
            df['macd'], df['signal'] = self._calculate_macd(df['close'])

            # Calculate Bollinger Bands
            df['upper_band'], df['lower_band'] = self._calculate_bollinger_bands(df['close'], window=20, num_std=2)

            # Calculate ATR (Average True Range)
            df['atr'] = self._calculate_atr(df['high'], df['low'], df['close'], period=14)

            # Calculate VWAP (Volume Weighted Average Price)
            df['vwap'] = self._calculate_vwap(df['close'], df['volume'])

            # Verify that all indicator columns are present; if not, raise an error.
            missing = set(self.required_columns) - set(df.columns)
            if missing:
                raise ValueError(f"Missing calculated columns: {missing}")

            # Return the dataframe with rows that have no missing values
            return df.dropna()

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

    def prepare_data(self, df):
        """
        Prepare the input sequences (X) and target values (y) for LSTM training.
        
        Args:
            df (pd.DataFrame): DataFrame with features created by create_features.
            
        Returns:
            tuple: (X, y) where X is a numpy array of input sequences and y is a numpy array of targets.
        """
        if df.empty:
            logger.warning("Input DataFrame is empty in prepare_data")
            return np.array([]), np.array([])
        
        # Check for required columns
        missing_cols = set(self.required_columns) - set(df.columns)
        if missing_cols:
            logger.error("Missing required columns in prepare_data", missing=missing_cols)
            return np.array([]), np.array([])
        
        # Extract features and fit scaler
        features = df[self.required_columns]
        scaled_features = self.scaler.fit_transform(features)
        
        # Determine the index of the 'close' price in the features
        close_index = self.required_columns.index('close')
        
        X = []
        y = []
        
        # Generate sequences
        for i in range(len(scaled_features) - self.lookback - self.prediction_window + 1):
            # Input sequence (lookback steps)
            X_seq = scaled_features[i:i + self.lookback]
            # Target (prediction_window steps ahead)
            y_value = scaled_features[i + self.lookback + self.prediction_window - 1][close_index]
            X.append(X_seq)
            y.append(y_value)
        
        return np.array(X), np.array(y)
        
    def prepare_data(self, df: pd.DataFrame):
        """
        Prepare training (or prediction) data using a sliding window.
        Assumes that the DataFrame already contains the required features
        in the following order (as defined by self.required_columns):
          - base_columns: ['open', 'high', 'low', 'close', 'volume']
          - indicator_columns: ['rsi', 'macd', 'signal', 'upper_band', 'lower_band', 'atr', 'vwap']
        
        The output X will have shape (n_samples, lookback_window, 12)
        and y is taken as the 'close' value immediately after the window.
        """
        from app.core.config import Config  # ensure you have access to configuration
        lookback = Config.MODEL_CONFIG['lookback_window']  # typically 60

        # Ensure that df contains all required columns
        missing = set(self.required_columns) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Reorder the DataFrame columns to match the required order
        df = df[self.required_columns]
        data = df.values  # shape: (num_rows, 12)

        X, y = [], []
        # Create sliding windows. We use the next 'close' value (column index 3) as the target.
        for i in range(len(data) - lookback):
            X.append(data[i:i + lookback])
            y.append(data[i + lookback][3])
        X = np.array(X)
        y = np.array(y)
        return X, y
