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

        # We'll have separate columns for 1m and 5m data + new indicators
        # For 1-minute bars:
        self.base_1m = ['open_1m', 'high_1m', 'low_1m', 'close_1m', 'volume_1m']
        # For 5-minute bars:
        self.base_5m = ['open_5m', 'high_5m', 'low_5m', 'close_5m', 'volume_5m']

        # Indicators (some apply to 1m, some to 5m, or both). We'll label them accordingly.
        self.indicators_1m = [
            'rsi_1m', 'macd_1m', 'signal_1m', 'upper_band_1m', 'lower_band_1m',
            'ema9_1m', 'ema21_1m', 'ema55_1m', 'stoch_k_1m', 'stoch_d_1m', 'obv_1m'
        ]
        self.indicators_5m = [
            'ema9_5m', 'ema21_5m', 'ema55_5m', 'rsi_5m', 'macd_5m', 'signal_5m'
            # ... add more if desired
        ]

        # Combine into one comprehensive list:
        self.required_columns = (
            self.base_1m + 
            self.base_5m + 
            self.indicators_1m + 
            self.indicators_5m
        )

    def merge_timeframes(self, df1m, df5m):
        """
        Merge 1m & 5m data into a single DataFrame on the time index.
        We suffix columns with _1m or _5m so they don't collide.
        """
        # df1m and df5m are both indexed by time (after groupby in your strategy).
        # We'll do an outer join to align times. 5m updates less frequently,
        # so forward-fill or fill with NaN as needed.
        df = df1m.join(df5m, how='outer')
        # Forward fill 5m columns so that each 1-minute row in between gets the same 5-min bar
        df.ffill(inplace=True)
        return df

    def create_features_for_1m(self, df):
        """Calculate indicators for the 1-minute data."""
        df['rsi_1m'] = self._calculate_rsi(df['close_1m'], 14)
        df['macd_1m'], df['signal_1m'] = self._calculate_macd(df['close_1m'])
        df['upper_band_1m'], df['lower_band_1m'] = self._calculate_bollinger_bands(df['close_1m'])
        df['ema9_1m'] = df['close_1m'].ewm(span=9, adjust=False).mean()
        df['ema21_1m'] = df['close_1m'].ewm(span=21, adjust=False).mean()
        df['ema55_1m'] = df['close_1m'].ewm(span=55, adjust=False).mean()
        df['stoch_k_1m'], df['stoch_d_1m'] = self._calculate_stoch(df['high_1m'], df['low_1m'], df['close_1m'])
        df['obv_1m'] = self._calculate_obv(df['close_1m'], df['volume_1m'])
        df.dropna(inplace=True)

    def create_features_for_5m(self, df):
        """Calculate indicators for the 5-minute data."""
        df['ema9_5m'] = df['close_5m'].ewm(span=9, adjust=False).mean()
        df['ema21_5m'] = df['close_5m'].ewm(span=21, adjust=False).mean()
        df['ema55_5m'] = df['close_5m'].ewm(span=55, adjust=False).mean()
        df['rsi_5m'] = self._calculate_rsi(df['close_5m'], 14)
        df['macd_5m'], df['signal_5m'] = self._calculate_macd(df['close_5m'])
        df.dropna(inplace=True)

    def prepare_data(self, df):
        """
        Scale all required columns and create the final (X, y) for LSTM.
        We'll assume we predict future close_1m as the target or something similar.
        """
        try:
            if df.empty:
                return np.array([]), np.array([])

            # Check if required columns exist
            for col in self.required_columns:
                if col not in df.columns:
                    raise ValueError(f"Missing column {col} in merged DF")

            # Sort by time ascending
            df.sort_index(ascending=True, inplace=True)

            # Scale only the required columns
            scaled_data = self.scaler.fit_transform(df[self.required_columns])

            X, y = [], []
            # We'll define our target as `close_1m` in the future:
            close_1m_idx = self.required_columns.index('close_1m')

            for i in range(self.lookback, len(scaled_data) - self.prediction_window):
                X.append(scaled_data[i - self.lookback : i])
                # Predict the close_1m <prediction_window> steps in the future
                y.append(scaled_data[i + self.prediction_window, close_1m_idx])

            X, y = np.array(X), np.array(y)

            logger.info("Data prepared successfully", samples=X.shape[0], features=X.shape[2])
            return X, y
        
        except Exception as e:
            logger.error("Data preparation failed", error=str(e))
            return np.array([]), np.array([])

    # -------------------------------------------------
    # Helper indicator methods
    # -------------------------------------------------
    def _calculate_rsi(self, series, period=14):
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_macd(self, series, fastperiod=12, slowperiod=26, signalperiod=9):
        ema_fast = series.ewm(span=fastperiod, adjust=False).mean()
        ema_slow = series.ewm(span=slowperiod, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=signalperiod, adjust=False).mean()
        return macd, signal

    def _calculate_bollinger_bands(self, series, window=20, num_std=2):
        sma = series.rolling(window=window).mean()
        std = series.rolling(window=window).std()
        upper_band = sma + (num_std * std)
        lower_band = sma - (num_std * std)
        return upper_band, lower_band

    def _calculate_stoch(self, high, low, close, k_period=14, d_period=3):
        """Stochastic Oscillator %K and %D."""
        lowest_low = low.rolling(k_period).min()
        highest_high = high.rolling(k_period).max()
        stoch_k = 100 * ((close - lowest_low) / (highest_high - lowest_low + 1e-9))
        stoch_d = stoch_k.rolling(d_period).mean()
        return stoch_k, stoch_d

    def _calculate_obv(self, close, volume):
        """On-Balance Volume."""
        # If today's close > yesterday's close => +volume else -volume
        signed_vol = np.where(close > close.shift(1), volume, 
                        np.where(close < close.shift(1), -volume, 0))
        obv = pd.Series(signed_vol).cumsum()
        return obv
