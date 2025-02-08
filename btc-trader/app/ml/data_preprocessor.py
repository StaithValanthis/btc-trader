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

    def _calculate_rsi(self, series, period=14):
        """Calculate Relative Strength Index (RSI)."""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_macd(self, series, fast=12, slow=26, signal=9):
        """Calculate Moving Average Convergence Divergence (MACD)."""
        exp1 = series.ewm(span=fast, adjust=False).mean()
        exp2 = series.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd, signal_line

    def _calculate_bollinger_bands(self, series, window=20):
        """Calculate Bollinger Bands."""
        sma = series.rolling(window).mean()
        std = series.rolling(window).std()
        upper_band = sma + (std * 2)
        lower_band = sma - (std * 2)
        return upper_band, lower_band

    def _calculate_atr(self, high, low, close, period=14):
        """Calculate Average True Range (ATR)."""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    def _calculate_vwap(self, close, volume):
        """Calculate Volume-Weighted Average Price (VWAP)."""
        return (close * volume).cumsum() / volume.cumsum()

    def create_features(self, df):
        """Generate technical indicators."""
        df = df.sort_index(ascending=True)
        
        df['rsi'] = self._calculate_rsi(df['close'], 14)
        df['macd'], df['signal'] = self._calculate_macd(df['close'])
        df['upper_band'], df['lower_band'] = self._calculate_bollinger_bands(df['close'])
        df['atr'] = self._calculate_atr(df['high'], df['low'], df['close'])
        df['vwap'] = self._calculate_vwap(df['close'], df['volume'])
        df['returns'] = df['close'].pct_change()
        
        return df.dropna()

    def prepare_data(self, df):
        """Prepare data for LSTM model training."""
        try:
            required_samples = self.lookback * 2
            if df.empty or len(df) < required_samples:
                #logger.warning(f"Insufficient data for training: {len(df)} rows available, {required_samples} needed.")
                return np.array([]), np.array([])
            
            df = self.create_features(df)
            df = df[['open', 'high', 'low', 'close', 'volume', 'rsi', 'macd', 'signal', 'upper_band', 'lower_band', 'atr', 'vwap']]
            
            scaled_data = self.scaler.fit_transform(df)
            X, y = [], []
            
            for i in range(self.lookback, len(scaled_data) - self.prediction_window):
                X.append(scaled_data[i-self.lookback:i])
                y.append(scaled_data[i+self.prediction_window, 3])  # Predict future 'close' price
            
            X, y = np.array(X), np.array(y)
            logger.info(f"Data prepared successfully: X shape: {X.shape}, y shape: {y.shape}, available samples: {len(df)}")
            return X, y
        
        except Exception as e:
            logger.error("Data preparation failed", error=str(e))
            return np.array([]), np.array([])
