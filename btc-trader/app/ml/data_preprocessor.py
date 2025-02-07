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

    def _calculate_volatility(self, series, window=20):
        """Calculate volatility (standard deviation of returns)."""
        returns = series.pct_change()
        return returns.rolling(window).std()

    def create_features(self, df):
        """Generate technical indicators."""
        df = df.sort_index(ascending=True)
        
        # Calculate RSI
        df['rsi'] = self._calculate_rsi(df['close'], 14)
        
        # Calculate MACD and Signal Line
        df['macd'], df['signal'] = self._calculate_macd(df['close'])
        
        # Calculate Volatility
        df['volatility'] = self._calculate_volatility(df['close'], 20)
        
        # Calculate Returns
        df['returns'] = df['close'].pct_change()
        
        # Keep relevant columns
        return df[['open', 'high', 'low', 'close', 'volume', 'rsi', 'macd', 'signal', 'volatility', 'returns']]

    def prepare_data(self, df):
        """Prepare data for LSTM model training."""
        try:
            if df.empty or len(df) < self.lookback * 2:
                return np.array([]), np.array([])
                
            # Generate features
            df = self.create_features(df).dropna()
            
            if len(df) < self.lookback:
                return np.array([]), np.array([])
                
            # Scale data
            scaled_data = self.scaler.fit_transform(df)
            X, y = [], []
            
            # Create sequences for LSTM
            for i in range(self.lookback, len(scaled_data)-self.prediction_window):
                X.append(scaled_data[i-self.lookback:i])
                y.append(scaled_data[i:i+self.prediction_window, 0])  # Predict 'close' price
                
            return np.array(X), np.array(y)
            
        except Exception as e:
            logger.error("Data preparation failed", error=str(e))
            return np.array([]), np.array([])