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
        self.required_columns = [
            'open', 'high', 'low', 'close', 'volume',
            'rsi', 'macd', 'signal', 'upper_band', 'lower_band', 'atr', 'vwap'
        ]

    def create_features(self, df):
        """Generate technical indicators with validation"""
        try:
            if df.empty:
                return df
                
            # Ensure required base columns exist
            if not all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume']):
                raise ValueError("Missing required price columns")
                
            # Calculate indicators
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
            
            # Returns
            df['returns'] = df['close'].pct_change()
            
            # Ensure all required columns are present
            missing = set(self.required_columns) - set(df.columns)
            if missing:
                raise ValueError(f"Missing calculated columns: {missing}")
                
            return df.dropna()
            
        except Exception as e:
            logger.error("Feature creation failed", error=str(e))
            return pd.DataFrame()

    def prepare_data(self, df):
        """Prepare data for LSTM model with validation"""
        try:
            if df.empty:
                return np.array([]), np.array([])
                
            # Ensure required columns
            if not all(col in df.columns for col in self.required_columns):
                raise ValueError("Missing required features")
                
            # Scale data
            scaled_data = self.scaler.fit_transform(df[self.required_columns])
            
            # Create sequences
            X, y = [], []
            for i in range(self.lookback, len(scaled_data) - self.prediction_window):
                X.append(scaled_data[i-self.lookback:i])
                y.append(scaled_data[i+self.prediction_window, 3])  # Predict 'close' price
                
            X, y = np.array(X), np.array(y)
            
            logger.info("Data prepared successfully", 
                       samples=X.shape[0], 
                       features=X.shape[2])
            return X, y
            
        except Exception as e:
            logger.error("Data preparation failed", error=str(e))
            return np.array([]), np.array([])
