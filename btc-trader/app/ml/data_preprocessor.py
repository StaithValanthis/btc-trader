# app/ml/data_preprocessor.py
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
        self.required_columns = ['price', 'volume']

    def create_features(self, df):
        """Generate required features from raw data"""
        try:
            if df.empty:
                return df
                
            # Ensure required base columns exist
            if not all(col in df.columns for col in ['price', 'volume']):
                raise ValueError("Missing required price/volume data")
                
            # Calculate simple features
            df = df.sort_index(ascending=True)
            df['returns'] = df['price'].pct_change()
            df = df.dropna()
            
            return df
            
        except Exception as e:
            logger.error("Feature creation failed", error=str(e))
            return pd.DataFrame()

    def prepare_data(self, df):
        """Prepare data for LSTM model"""
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
                y.append(scaled_data[i+self.prediction_window, 0])  # Predict price
                
            return np.array(X), np.array(y)
            
        except Exception as e:
            logger.error("Data preparation failed", error=str(e))
            return np.array([]), np.array([])