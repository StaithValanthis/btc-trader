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

    def create_features(self, df):
        # Calculate technical indicators
        df['rsi'] = self.calculate_rsi(df['price'], 14)
        df['macd'], df['signal'] = self.calculate_macd(df['price'])
        return df[['price', 'volume', 'rsi', 'macd']]

    def calculate_rsi(self, series, period):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def calculate_macd(self, series, fast=12, slow=26, signal=9):
        exp1 = series.ewm(span=fast, adjust=False).mean()
        exp2 = series.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd, signal_line

    def prepare_data(self, df):
        try:
            df = self.create_features(df)
            scaled_data = self.scaler.fit_transform(df)
            
            X, y = [], []
            for i in range(self.lookback, len(scaled_data)-self.prediction_window):
                X.append(scaled_data[i-self.lookback:i])
                y.append(scaled_data[i:i+self.prediction_window, 0])
                
            return np.array(X), np.array(y)
        except Exception as e:
            logger.error("Data preparation failed", error=str(e))
            raise