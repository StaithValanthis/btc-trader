import pandas as pd
from ml_model import MLModel

class MLStrategy:
    def __init__(self):
        self.model = MLModel()
        self.trained = False

    def create_features(self, data):
        """Create features with validation."""
        try:
            df = data.copy()
            df['returns'] = df['close'].pct_change()
            df['volatility'] = df['close'].rolling(5).std().bfill()
            df['momentum'] = df['close'] - df['close'].shift(5).bfill()
            return df[['returns', 'volatility', 'momentum']].dropna()
        except Exception as e:
            print(f"Feature error: {str(e)}")
            return pd.DataFrame()

    def calculate_signals(self, data):
        """Generate signals with validation."""
        try:
            if not self.trained:
                self.train_model(data)
            
            features = self.create_features(data)
            if features.empty:
                return 'Hold'
            
            pred = self.model.predict(features.iloc[-1:].values)
            return 'Buy' if pred > 0.6 else 'Sell' if pred < 0.4 else 'Hold'
        except Exception as e:
            print(f"Signal error: {str(e)}")
            return 'Hold'

    def train_model(self, data):
        """Robust model training."""
        try:
            features = self.create_features(data)
            labels = (data['close'].shift(-4) > data['close']).astype(int).dropna()
            
            # Align data
            min_len = min(len(features), len(labels))
            features = features.iloc[:min_len]
            labels = labels.iloc[:min_len]
            
            if min_len < 10:
                raise ValueError("Insufficient training data")
            
            self.model.train(features.values, labels.values)
            self.trained = True
        except Exception as e:
            print(f"Training error: {str(e)}")