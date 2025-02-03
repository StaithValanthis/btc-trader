import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU-only
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN warnings
import pandas as pd
from river import compose, linear_model, preprocessing
from alibi_detect.cd import MMDDriftOnline

class OnlineLearner:
    def __init__(self):
        self.x_ref = self._load_reference_data()
        self.drift_detector = MMDDriftOnline(
            x_ref=self.x_ref,
            ert=100,
            window_size=50,
            backend='tensorflow',
            n_permutations=100,
            input_shape=(3,)
        )
        self.model = compose.Pipeline(
            preprocessing.StandardScaler(),
            linear_model.LogisticRegression()
        )

    def _load_reference_data(self):
        """Load and validate historical BTC data"""
        try:
            df = pd.read_csv('data/btc_historical.csv', parse_dates=['timestamp'])
            df['close'] = pd.to_numeric(df['close'], errors='coerce')
            df = df.dropna(subset=['close'])
            
            # Feature engineering
            df['returns'] = df['close'].pct_change().bfill()
            df['volatility'] = df['close'].rolling(5).std().bfill()
            df['momentum'] = (df['close'] - df['close'].shift(5)).bfill()
            
            features = df[['returns', 'volatility', 'momentum']].dropna()
            if len(features) < 500:
                raise ValueError("Need ≥500 samples")
                
            return features.values[-500:]
            
        except Exception as e:
            print(f"Data loading failed: {str(e)}")
            exit(1)

    def update(self, X, y):
        """Update model with new data"""
        try:
            X_dict = {str(i): float(val) for i, val in enumerate(X)}
            self.model.learn_one(X_dict, y)
        except Exception as e:
            print(f"Update error: {str(e)}")

    def detect_drift(self, X):
        """Check for concept drift"""
        try:
            X = X.reshape(-1, 3) if X.ndim == 1 else X[:, :3]
            return self.drift_detector.predict(X)['data']['is_drift']
        except Exception as e:
            print(f"Drift check failed: {str(e)}")
            return False