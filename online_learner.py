from alibi_detect.cd import MMDDriftOnline
from ml_model import MLModel
import numpy as np

class OnlineLearner:
    def __init__(self, window_size=50, n_features=3):
        self.window_size = window_size
        self.n_features = n_features
        self.model = MLModel()
        
        # Initialize with dummy reference data (replace with real data later)
        self.x_ref = np.random.randn(100, self.n_features)  # Example shape (100, 3)
        
        self.drift_detector = MMDDriftOnline(
            x_ref=self.x_ref,  # Provide reference dataset
            ert=25,
            window_size=self.window_size,
            input_shape=(self.n_features,)
        )

    def update(self, X: np.ndarray, y: np.ndarray):
        drift_pred = self.drift_detector.predict(X.reshape(1, -1))
        if drift_pred['data']['is_drift']:
            print("Drift detected! Retraining model...")
            self.model.train(X.reshape(1, -1), y)
        return drift_pred