from alibi_detect.cd import MMDDriftOnline
from ml_model import MLModel
import numpy as np

class OnlineLearner:
    def __init__(self, window_size=50, n_features=3):
        self.window_size = window_size
        self.n_features = n_features
        self.model = MLModel()
        self.drift_detector = MMDDriftOnline(
            ert=25,
            window_size=self.window_size,
            input_shape=(self.n_features,)
        )

    def update(self, X: np.ndarray, y: np.ndarray):
        """
        Update the model and check for drift.
        """
        # Detect drift
        drift_pred = self.drift_detector.predict(X.reshape(1, -1))
        if drift_pred['data']['is_drift']:
            print("Drift detected! Retraining model...")
            self.model.train(X.reshape(1, -1), y)
        return drift_pred