from sklearn.linear_model import SGDRegressor
import numpy as np
import joblib

class MLModel:
    def __init__(self):
        self.model = SGDRegressor(warm_start=True, loss='huber')
        self.is_initialized = False

    def train(self, X: np.ndarray, y: np.ndarray):
        """
        Train the model incrementally.
        """
        if not self.is_initialized:
            self.model.partial_fit(X, y)
            self.is_initialized = True
        else:
            self.model.partial_fit(X, y)

    def predict(self, X: np.ndarray) -> float:
        """
        Predict using the model.
        """
        return self.model.predict(X.reshape(1, -1))[0]

    def save(self, path: str = 'ml_model.pkl'):
        """
        Save the model to disk.
        """
        joblib.dump(self.model, path)

    def load(self, path: str = 'ml_model.pkl'):
        """
        Load the model from disk.
        """
        self.model = joblib.load(path)
        self.is_initialized = True