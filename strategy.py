import numpy as np

class TradingStrategy:
    def __init__(self, model):
        self.model = model  # Reference to MLModel
        self.threshold = 0.02  # Confidence threshold

    def generate_signal(self, X: np.ndarray) -> str:
        """
        Generate a trading signal based on ML predictions.
        """
        prediction = self.model.predict(X)
        if prediction > self.threshold:
            return "BUY"
        elif prediction < -self.threshold:
            return "SELL"
        else:
            return "HOLD"

    def update_threshold(self, volatility: float):
        """
        Adjust the confidence threshold based on market volatility.
        """
        self.threshold = 0.02 * (1 + volatility)  # Scale with volatility