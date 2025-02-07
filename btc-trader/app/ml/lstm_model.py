import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from structlog import get_logger
from app.core.config import Config

logger = get_logger(__name__)

class LSTMModel:
    def __init__(self, input_shape):
        self.model = self.build_model(input_shape)
        self.model.compile(optimizer='adam', loss='mse')

    def build_model(self, input_shape):
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=input_shape),
            Dropout(0.3),
            LSTM(64, return_sequences=False),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dense(5)  # Predict next 5 time steps
        ])
        return model

    def train(self, X_train, y_train):
        try:
            self.model.fit(
                X_train, y_train,
                epochs=Config.MODEL_CONFIG['train_epochs'],
                batch_size=Config.MODEL_CONFIG['batch_size'],
                verbose=0
            )
            logger.info("Model training completed successfully")
            return True
        except Exception as e:
            logger.error("Training failed", error=str(e))
            return False

    def predict(self, X):
        try:
            predictions = self.model.predict(X, verbose=0)
            logger.info("Predictions generated successfully")
            return predictions
        except Exception as e:
            logger.error("Prediction failed", error=str(e))
            return None

    def save(self, path):
        try:
            self.model.save(path)
            logger.info(f"Model saved successfully at {path}")
        except Exception as e:
            logger.error("Failed to save model", error=str(e))

    def load(self, path):
        try:
            self.model = tf.keras.models.load_model(path)
            logger.info(f"Model loaded successfully from {path}")
        except Exception as e:
            logger.error("Failed to load model", error=str(e))