import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from structlog import get_logger
from keras_tuner import RandomSearch

logger = get_logger(__name__)

class LSTMModel:
    def __init__(self, input_shape):
        self.model = self.build_model(input_shape)
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

    def build_model(self, input_shape):
        model = Sequential([
            Bidirectional(LSTM(128, return_sequences=True, input_shape=input_shape)),
            Dropout(0.3),
            LSTM(64, return_sequences=False),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dense(1)  # Predict next closing price
        ])
        return model

    def train(self, X_train, y_train, epochs=50, batch_size=32):
        try:
            logger.info(f"Training model with input shape: {X_train.shape}")
            self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                verbose=1
            )
            logger.info("Model training completed successfully")
        except Exception as e:
            logger.error("Training failed", error=str(e))

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

    def hyperparameter_tuning(self, X_train, y_train):
        """Use KerasTuner to find optimal hyperparameters for LSTM"""
        def build_lstm(hp):
            model = Sequential()
            model.add(Bidirectional(LSTM(
                units=hp.Int('units', min_value=32, max_value=256, step=32),
                return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])
            )))
            model.add(Dropout(hp.Float('dropout', 0.1, 0.5, step=0.1)))
            model.add(LSTM(64, return_sequences=False))
            model.add(Dense(32, activation='relu'))
            model.add(Dense(1))  # Predict next closing price
            model.compile(optimizer=Adam(learning_rate=hp.Choice('learning_rate', [0.001, 0.0005, 0.0001])), loss='mse')
            return model

        tuner = RandomSearch(
            build_lstm,
            objective='val_loss',
            max_trials=10,
            executions_per_trial=1,
            directory='lstm_tuning',
            project_name='lstm_opt'
        )
        
        tuner.search(X_train, y_train, epochs=10, validation_split=0.2, verbose=1)
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        logger.info(f"Best Hyperparameters Found: {best_hps.values}")
