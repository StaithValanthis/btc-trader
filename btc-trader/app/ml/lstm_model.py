import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten, Bidirectional
from tensorflow.keras.optimizers import Adam
from structlog import get_logger
# Optional tuner if you want:
try:
    import keras_tuner as kt
except ImportError:
    kt = None

logger = get_logger(__name__)

class LSTMModel:
    def __init__(self, input_shape):
        self.input_shape = input_shape  # (lookback_window, feature_count)
        self.model = self.build_model()
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

    def build_model(self):
        """
        You can experiment:
        - Basic LSTM
        - Bi-LSTM
        - Optional small Conv1D before LSTM, etc.
        """
        model = Sequential()
        # Example: small 1D conv for local pattern extraction:
        model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=self.input_shape))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Bidirectional(LSTM(64, return_sequences=True)))
        model.add(Dropout(0.2))
        model.add(LSTM(32, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(1))  # Predict next closing price
        return model

    def train(self, X_train, y_train, epochs=50, batch_size=32):
        """Train with EarlyStopping to reduce overfitting."""
        try:
            from tensorflow.keras.callbacks import EarlyStopping
            early_stopping = EarlyStopping(
                monitor='loss',
                patience=5,
                restore_best_weights=True
            )
            logger.info(f"Training model with input shape: {X_train.shape}")
            self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                verbose=1,
                callbacks=[early_stopping]
            )
            logger.info("Model training completed successfully")
        except Exception as e:
            logger.error("Training failed", error=str(e))

    def predict(self, X):
        try:
            predictions = self.model.predict(X, verbose=0)
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

    # ------------------------------------------------------------------
    # Optional KerasTuner approach
    # ------------------------------------------------------------------
    def hyperparameter_tuning(self, X_train, y_train):
        """Use KerasTuner to find optimal hyperparameters for LSTM (optional)."""
        if not kt:
            logger.warning("keras-tuner not installed. Skipping.")
            return

        def build_lstm(hp):
            model = Sequential()
            model.add(Conv1D(
                filters=hp.Int('conv_filters', 16, 64, step=16),
                kernel_size=3,
                activation='relu',
                input_shape=self.input_shape
            ))
            model.add(MaxPooling1D(pool_size=2))
            model.add(Bidirectional(LSTM(hp.Int('units', 32, 128, step=32), return_sequences=True)))
            model.add(Dropout(hp.Float('dropout', 0.1, 0.5, step=0.1)))
            model.add(LSTM(hp.Int('units2', 32, 128, step=32), return_sequences=False))
            model.add(Dense(16, activation='relu'))
            model.add(Dense(1))
            lr = hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])
            model.compile(optimizer=Adam(lr), loss='mse')
            return model

        tuner = kt.RandomSearch(
            build_lstm,
            objective='val_loss',
            max_trials=5,
            executions_per_trial=1,
            directory='lstm_tuning',
            project_name='lstm_opt'
        )
        
        tuner.search(X_train, y_train, epochs=10, validation_split=0.2, verbose=1)
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        logger.info(f"Best Hyperparameters Found: {best_hps.values}")
