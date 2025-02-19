# File: app/services/ml_service.py

import asyncio
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from datetime import datetime, timezone
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from structlog import get_logger

import ta  # Used for ATR, RSI, MACD, Bollinger Bands, etc.

# Use the new function name (if needed)
from app.services.backfill_service import backfill_bybit_kline

from app.core.database import Database
from app.core.config import Config

logger = get_logger(__name__)

MODEL_PATH = os.path.join("model_storage", "lstm_model.keras")

# Require 2000 rows for a decent LSTM start (from candles table)
MIN_TRAINING_ROWS = 2000  

class MLService:
    """
    A service to train and predict market direction using an LSTM model.
    This version uses historical candle (OHLCV) data and computes the following features:
      - close, returns, rsi, macd, macd_signal, macd_diff, bb_high, bb_low, bb_mavg, atr
    (A total of 10 features.)
    """
    def __init__(self, lookback=60):
        """
        :param lookback: Number of timesteps (candles) to look back for LSTM sequence.
        """
        self.lookback = lookback
        self.model = None
        self.initialized = False
        self.model_ready = False  # Flag indicating if a trained model exists
        self.running = True       # Used for schedule_daily_retrain
        self.epochs = 20          # Number of training epochs

    async def initialize(self):
        """
        Initialize the ML service. To enforce the correct input shape (lookback x 10),
        we ignore any pre-existing model file.
        (In production, you might wish to handle versioning instead.)
        """
        try:
            if os.path.exists(MODEL_PATH):
                # Force a rebuild by ignoring any pre-existing model.
                logger.info("Existing model found but will be ignored to enforce correct input shape.")
            else:
                logger.info("No existing model found; will create a new one on first training.")
            self.initialized = True
        except Exception as e:
            logger.error("Could not initialize MLService", error=str(e))
            self.initialized = False

    async def schedule_daily_retrain(self):
        """
        Periodically retrain the model with the latest candle data.
        """
        while self.running:
            await self.train_model()
            await asyncio.sleep(86400)  # Sleep for 24 hours

    async def stop(self):
        """Stop the ML service so that daily retraining ends."""
        self.running = False
        logger.info("MLService stopped")

    async def train_model(self):
        """
        Query the database for candle data, compute ATR and other indicators,
        and retrain the LSTM model.
        If there's not enough data, attempt a backfill.
        """
        if not self.initialized:
            logger.warning("MLService not initialized; cannot train yet.")
            return

        logger.info("Retraining LSTM model with candle + ATR data...")

        # 1. Fetch candle data from the 'candles' table.
        query = """
            SELECT time, open, high, low, close, volume
            FROM candles
            ORDER BY time ASC
        """
        rows = await Database.fetch(query)
        if not rows:
            logger.warning("No candle data found for training.")
            return

        # Trigger backfill if not enough rows.
        if len(rows) < MIN_TRAINING_ROWS:
            logger.warning(f"Only {len(rows)} rows found. Initiating backfill to reach {MIN_TRAINING_ROWS} rows.")
            await backfill_bybit_kline(
                symbol="BTCUSD",
                interval=1,
                start_time_ms=1676000000000,  # example start time (adjust as needed)
                days_to_fetch=7
            )
            # Re-fetch candle data after backfill.
            rows = await Database.fetch(query)
            if len(rows) < MIN_TRAINING_ROWS:
                logger.warning(f"Still only {len(rows)} rows after backfill. Skipping training.")
                return

        df = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close", "volume"])
        df.set_index("time", inplace=True)
        df.sort_index(inplace=True)

        # 2. Compute technical indicators using 'close' price.
        df["returns"] = df["close"].pct_change()
        df["rsi"] = ta.momentum.rsi(df["close"], window=14)

        macd = ta.trend.MACD(close=df["close"], window_slow=26, window_fast=12, window_sign=9)
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["macd_diff"] = macd.macd_diff()

        boll = ta.volatility.BollingerBands(close=df["close"], window=20, window_dev=2)
        df["bb_high"] = boll.bollinger_hband()
        df["bb_low"] = boll.bollinger_lband()
        df["bb_mavg"] = boll.bollinger_mavg()

        atr_indicator = ta.volatility.AverageTrueRange(
            high=df["high"],
            low=df["low"],
            close=df["close"],
            window=14
        )
        df["atr"] = atr_indicator.average_true_range()

        df.dropna(inplace=True)

        # 3. Create classification target: 1 if next bar's return > 0, else 0.
        df["target"] = (df["returns"].shift(-1) > 0).astype(int)
        df.dropna(inplace=True)

        # 4. Build feature array with 10 features.
        feature_cols = [
            "close", "returns", "rsi",
            "macd", "macd_signal", "macd_diff",
            "bb_high", "bb_low", "bb_mavg", "atr"
        ]
        features = df[feature_cols].values
        labels = df["target"].values

        # 5. Create LSTM sequences.
        X, y = self._make_sequences(features, labels, self.lookback)
        if len(X) < 1:
            logger.warning("Not enough data for training after sequence creation.")
            return

        # 6. Split data into training and validation sets.
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        # 7. Build or reuse the model.
        if self.model is None:
            input_shape = (self.lookback, len(feature_cols))
            self.model = self._build_model(input_shape=input_shape)

        # 8. Train the model.
        es = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
        history = self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=self.epochs,
            batch_size=32,
            callbacks=[es],
            verbose=1
        )

        # 9. Save the model.
        self.model.save(MODEL_PATH, save_format="keras")
        logger.info("Model training complete and saved", final_loss=history.history["loss"][-1])
        self.model_ready = True

    def predict_signal(self, recent_data: pd.DataFrame) -> str:
        """
        Predict the trading signal ("Buy" or "Sell") using the trained LSTM.
        Expects recent_data to have at least self.lookback rows and include the following columns:
            close, returns, rsi, macd, macd_signal, macd_diff, bb_high, bb_low, bb_mavg, atr.
        """
        # Ensure required features are present; compute if necessary.
        if "returns" not in recent_data.columns and "close" in recent_data.columns:
            recent_data = recent_data.copy()
            recent_data["returns"] = recent_data["close"].pct_change()
        if "rsi" not in recent_data.columns and "close" in recent_data.columns:
            recent_data = recent_data.copy()
            recent_data["rsi"] = ta.momentum.rsi(recent_data["close"], window=14)
        if "macd" not in recent_data.columns and "close" in recent_data.columns:
            recent_data = recent_data.copy()
            macd = ta.trend.MACD(recent_data["close"], window_slow=26, window_fast=12, window_sign=9)
            recent_data["macd"] = macd.macd()
            recent_data["macd_signal"] = macd.macd_signal()
            recent_data["macd_diff"] = macd.macd_diff()
        for col in ["bb_high", "bb_low", "bb_mavg"]:
            if col not in recent_data.columns and "close" in recent_data.columns:
                recent_data = recent_data.copy()
                boll = ta.volatility.BollingerBands(recent_data["close"], window=20, window_dev=2)
                recent_data["bb_high"] = boll.bollinger_hband()
                recent_data["bb_low"] = boll.bollinger_lband()
                recent_data["bb_mavg"] = boll.bollinger_mavg()
                break
        if "atr" not in recent_data.columns and {"high", "low", "close"}.issubset(recent_data.columns):
            recent_data = recent_data.copy()
            atr_indicator = ta.volatility.AverageTrueRange(
                high=recent_data["high"],
                low=recent_data["low"],
                close=recent_data["close"],
                window=14
            )
            recent_data["atr"] = atr_indicator.average_true_range()

        feature_cols = [
            "close", "returns", "rsi",
            "macd", "macd_signal", "macd_diff",
            "bb_high", "bb_low", "bb_mavg", "atr"
        ]
        for col in feature_cols:
            if col not in recent_data.columns:
                logger.warning(f"Missing column {col}; cannot predict.")
                return "Hold"

        # Instead of dropping rows, use only the last `self.lookback` rows and fill any missing values.
        seq = recent_data[feature_cols].tail(self.lookback).copy()
        seq = seq.ffill().bfill()  # fill forward and backward in case of NaNs

        if len(seq) < self.lookback:
            logger.warning("Not enough data for a full sequence after fillna; defaulting to 'Hold'.")
            return "Hold"
        
        data_seq = seq.values
        data_seq = np.expand_dims(data_seq, axis=0)  # shape: (1, lookback, num_features)
        logger.info("Predicting signal using data sequence", shape=data_seq.shape)
        pred = self.model.predict(data_seq)
        return "Buy" if pred[0][0] > 0.5 else "Sell"


    def _build_model(self, input_shape):
        """
        Build a simple LSTM model for binary classification: next price up or down.
        The model expects an input shape of (lookback, num_features).
        """
        model = Sequential()
        model.add(LSTM(64, input_shape=input_shape, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation="sigmoid"))
        model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
        return model

    def _make_sequences(self, features, labels, lookback):
        """
        Create sequences of length 'lookback' for LSTM input.
        """
        X, y = [], []
        for i in range(len(features) - lookback):
            X.append(features[i : i + lookback])
            y.append(labels[i + lookback])
        return np.array(X), np.array(y)
