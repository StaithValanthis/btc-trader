# Updated file: app/services/ml_service.py

import asyncio
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from datetime import datetime, timezone
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Layer
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from structlog import get_logger
import ta
from sklearn.utils import class_weight
import collections

from app.services.backfill_service import backfill_bybit_kline
from app.core.database import Database
from app.core.config import Config

logger = get_logger(__name__)

MODEL_PATH = os.path.join("model_storage", "lstm_model.keras")
MIN_TRAINING_ROWS = 2000  # This minimum is for 1-min candles; after aggregation, adjust if needed.

# Get the labeling epsilon threshold from environment (default: 0.0005)
# TODO: Consider making this threshold dynamic based on current volatility
LABEL_EPSILON = float(os.getenv("LABEL_EPSILON", "0.0005"))

# Custom Attention Layer
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1), initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1), initializer="zeros")
        super(AttentionLayer, self).build(input_shape)
    
    def call(self, x):
        e = tf.nn.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
        a = tf.nn.softmax(e, axis=1)
        output = tf.reduce_sum(x * a, axis=1)
        return output

class MLService:
    def __init__(self, lookback=60):
        self.lookback = lookback  # Lookback is in 15-min intervals now.
        self.model = None
        self.initialized = False
        self.model_ready = False
        self.running = True
        self.epochs = 20

    async def initialize(self):
        try:
            if os.path.exists(MODEL_PATH):
                logger.info("Existing model found but will be ignored to enforce correct input shape.")
            else:
                logger.info("No existing model found; will create a new one on first training.")
            self.initialized = True
        except Exception as e:
            logger.error("Could not initialize MLService", error=str(e))
            self.initialized = False

    async def schedule_daily_retrain(self):
        while self.running:
            await self.train_model()
            await asyncio.sleep(86400)  # Retrain every 24 hours

    async def stop(self):
        self.running = False
        logger.info("MLService stopped")

    async def train_model(self):
        if not self.initialized:
            logger.warning("MLService not initialized; cannot train yet.")
            return

        logger.info("Retraining LSTM model with enhanced features on 15-minute aggregated data...")

        query = """
            SELECT time, open, high, low, close, volume
            FROM candles
            ORDER BY time ASC
        """
        rows = await Database.fetch(query)
        if not rows:
            logger.warning("No candle data found for training.")
            return

        # Convert rows to a DataFrame
        df = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close", "volume"])
        df['time'] = pd.to_datetime(df['time'])
        df.set_index("time", inplace=True)
        # Ensure data is at 1-min frequency (as inserted by the candle service)
        df = df.asfreq('1min')
        # Aggregate to 15-minute candles
        df_15 = df.resample('15min').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).ffill().dropna()
        
        # If insufficient rows exist, warn and exit.
        if len(df_15) < (MIN_TRAINING_ROWS // 15):
            logger.warning(f"Not enough 15-minute data for training; got {len(df_15)} rows.")
            return

        # Compute technical indicators on the 15-min aggregated data
        df_15["returns"] = df_15["close"].pct_change()
        df_15["rsi"] = ta.momentum.rsi(df_15["close"], window=14)
        macd = ta.trend.MACD(close=df_15["close"], window_slow=26, window_fast=12, window_sign=9)
        df_15["macd"] = macd.macd()
        df_15["macd_signal"] = macd.macd_signal()
        df_15["macd_diff"] = macd.macd_diff()
        boll = ta.volatility.BollingerBands(close=df_15["close"], window=20, window_dev=2)
        df_15["bb_high"] = boll.bollinger_hband()
        df_15["bb_low"] = boll.bollinger_lband()
        df_15["bb_mavg"] = boll.bollinger_mavg()
        atr_indicator = ta.volatility.AverageTrueRange(high=df_15["high"], low=df_15["low"], close=df_15["close"], window=14)
        df_15["atr"] = atr_indicator.average_true_range()

        # Additional features
        df_15["mfi"] = ta.volume.MFIIndicator(high=df_15["high"], low=df_15["low"], close=df_15["close"], volume=df_15["volume"], window=14).money_flow_index()
        stoch = ta.momentum.StochasticOscillator(high=df_15["high"], low=df_15["low"], close=df_15["close"], window=14, smooth_window=3)
        df_15["stoch"] = stoch.stoch()
        df_15["obv"] = ta.volume.OnBalanceVolumeIndicator(close=df_15["close"], volume=df_15["volume"]).on_balance_volume()
        df_15["vwap"] = ta.volume.VolumeWeightedAveragePrice(high=df_15["high"], low=df_15["low"], close=df_15["close"], volume=df_15["volume"]).volume_weighted_average_price()
        df_15["ema_20"] = ta.trend.EMAIndicator(close=df_15["close"], window=20).ema_indicator()
        df_15["cci"] = ta.trend.CCIIndicator(high=df_15["high"], low=df_15["low"], close=df_15["close"], window=20).cci()
        df_15["bb_width"] = df_15["bb_high"] - df_15["bb_low"]
        df_15["lag1_return"] = df_15["returns"].shift(1)

        # Ichimoku Cloud features
        ichimoku = ta.trend.IchimokuIndicator(high=df_15["high"], low=df_15["low"], window1=9, window2=26, window3=52)
        df_15["tenkan_sen"] = ichimoku.ichimoku_conversion_line()
        df_15["kijun_sen"] = ichimoku.ichimoku_base_line()
        df_15["senkou_span_a"] = ichimoku.ichimoku_a()
        df_15["senkou_span_b"] = ichimoku.ichimoku_b()

        # New Moving Average features
        df_15["sma_10"] = ta.trend.SMAIndicator(close=df_15["close"], window=10).sma_indicator()
        df_15["ema_10"] = ta.trend.EMAIndicator(close=df_15["close"], window=10).ema_indicator()
        df_15["smma_10"] = df_15["close"].ewm(alpha=1/10, adjust=False).mean()

        df_15.dropna(inplace=True)

        # Create multi-class target based on the 15-minute returns.
        # If the next 15-min return is greater than LABEL_EPSILON, label as Buy (1);
        # if less than -LABEL_EPSILON, label as Sell (0); otherwise, label as Hold (2).
        conditions = [
            (df_15["returns"].shift(-1) > LABEL_EPSILON),
            (df_15["returns"].shift(-1) < -LABEL_EPSILON)
        ]
        choices = [1, 0]
        df_15["target"] = np.select(conditions, choices, default=2)
        df_15.dropna(inplace=True)

        label_counts = collections.Counter(df_15["target"])
        logger.info("Label distribution in training data (15min)", label_distribution=label_counts)

        base_features = [
            "close", "returns", "rsi", "macd", "macd_signal", "macd_diff",
            "bb_high", "bb_low", "bb_mavg", "atr"
        ]
        additional_features = [
            "mfi", "stoch", "obv", "vwap", "ema_20", "cci", "bb_width", "lag1_return"
        ]
        ichimoku_features = [
            "tenkan_sen", "kijun_sen", "senkou_span_a", "senkou_span_b"
        ]
        moving_average_features = [
            "sma_10", "ema_10", "smma_10"
        ]
        feature_cols = base_features + additional_features + ichimoku_features + moving_average_features

        features = df_15[feature_cols].values
        labels = df_15["target"].values

        X, y = self._make_sequences(features, labels, self.lookback)
        if len(X) < 1:
            logger.warning("Not enough data for training after sequence creation.")
            return

        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        y_train_cat = to_categorical(y_train, num_classes=3)
        y_val_cat = to_categorical(y_val, num_classes=3)

        # Time-based weighting: assign more weight to recent samples (linearly increasing)
        weights = np.linspace(1, 2, num=len(X_train))

        if self.model is None:
            input_shape = (self.lookback, len(feature_cols))
            self.model = self._build_model(input_shape=input_shape)

        es = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
        history = self.model.fit(
            X_train, y_train_cat,
            validation_data=(X_val, y_val_cat),
            epochs=self.epochs,
            batch_size=32,
            callbacks=[es],
            verbose=1,
            sample_weight=weights
        )

        self.model.save(MODEL_PATH, save_format="keras")
        logger.info("Model training complete and saved", final_loss=history.history["loss"][-1])
        self.model_ready = True

    def predict_signal(self, recent_data: pd.DataFrame) -> str:
        if self.model is None:
            logger.warning("ML model not trained yet; returning 'Hold'.")
            return "Hold"

        data = recent_data.copy()
        if "returns" not in data.columns and "close" in data.columns:
            data["returns"] = data["close"].pct_change()
        if "rsi" not in data.columns and "close" in data.columns:
            data["rsi"] = ta.momentum.rsi(data["close"], window=14)
        if "macd" not in data.columns and "close" in data.columns:
            macd = ta.trend.MACD(data["close"], window_slow=26, window_fast=12, window_sign=9)
            data["macd"] = macd.macd()
            data["macd_signal"] = macd.macd_signal()
            data["macd_diff"] = macd.macd_diff()
        for col in ["bb_high", "bb_low", "bb_mavg"]:
            if col not in data.columns and "close" in data.columns:
                boll = ta.volatility.BollingerBands(data["close"], window=20, window_dev=2)
                data["bb_high"] = boll.bollinger_hband()
                data["bb_low"] = boll.bollinger_lband()
                data["bb_mavg"] = boll.bollinger_mavg()
                break
        if "atr" not in data.columns and {"high", "low", "close"}.issubset(data.columns):
            atr_indicator = ta.volatility.AverageTrueRange(
                high=data["high"], low=data["low"], close=data["close"], window=14
            )
            data["atr"] = atr_indicator.average_true_range()

        additional_features = {
            "mfi": lambda d: ta.volume.MFIIndicator(high=d["high"], low=d["low"], close=d["close"], volume=d["volume"], window=14).money_flow_index(),
            "stoch": lambda d: ta.momentum.StochasticOscillator(high=d["high"], low=d["low"], close=d["close"], window=14, smooth_window=3).stoch(),
            "obv": lambda d: ta.volume.OnBalanceVolumeIndicator(close=d["close"], volume=d["volume"]).on_balance_volume(),
            "vwap": lambda d: ta.volume.VolumeWeightedAveragePrice(high=d["high"], low=d["low"], close=d["close"], volume=d["volume"]).volume_weighted_average_price(),
            "ema_20": lambda d: ta.trend.EMAIndicator(close=d["close"], window=20).ema_indicator(),
            "cci": lambda d: ta.trend.CCIIndicator(high=d["high"], low=d["low"], close=d["close"], window=20).cci(),
            "bb_width": lambda d: d["bb_high"] - d["bb_low"],
            "lag1_return": lambda d: d["returns"].shift(1)
        }
        for feature, func in additional_features.items():
            if feature not in data.columns:
                try:
                    data[feature] = func(data)
                except Exception as ex:
                    logger.warning(f"Could not compute {feature}: {ex}")
        if "sma_10" not in data.columns:
            data["sma_10"] = ta.trend.SMAIndicator(close=data["close"], window=10).sma_indicator()
        if "ema_10" not in data.columns:
            data["ema_10"] = ta.trend.EMAIndicator(close=data["close"], window=10).ema_indicator()
        if "smma_10" not in data.columns:
            data["smma_10"] = data["close"].ewm(alpha=1/10, adjust=False).mean()
        required_ichimoku = ["tenkan_sen", "kijun_sen", "senkou_span_a", "senkou_span_b"]
        if not all(col in data.columns for col in required_ichimoku):
            ichimoku = ta.trend.IchimokuIndicator(
                high=data["high"],
                low=data["low"],
                window1=9,
                window2=26,
                window3=52
            )
            data["tenkan_sen"] = ichimoku.ichimoku_conversion_line()
            data["kijun_sen"] = ichimoku.ichimoku_base_line()
            data["senkou_span_a"] = ichimoku.ichimoku_a()
            data["senkou_span_b"] = ichimoku.ichimoku_b()
        feature_cols = [
            "close", "returns", "rsi", "macd", "macd_signal", "macd_diff",
            "bb_high", "bb_low", "bb_mavg", "atr",
            "mfi", "stoch", "obv", "vwap", "ema_20", "cci", "bb_width", "lag1_return",
            "tenkan_sen", "kijun_sen", "senkou_span_a", "senkou_span_b",
            "sma_10", "ema_10", "smma_10"
        ]
        for col in feature_cols:
            if col not in data.columns:
                logger.warning(f"Missing column {col}; cannot predict.")
                return "Hold"
        seq = data[feature_cols].tail(self.lookback).copy()
        seq = seq.ffill().bfill()
        if len(seq) < self.lookback:
            missing_count = self.lookback - len(seq)
            pad = pd.DataFrame([seq.iloc[0].values] * missing_count, columns=seq.columns)
            seq = pd.concat([pad, seq], ignore_index=True)
            logger.info("Padded the sequence to meet the required lookback length",
                        padded_rows=missing_count, new_length=len(seq))
        data_seq = seq.values
        data_seq = np.expand_dims(data_seq, axis=0)
        logger.info("Predicting signal using data sequence", shape=data_seq.shape)
        pred = self.model.predict(data_seq)
        class_idx = np.argmax(pred, axis=1)[0]
        if class_idx == 1:
            return "Buy"
        elif class_idx == 0:
            return "Sell"
        else:
            return "Hold"

    def _build_model(self, input_shape):
        inputs = Input(shape=input_shape)
        x = LSTM(64, return_sequences=True)(inputs)
        x = Dropout(0.2)(x)
        # Apply attention mechanism
        x = AttentionLayer()(x)
        outputs = Dense(3, activation="softmax")(x)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        return model

    def _make_sequences(self, features, labels, lookback):
        X, y = [], []
        for i in range(len(features) - lookback):
            X.append(features[i:i+lookback])
            y.append(labels[i+lookback])
        return np.array(X), np.array(y)
