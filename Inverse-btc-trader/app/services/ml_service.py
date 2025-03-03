# File: app/services/ml_service.py

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
import ta
from structlog import get_logger
import collections

from app.services.backfill_service import backfill_bybit_kline
from app.core.database import Database
from app.core.config import Config

logger = get_logger(__name__)

# Paths for saving both models
SIGNAL_MODEL_PATH = os.path.join("model_storage", "signal_model.keras")
TREND_MODEL_PATH = os.path.join("model_storage", "trend_model.keras")

# Minimum training rows for 15-min candles
MIN_TRAINING_ROWS = 2000  
LABEL_EPSILON = float(os.getenv("LABEL_EPSILON", "0.0005"))

# ---------------------
#  Custom Attention
# ---------------------
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

# ---------------------
#  MLService Class
# ---------------------
class MLService:
    """
    Two-model architecture:
      1) TrendModel: Classify the trend as [Up, Down, Sideways].
      2) SignalModel: Classify next move as [Buy, Sell, Hold].
    Uses an extended lookback (default=120) and merges signals to avoid countertrend trades.
    Both training and inference share the exact same set of features to prevent shape mismatches.
    """
    def __init__(self, lookback=120):
        self.lookback = lookback
        self.signal_model = None
        self.trend_model = None

        self.initialized = False
        self.signal_model_ready = False
        self.trend_model_ready = False
        self.running = True
        self.epochs = 20

        # IMPORTANT: The definitive list of feature columns used in both training & inference
        self.feature_cols = [
            "close", "returns", "rsi", "macd", "macd_signal", "macd_diff",
            "bb_high", "bb_low", "bb_mavg", "atr",
            "mfi", "stoch", "obv", "vwap", "ema_20", "cci",
            "bb_width", "lag1_return",
            "tenkan_sen", "kijun_sen", "senkou_span_a", "senkou_span_b",
            "sma_10", "ema_10", "smma_10",
            "ADX",
            # We'll add one-hot regime columns forcibly:
            # "regime_uptrending", "regime_downtrending", "regime_sideways"
        ]
        # We'll dynamically append "regime_uptrending", etc. if they're missing.

    async def initialize(self):
        """
        Check if existing models are on disk. If so, we can load them. 
        Otherwise, new models will be created at first training.
        """
        try:
            if os.path.exists(SIGNAL_MODEL_PATH):
                logger.info("Found existing SignalModel on disk.")
            else:
                logger.info("No existing SignalModel found; will create a new one on first training.")

            if os.path.exists(TREND_MODEL_PATH):
                logger.info("Found existing TrendModel on disk.")
            else:
                logger.info("No existing TrendModel found; will create a new one on first training.")

            self.initialized = True
        except Exception as e:
            logger.error("Could not initialize MLService", error=str(e))
            self.initialized = False

    async def schedule_daily_retrain(self):
        """
        Retrain both models once every 24h in the background.
        """
        while self.running:
            await self.train_model()
            await asyncio.sleep(86400)  # 24 hours

    async def stop(self):
        self.running = False
        logger.info("MLService stopped")

    # =========================
    #  Main Training Method
    # =========================
    async def train_model(self):
        if not self.initialized:
            logger.warning("MLService not initialized; cannot train yet.")
            return

        logger.info("Retraining LSTM models with extended lookback & consistent features (15-min data).")
        query = """
            SELECT time, open, high, low, close, volume
            FROM candles
            ORDER BY time ASC
        """
        rows = await Database.fetch(query)
        if not rows:
            logger.warning("No candle data found for training.")
            return

        df = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close", "volume"])
        df['time'] = pd.to_datetime(df['time'])
        df.set_index("time", inplace=True)
        df = df.asfreq('1min')

        # Resample to 15-min
        df_15 = df.resample('15min').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).ffill().dropna()

        if len(df_15) < (MIN_TRAINING_ROWS // 15):
            logger.warning(f"Not enough 15-min data for training; got {len(df_15)} rows.")
            return

        # Compute basic technicals
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

        atr_indicator = ta.volatility.AverageTrueRange(
            high=df_15["high"], low=df_15["low"], close=df_15["close"], window=14
        )
        df_15["atr"] = atr_indicator.average_true_range()

        # Additional features
        df_15["mfi"] = ta.volume.MFIIndicator(
            high=df_15["high"], low=df_15["low"], close=df_15["close"], volume=df_15["volume"], window=14
        ).money_flow_index()

        stoch = ta.momentum.StochasticOscillator(
            high=df_15["high"], low=df_15["low"], close=df_15["close"],
            window=14, smooth_window=3
        )
        df_15["stoch"] = stoch.stoch()
        df_15["obv"] = ta.volume.OnBalanceVolumeIndicator(
            close=df_15["close"], volume=df_15["volume"]
        ).on_balance_volume()
        df_15["vwap"] = ta.volume.VolumeWeightedAveragePrice(
            high=df_15["high"], low=df_15["low"], close=df_15["close"], volume=df_15["volume"]
        ).volume_weighted_average_price()
        df_15["ema_20"] = ta.trend.EMAIndicator(close=df_15["close"], window=20).ema_indicator()
        df_15["cci"] = ta.trend.CCIIndicator(
            high=df_15["high"], low=df_15["low"], close=df_15["close"], window=20
        ).cci()
        df_15["bb_width"] = df_15["bb_high"] - df_15["bb_low"]
        df_15["lag1_return"] = df_15["returns"].shift(1)

        # Ichimoku
        ichimoku = ta.trend.IchimokuIndicator(
            high=df_15["high"], low=df_15["low"], window1=9, window2=26, window3=52
        )
        df_15["tenkan_sen"] = ichimoku.ichimoku_conversion_line()
        df_15["kijun_sen"] = ichimoku.ichimoku_base_line()
        df_15["senkou_span_a"] = ichimoku.ichimoku_a()
        df_15["senkou_span_b"] = ichimoku.ichimoku_b()

        # More MAs
        df_15["sma_10"] = ta.trend.SMAIndicator(df_15["close"], window=10).sma_indicator()
        df_15["ema_10"] = ta.trend.EMAIndicator(df_15["close"], window=10).ema_indicator()
        df_15["smma_10"] = df_15["close"].ewm(alpha=1/10, adjust=False).mean()

        # Regime detection
        df_15["SMA20"] = df_15["close"].rolling(window=20).mean()
        df_15["SMA50"] = df_15["close"].rolling(window=50).mean()
        adx_indicator = ta.trend.ADXIndicator(
            high=df_15["high"], low=df_15["low"], close=df_15["close"], window=14
        )
        df_15["ADX"] = adx_indicator.adx()

        df_15["ma_trend"] = np.where(df_15["SMA20"] > df_15["SMA50"], "up", "down")
        df_15["regime"] = "sideways"  # default
        df_15.loc[(df_15["ADX"] > 25) & (df_15["ma_trend"]=="up"), "regime"] = "uptrending"
        df_15.loc[(df_15["ADX"] > 25) & (df_15["ma_trend"]=="down"), "regime"] = "downtrending"

        # Trend targets
        regime_map = {"uptrending": 0, "downtrending": 1, "sideways": 2}
        df_15["trend_target"] = df_15["regime"].map(regime_map)

        # One-hot encode the regime
        regime_dummies = pd.get_dummies(df_15["regime"], prefix="regime")
        df_15 = pd.concat([df_15, regime_dummies], axis=1)

        df_15.dropna(inplace=True)

        # Signal target: [Buy=1, Sell=0, Hold=2]
        conditions = [
            (df_15["returns"].shift(-1) > LABEL_EPSILON),
            (df_15["returns"].shift(-1) < -LABEL_EPSILON)
        ]
        df_15["signal_target"] = np.select(conditions, [1, 0], default=2)
        df_15.dropna(inplace=True)

        logger.info("Signal label distribution", 
                    distribution=collections.Counter(df_15["signal_target"]))
        logger.info("Trend label distribution", 
                    distribution=collections.Counter(df_15["trend_target"]))

        # Ensure columns: add missing regime_* columns
        for cat_col in ["regime_uptrending", "regime_downtrending", "regime_sideways"]:
            if cat_col not in df_15.columns:
                df_15[cat_col] = 0

        actual_cols = [c for c in self.feature_cols if c in df_15.columns]
        missing_cols = set(self.feature_cols) - set(actual_cols)
        if missing_cols:
            logger.warning(f"Missing feature columns: {missing_cols}. They won't be used in training.")

        # Convert to numeric & drop any NaN
        df_15[actual_cols] = df_15[actual_cols].apply(pd.to_numeric, errors="coerce")
        before_drop = len(df_15)
        df_15.dropna(subset=actual_cols, inplace=True)
        after_drop = len(df_15)
        if after_drop < before_drop:
            logger.info(f"Dropped {before_drop - after_drop} rows due to NaN in features.")

        features = df_15[actual_cols].values
        trend_labels = df_15["trend_target"].astype(np.int32).values
        signal_labels = df_15["signal_target"].astype(np.int32).values

        # Make sequences
        X_trend, y_trend = self._make_sequences(features, trend_labels, self.lookback)
        X_signal, y_signal = self._make_sequences(features, signal_labels, self.lookback)

        if len(X_trend) < 1 or len(X_signal) < 1:
            logger.warning("Not enough data after sequence creation.")
            return

        # Time-based train/val split
        split_idx_trend = int(len(X_trend) * 0.8)
        X_trend_train, X_trend_val = X_trend[:split_idx_trend], X_trend[split_idx_trend:]
        y_trend_train, y_trend_val = y_trend[:split_idx_trend], y_trend[split_idx_trend:]

        split_idx_signal = int(len(X_signal) * 0.8)
        X_signal_train, X_signal_val = X_signal[:split_idx_signal], X_signal[split_idx_signal:]
        y_signal_train, y_signal_val = y_signal[:split_idx_signal], y_signal[split_idx_signal:]

        # One-hot
        y_trend_train_cat = to_categorical(y_trend_train, num_classes=3)
        y_trend_val_cat   = to_categorical(y_trend_val,   num_classes=3)
        y_signal_train_cat = to_categorical(y_signal_train, num_classes=3)
        y_signal_val_cat   = to_categorical(y_signal_val,   num_classes=3)

        # Sample weights
        weights_trend  = np.linspace(1, 2, num=len(X_trend_train))
        weights_signal = np.linspace(1, 2, num=len(X_signal_train))

        # Final cast to float32, replace NaN/inf with 0
        X_trend_train = np.nan_to_num(X_trend_train, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        X_trend_val   = np.nan_to_num(X_trend_val,   nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        y_trend_train_cat = np.nan_to_num(y_trend_train_cat, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        y_trend_val_cat   = np.nan_to_num(y_trend_val_cat,   nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        X_signal_train = np.nan_to_num(X_signal_train, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        X_signal_val   = np.nan_to_num(X_signal_val,   nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        y_signal_train_cat = np.nan_to_num(y_signal_train_cat, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        y_signal_val_cat   = np.nan_to_num(y_signal_val_cat,   nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        # Debug shapes
        logger.info(f"X_trend_train.shape={X_trend_train.shape}, dtype={X_trend_train.dtype}")
        logger.info(f"X_signal_train.shape={X_signal_train.shape}, dtype={X_signal_train.dtype}")

        if X_trend_train.shape[0] == 0 or X_signal_train.shape[0] == 0:
            logger.warning("Empty training sets after final cleaning.")
            return

        # Build or reuse models
        input_shape = (self.lookback, X_trend_train.shape[2])
        if self.trend_model is None:
            self.trend_model = self._build_model(input_shape, num_classes=3)
        if self.signal_model is None:
            self.signal_model = self._build_model(input_shape, num_classes=3)

        es_trend = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
        es_signal = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

        # Train TrendModel
        history_trend = self.trend_model.fit(
            X_trend_train, y_trend_train_cat,
            validation_data=(X_trend_val, y_trend_val_cat),
            epochs=self.epochs,
            batch_size=32,
            callbacks=[es_trend],
            verbose=1,
            sample_weight=weights_trend
        )
        self.trend_model.save(TREND_MODEL_PATH)
        logger.info("TrendModel training complete",
                    final_loss=history_trend.history["loss"][-1] if history_trend.history["loss"] else None)
        self.trend_model_ready = True

        # Train SignalModel
        history_signal = self.signal_model.fit(
            X_signal_train, y_signal_train_cat,
            validation_data=(X_signal_val, y_signal_val_cat),
            epochs=self.epochs,
            batch_size=32,
            callbacks=[es_signal],
            verbose=1,
            sample_weight=weights_signal
        )
        self.signal_model.save(SIGNAL_MODEL_PATH)
        logger.info("SignalModel training complete",
                    final_loss=history_signal.history["loss"][-1] if history_signal.history["loss"] else None)
        self.signal_model_ready = True

    def predict_signal(self, recent_data: pd.DataFrame) -> str:
        """
        Ensemble approach: 
          1) TrendModel => up/down/side
          2) SignalModel => buy/sell/hold
          3) If mismatch with strong trend, override to "Hold"
        """
        if not self.trend_model_ready or not self.signal_model_ready:
            logger.warning("ML models not ready; returning 'Hold'.")
            return "Hold"

        data_seq = self._prepare_data_sequence(recent_data)
        if data_seq is None:
            return "Hold"

        trend_pred = self.trend_model.predict(data_seq)
        trend_class = np.argmax(trend_pred, axis=1)[0]  # 0=up,1=down,2=side

        signal_pred = self.signal_model.predict(data_seq)
        signal_class = np.argmax(signal_pred, axis=1)[0]  # 0=sell,1=buy,2=hold

        trend_label = {0: "uptrending", 1: "downtrending", 2: "sideways"}.get(trend_class, "sideways")
        signal_label = {0: "Sell", 1: "Buy", 2: "Hold"}.get(signal_class, "Hold")

        final_label = signal_label
        if trend_label == "uptrending" and signal_label == "Sell":
            final_label = "Hold"
        elif trend_label == "downtrending" and signal_label == "Buy":
            final_label = "Hold"

        logger.info("Ensemble Prediction",
                    trend=trend_label,
                    raw_signal=signal_label,
                    final_signal=final_label)
        return final_label

    # ================================
    #  Helper Methods
    # ================================
    def _build_model(self, input_shape, num_classes=3):
        """
        Simple LSTM -> Attention -> Dense architecture for classification
        """
        inputs = Input(shape=input_shape)
        x = LSTM(64, return_sequences=True)(inputs)
        x = Dropout(0.2)(x)
        x = AttentionLayer()(x)
        outputs = Dense(num_classes, activation="softmax")(x)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        return model

    def _make_sequences(self, features, labels, lookback):
        """
        Convert the entire dataset into (X, y) sequences of length=lookback
        """
        X, y = [], []
        for i in range(len(features) - lookback):
            X.append(features[i : i + lookback])
            y.append(labels[i + lookback])
        return np.array(X), np.array(y)

    def _prepare_data_sequence(self, df: pd.DataFrame):
        """
        Prepare a single inference sequence from recent_data 
        using the same transformations as in train_model.
        If not enough data => returns None.
        """
        data = df.copy()
        data = data.asfreq('1min')

        # 15-min resample
        df_15 = data.resample('15min').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).ffill().dropna()

        if len(df_15) < self.lookback:
            logger.warning("Not enough 15-min bars for inference.")
            return None

        # Repeat the same technical indicator calculations
        df_15["returns"] = df_15["close"].pct_change()
        df_15["rsi"] = ta.momentum.rsi(df_15["close"], window=14)
        macd = ta.trend.MACD(df_15["close"], window_slow=26, window_fast=12, window_sign=9)
        df_15["macd"] = macd.macd()
        df_15["macd_signal"] = macd.macd_signal()
        df_15["macd_diff"] = macd.macd_diff()

        boll = ta.volatility.BollingerBands(df_15["close"], window=20, window_dev=2)
        df_15["bb_high"] = boll.bollinger_hband()
        df_15["bb_low"] = boll.bollinger_lband()
        df_15["bb_mavg"] = boll.bollinger_mavg()

        atr_indicator = ta.volatility.AverageTrueRange(
            high=df_15["high"], low=df_15["low"], close=df_15["close"], window=14
        )
        df_15["atr"] = atr_indicator.average_true_range()

        df_15["mfi"] = ta.volume.MFIIndicator(
            high=df_15["high"], low=df_15["low"], 
            close=df_15["close"], volume=df_15["volume"], window=14
        ).money_flow_index()

        stoch = ta.momentum.StochasticOscillator(
            high=df_15["high"], low=df_15["low"], close=df_15["close"],
            window=14, smooth_window=3
        )
        df_15["stoch"] = stoch.stoch()
        df_15["obv"] = ta.volume.OnBalanceVolumeIndicator(
            close=df_15["close"], volume=df_15["volume"]
        ).on_balance_volume()
        df_15["vwap"] = ta.volume.VolumeWeightedAveragePrice(
            high=df_15["high"], low=df_15["low"], close=df_15["close"], volume=df_15["volume"]
        ).volume_weighted_average_price()
        df_15["ema_20"] = ta.trend.EMAIndicator(df_15["close"], window=20).ema_indicator()
        df_15["cci"] = ta.trend.CCIIndicator(
            high=df_15["high"], low=df_15["low"], close=df_15["close"], window=20
        ).cci()
        df_15["bb_width"] = df_15["bb_high"] - df_15["bb_low"]
        df_15["lag1_return"] = df_15["returns"].shift(1)

        ichimoku = ta.trend.IchimokuIndicator(
            high=df_15["high"], low=df_15["low"], window1=9, window2=26, window3=52
        )
        df_15["tenkan_sen"] = ichimoku.ichimoku_conversion_line()
        df_15["kijun_sen"] = ichimoku.ichimoku_base_line()
        df_15["senkou_span_a"] = ichimoku.ichimoku_a()
        df_15["senkou_span_b"] = ichimoku.ichimoku_b()

        df_15["sma_10"] = ta.trend.SMAIndicator(df_15["close"], window=10).sma_indicator()
        df_15["ema_10"] = ta.trend.EMAIndicator(df_15["close"], window=10).ema_indicator()
        df_15["smma_10"] = df_15["close"].ewm(alpha=1/10, adjust=False).mean()

        # Regime
        df_15["SMA20"] = df_15["close"].rolling(window=20).mean()
        df_15["SMA50"] = df_15["close"].rolling(window=50).mean()
        adx_indicator = ta.trend.ADXIndicator(
            high=df_15["high"], low=df_15["low"], close=df_15["close"], window=14
        )
        df_15["ADX"] = adx_indicator.adx()
        df_15["ma_trend"] = np.where(df_15["SMA20"] > df_15["SMA50"], "up", "down")
        df_15["regime"] = "sideways"
        df_15.loc[(df_15["ADX"] > 25) & (df_15["ma_trend"]=="up"), "regime"] = "uptrending"
        df_15.loc[(df_15["ADX"] > 25) & (df_15["ma_trend"]=="down"), "regime"] = "downtrending"

        regime_dummies = pd.get_dummies(df_15["regime"], prefix="regime")
        df_15 = pd.concat([df_15, regime_dummies], axis=1)

        for cat_col in ["regime_uptrending", "regime_downtrending", "regime_sideways"]:
            if cat_col not in df_15.columns:
                df_15[cat_col] = 0

        df_15.dropna(inplace=True)

        # Build final sequence from self.feature_cols
        actual_cols = [c for c in self.feature_cols if c in df_15.columns]
        # Convert to numeric, drop NaN
        df_15[actual_cols] = df_15[actual_cols].apply(pd.to_numeric, errors="coerce")
        df_15.dropna(subset=actual_cols, inplace=True)

        # Grab last `self.lookback` rows
        recent_slice = df_15.tail(self.lookback).copy()
        if len(recent_slice) < self.lookback:
            missing = self.lookback - len(recent_slice)
            if len(df_15) == 0:
                logger.warning("No data left after indicator calculations.")
                return None
            # Pad with the earliest row in recent_slice
            pad_df = pd.DataFrame([recent_slice.iloc[0].values]*missing, columns=recent_slice.columns)
            recent_slice = pd.concat([pad_df, recent_slice], ignore_index=True)

        seq = recent_slice[actual_cols].values
        seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        seq = np.expand_dims(seq, axis=0)  # shape => (1, lookback, num_features)
        return seq