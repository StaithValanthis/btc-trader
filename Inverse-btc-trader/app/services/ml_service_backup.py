# File: app/services/ml_service.py

import asyncio
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from datetime import datetime, timezone
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Layer, Conv1D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
import ta
from structlog import get_logger
import collections
from sklearn.decomposition import PCA
from sklearn.utils.class_weight import compute_class_weight

from app.services.backfill_service import backfill_bybit_kline
from app.core.database import Database
from app.core.config import Config

logger = get_logger(__name__)

# Paths for saving models
SIGNAL_MODEL_PATH = os.path.join("model_storage", "signal_model.keras")
TREND_MODEL_PATH = os.path.join("model_storage", "trend_model.keras")

MIN_TRAINING_ROWS = 2000  
LABEL_EPSILON = float(os.getenv("LABEL_EPSILON", "0.0005"))

# ---------------------
#  Custom Attention Layer
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
#  Focal Loss Function
# ---------------------
def focal_loss(gamma=2., alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = 1e-8
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * tf.pow(1 - y_pred, gamma)
        loss = weight * cross_entropy
        return tf.reduce_mean(tf.reduce_sum(loss, axis=1))
    return focal_loss_fixed

# ---------------------
#  MLService Class
# ---------------------
class MLService:
    """
    Two-model architecture:
      1) TrendModel: Classify the trend as [Up, Down, Sideways].
      2) SignalModel: Classify next move as [Buy, Sell, Hold].
    
    Enhancements include:
      - Lag features and rolling statistics,
      - Additional technical indicators (e.g., CMF, DMI difference),
      - Nonlinear interaction features,
      - An extra LSTM layer in the model,
      - Class imbalance handling via computed class weights and focal loss for the SignalModel,
      - PCA for dimensionality reduction.
    
    Note: You may further experiment with signal target generation and temporal cross-validation.
    """
    def __init__(self, lookback=120):
        self.lookback = lookback
        self.signal_model = None
        self.trend_model = None
        self.pca = None  # PCA transformer

        self.initialized = False
        self.signal_model_ready = False
        self.trend_model_ready = False
        self.running = True
        self.epochs = 20

        # Expected feature list (27 features)
        self.feature_cols = [
            "close", "returns", "rsi", "macd", "macd_diff",
            "bb_high", "bb_low", "bb_mavg", "atr",
            "mfi", "stoch", "obv", "vwap", "ema_20", "cci",
            "bb_width",
            "volatility_ratio",              # (high - low) / close
            "tenkan_sen", "kijun_sen", "senkou_span_a", "ADX",
            "rsi_lag1", "returns_rolling_mean", "returns_rolling_std",
            "cmf", "dmi_diff",
            "rsi_atr_interaction", "macd_diff_atr_interaction"
        ]

    async def initialize(self):
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

        # Resample to 15-min candles.
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

        # Basic technical indicators.
        df_15["returns"] = df_15["close"].pct_change()
        df_15["rsi"] = ta.momentum.rsi(df_15["close"], window=14)
        macd = ta.trend.MACD(close=df_15["close"], window_slow=26, window_fast=12, window_sign=9)
        df_15["macd"] = macd.macd()
        df_15["macd_diff"] = macd.macd_diff()

        boll = ta.volatility.BollingerBands(close=df_15["close"], window=20, window_dev=2)
        df_15["bb_high"] = boll.bollinger_hband()
        df_15["bb_low"] = boll.bollinger_lband()
        df_15["bb_mavg"] = boll.bollinger_mavg()

        atr_indicator = ta.volatility.AverageTrueRange(
            high=df_15["high"], low=df_15["low"], close=df_15["close"], window=14
        )
        df_15["atr"] = atr_indicator.average_true_range()

        df_15["mfi"] = ta.volume.MFIIndicator(
            high=df_15["high"],
            low=df_15["low"],
            close=df_15["close"],
            volume=df_15["volume"],
            window=14
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

        # New feature: volatility_ratio = (high - low) / close.
        df_15["volatility_ratio"] = (df_15["high"] - df_15["low"]) / df_15["close"]

        # Lag and rolling statistics.
        df_15["rsi_lag1"] = df_15["rsi"].shift(1)
        df_15["returns_rolling_mean"] = df_15["returns"].rolling(window=3).mean()
        df_15["returns_rolling_std"] = df_15["returns"].rolling(window=3).std()

        # Additional technical indicators.
        df_15["cmf"] = ta.volume.ChaikinMoneyFlowIndicator(
            high=df_15["high"],
            low=df_15["low"],
            close=df_15["close"],
            volume=df_15["volume"],
            window=20
        ).chaikin_money_flow()
        try:
            from ta.trend import DMIIndicator
            dmi = DMIIndicator(high=df_15["high"], low=df_15["low"], close=df_15["close"], window=14)
            df_15["plus_di"] = dmi.plus_di()
            df_15["minus_di"] = dmi.minus_di()
            df_15["dmi_diff"] = df_15["plus_di"] - df_15["minus_di"]
        except (AttributeError, ImportError):
            df_15["dmi_diff"] = 0

        # Feature interaction terms.
        df_15["rsi_atr_interaction"] = df_15["rsi"] * df_15["atr"]
        df_15["macd_diff_atr_interaction"] = df_15["macd_diff"] * df_15["atr"]

        # Ichimoku indicators.
        from ta.trend import IchimokuIndicator
        ichimoku = IchimokuIndicator(high=df_15["high"], low=df_15["low"], window1=9, window2=26, window3=52)
        df_15["tenkan_sen"] = ichimoku.ichimoku_conversion_line()
        df_15["kijun_sen"] = ichimoku.ichimoku_base_line()
        df_15["senkou_span_a"] = ichimoku.ichimoku_a()
        # Removed senkou_span_b

        adx_indicator = ta.trend.ADXIndicator(
            high=df_15["high"], low=df_15["low"], close=df_15["close"], window=14
        )
        df_15["ADX"] = adx_indicator.adx()
        
        # 5-candle horizon target.
        df_15["future_return"] = (df_15["close"].shift(-5) / df_15["close"]) - 1
        conditions = [
            (df_15["future_return"] > LABEL_EPSILON),
            (df_15["future_return"] < -LABEL_EPSILON)
        ]
        df_15["signal_target"] = np.select(conditions, [1, 0], default=2)
        df_15.dropna(inplace=True)

        regime_map = {"uptrending": 0, "downtrending": 1, "sideways": 2}
        df_15["trend_target"] = df_15["regime"].map(regime_map) if "regime" in df_15.columns else 2

        if "regime" in df_15.columns:
            regime_dummies = pd.get_dummies(df_15["regime"], prefix="regime")
            df_15 = pd.concat([df_15, regime_dummies], axis=1)
        df_15.dropna(inplace=True)

        logger.info("Signal label distribution", 
                    distribution=collections.Counter(df_15["signal_target"]))
        logger.info("Trend label distribution", 
                    distribution=collections.Counter(df_15["trend_target"]))

        # Ensure every expected feature exists.
        for feature in self.feature_cols:
            if feature not in df_15.columns:
                df_15[feature] = 0

        actual_cols = [c for c in self.feature_cols if c in df_15.columns]
        df_15[actual_cols] = df_15[actual_cols].apply(pd.to_numeric, errors="coerce")
        df_15.dropna(subset=actual_cols, inplace=True)
        before_drop = len(df_15)
        after_drop = len(df_15)
        if after_drop < before_drop:
            logger.info(f"Dropped {before_drop - after_drop} rows due to NaN in features.")

        features = df_15[actual_cols].values
        trend_labels = df_15["trend_target"].astype(np.int32).values
        signal_labels = df_15["signal_target"].astype(np.int32).values

        X_trend, y_trend = self._make_sequences(features, trend_labels, self.lookback)
        X_signal, y_signal = self._make_sequences(features, signal_labels, self.lookback)

        if len(X_trend) < 1 or len(X_signal) < 1:
            logger.warning("Not enough data after sequence creation.")
            return

        # 70/15/15 data split.
        total_samples_trend = len(X_trend)
        train_end_trend = int(total_samples_trend * 0.70)
        val_end_trend = int(total_samples_trend * 0.85)
        X_trend_train = X_trend[:train_end_trend]
        y_trend_train = y_trend[:train_end_trend]
        X_trend_val = X_trend[train_end_trend:val_end_trend]
        y_trend_val = y_trend[train_end_trend:val_end_trend]
        X_trend_test = X_trend[val_end_trend:]
        y_trend_test = y_trend[val_end_trend:]
        logger.info(f"Trend data split: {len(X_trend_train)} train, {len(X_trend_val)} val, {len(X_trend_test)} test samples.")

        total_samples_signal = len(X_signal)
        train_end_signal = int(total_samples_signal * 0.70)
        val_end_signal = int(total_samples_signal * 0.85)
        X_signal_train = X_signal[:train_end_signal]
        y_signal_train = y_signal[:train_end_signal]
        X_signal_val = X_signal[train_end_signal:val_end_signal]
        y_signal_val = y_signal[train_end_signal:val_end_signal]
        X_signal_test = X_signal[val_end_signal:]
        y_signal_test = y_signal[val_end_signal:]
        logger.info(f"Signal data split: {len(X_signal_train)} train, {len(X_signal_val)} val, {len(X_signal_test)} test samples.")

        # Compute class weights.
        unique_trend = np.unique(y_trend_train)
        cw_trend = compute_class_weight('balanced', classes=unique_trend, y=y_trend_train)
        cw_trend_dict = dict(zip(unique_trend, cw_trend))
        weights_trend = np.array([cw_trend_dict[label] for label in y_trend_train])

        unique_signal = np.unique(y_signal_train)
        cw_signal = compute_class_weight('balanced', classes=unique_signal, y=y_signal_train)
        cw_signal_dict = dict(zip(unique_signal, cw_signal))
        weights_signal = np.array([cw_signal_dict[label] for label in y_signal_train])

        # Apply PCA for dimensionality reduction.
        def apply_pca(X):
            flat = X.reshape(-1, X.shape[2])
            flat_pca = self.pca.transform(flat)
            return flat_pca.reshape(X.shape[0], X.shape[1], -1)

        flat_train = X_trend_train.reshape(-1, X_trend_train.shape[2])
        self.pca = PCA(n_components=10)
        flat_train_pca = self.pca.fit_transform(flat_train)
        X_trend_train = flat_train_pca.reshape(X_trend_train.shape[0], X_trend_train.shape[1], -1)
        X_trend_val = apply_pca(X_trend_val)
        X_trend_test = apply_pca(X_trend_test)
        X_signal_train = apply_pca(X_signal_train)
        X_signal_val = apply_pca(X_signal_val)
        X_signal_test = apply_pca(X_signal_test)
        logger.info(f"After PCA, new feature dimension per time step: {X_trend_train.shape[2]}")

        y_trend_train_cat = to_categorical(y_trend_train, num_classes=3)
        y_trend_val_cat   = to_categorical(y_trend_val, num_classes=3)
        y_signal_train_cat = to_categorical(y_signal_train, num_classes=3)
        y_signal_val_cat   = to_categorical(y_signal_val, num_classes=3)

        X_trend_train = np.nan_to_num(X_trend_train, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        X_trend_val   = np.nan_to_num(X_trend_val, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        y_trend_train_cat = np.nan_to_num(y_trend_train_cat, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        y_trend_val_cat   = np.nan_to_num(y_trend_val_cat, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        X_signal_train = np.nan_to_num(X_signal_train, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        X_signal_val   = np.nan_to_num(X_signal_val, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        y_signal_train_cat = np.nan_to_num(y_signal_train_cat, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        y_signal_val_cat   = np.nan_to_num(y_signal_val_cat, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        logger.info(f"X_trend_train.shape={X_trend_train.shape}, dtype={X_trend_train.dtype}")
        logger.info(f"X_signal_train.shape={X_signal_train.shape}, dtype={X_signal_train.dtype}")

        if X_trend_train.shape[0] == 0 or X_signal_train.shape[0] == 0:
            logger.warning("Empty training sets after final cleaning.")
            return

        input_shape = (self.lookback, X_trend_train.shape[2])
        if self.trend_model is None:
            self.trend_model = self._build_trend_model(input_shape, num_classes=3)
        if self.signal_model is None:
            self.signal_model = self._build_signal_model(input_shape, num_classes=3)

        es_trend = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
        es_signal = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

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

        test_loss_trend, test_acc_trend = self.trend_model.evaluate(X_trend_test, to_categorical(y_trend_test, num_classes=3), verbose=0)
        test_loss_signal, test_acc_signal = self.signal_model.evaluate(X_signal_test, to_categorical(y_signal_test, num_classes=3), verbose=0)
        logger.info("Evaluation on Test Set",
                    trend_loss=test_loss_trend, trend_accuracy=test_acc_trend,
                    signal_loss=test_loss_signal, signal_accuracy=test_acc_signal)

    def _build_trend_model(self, input_shape, num_classes=3):
        inputs = Input(shape=input_shape)
        x = LSTM(64, return_sequences=True)(inputs)
        x = Dropout(0.2)(x)
        x = LSTM(32, return_sequences=True)(x)
        x = Dropout(0.2)(x)
        x = AttentionLayer()(x)
        outputs = Dense(num_classes, activation="softmax", kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        return model

    def _build_signal_model(self, input_shape, num_classes=3):
        inputs = Input(shape=input_shape)
        # Add a 1D convolutional layer before the LSTM layers.
        x = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(inputs)
        x = LSTM(64, return_sequences=True)(x)
        x = Dropout(0.2)(x)
        x = LSTM(32, return_sequences=True)(x)
        x = Dropout(0.2)(x)
        x = AttentionLayer()(x)
        outputs = Dense(num_classes, activation="softmax", kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
        model = Model(inputs=inputs, outputs=outputs)
        # Compile with focal loss for improved handling of class imbalance.
        model.compile(loss=focal_loss(gamma=2., alpha=0.25), optimizer="adam", metrics=["accuracy"])
        return model

    def _make_sequences(self, features, labels, lookback):
        X, y = [], []
        for i in range(len(features) - lookback):
            X.append(features[i : i + lookback])
            y.append(labels[i + lookback])
        return np.array(X), np.array(y)

    def predict_signal(self, recent_data: pd.DataFrame) -> str:
        if not self.trend_model_ready or not self.signal_model_ready:
            logger.warning("ML models not ready; returning 'Hold'.")
            return "Hold"

        data_seq = self._prepare_data_sequence(recent_data)
        if data_seq is None:
            return "Hold"

        trend_pred = self.trend_model.predict(data_seq)
        trend_class = np.argmax(trend_pred, axis=1)[0]
        signal_pred = self.signal_model.predict(data_seq)
        signal_class = np.argmax(signal_pred, axis=1)[0]

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

    def _prepare_data_sequence(self, df: pd.DataFrame) -> np.ndarray:
        data = df.copy()
        data = data.asfreq('1min')
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

        df_15["returns"] = df_15["close"].pct_change()
        df_15["rsi"] = ta.momentum.rsi(df_15["close"], window=14)
        macd = ta.trend.MACD(df_15["close"], window_slow=26, window_fast=12, window_sign=9)
        df_15["macd"] = macd.macd()
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
            high=df_15["high"],
            low=df_15["low"],
            close=df_15["close"],
            volume=df_15["volume"],
            window=14
        ).money_flow_index()

        df_15["stoch"] = ta.momentum.StochasticOscillator(
            high=df_15["high"], low=df_15["low"], close=df_15["close"],
            window=14, smooth_window=3
        ).stoch()
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

        df_15["rsi_lag1"] = df_15["rsi"].shift(1)
        df_15["returns_rolling_mean"] = df_15["returns"].rolling(window=3).mean()
        df_15["returns_rolling_std"] = df_15["returns"].rolling(window=3).std()

        df_15["cmf"] = ta.volume.ChaikinMoneyFlowIndicator(
            high=df_15["high"],
            low=df_15["low"],
            close=df_15["close"],
            volume=df_15["volume"],
            window=20
        ).chaikin_money_flow()
        try:
            from ta.trend import DMIIndicator
            dmi = DMIIndicator(high=df_15["high"], low=df_15["low"], close=df_15["close"], window=14)
            df_15["plus_di"] = dmi.plus_di()
            df_15["minus_di"] = dmi.minus_di()
            df_15["dmi_diff"] = df_15["plus_di"] - df_15["minus_di"]
        except (AttributeError, ImportError):
            df_15["dmi_diff"] = 0

        from ta.trend import IchimokuIndicator
        ichimoku = IchimokuIndicator(high=df_15["high"], low=df_15["low"], window1=9, window2=26, window3=52)
        df_15["tenkan_sen"] = ichimoku.ichimoku_conversion_line()
        df_15["kijun_sen"] = ichimoku.ichimoku_base_line()
        df_15["senkou_span_a"] = ichimoku.ichimoku_a()

        adx_indicator = ta.trend.ADXIndicator(
            high=df_15["high"], low=df_15["low"], close=df_15["close"], window=14
        )
        df_15["ADX"] = adx_indicator.adx()

        df_15.dropna(inplace=True)

        for feature in self.feature_cols:
            if feature not in df_15.columns:
                df_15[feature] = 0

        actual_cols = [c for c in self.feature_cols if c in df_15.columns]
        df_15[actual_cols] = df_15[actual_cols].apply(pd.to_numeric, errors="coerce")
        df_15.dropna(subset=actual_cols, inplace=True)

        recent_slice = df_15.tail(self.lookback).copy()
        if len(recent_slice) < self.lookback:
            missing = self.lookback - len(recent_slice)
            pad_df = pd.DataFrame([recent_slice.iloc[0].values]*missing, columns=recent_slice.columns)
            recent_slice = pd.concat([pad_df, recent_slice], ignore_index=True)

        seq = recent_slice[actual_cols].values
        seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        if self.pca is not None:
            flat_seq = seq.reshape(-1, seq.shape[1])
            flat_seq = self.pca.transform(flat_seq)
            seq = flat_seq.reshape(seq.shape[0], -1)
        seq = np.expand_dims(seq, axis=0)
        return seq
