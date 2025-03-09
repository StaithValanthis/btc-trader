# File: app/services/ml_service.py

import asyncio
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from datetime import datetime, timezone
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (LSTM, Dense, Dropout, Input, Layer, Conv1D,
                                     Bidirectional, Add, BatchNormalization)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
import ta
from structlog import get_logger
import collections
from sklearn.decomposition import PCA
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import RobustScaler

# For automated hyperparameter tuning
try:
    import keras_tuner as kt
except ImportError:
    raise ImportError("Please install keras-tuner via: pip install keras-tuner")

# For optional SHAP-based feature importance analysis
try:
    import shap
except ImportError:
    shap = None
    print("SHAP not installed; feature importance analysis will be skipped.")

from app.services.backfill_service import backfill_bybit_kline
from app.core.database import Database
from app.core.config import Config

logger = get_logger(__name__)

MULTI_TASK_MODEL_PATH = os.path.join("model_storage", "multi_task_model.keras")
DEDICATED_SIGNAL_MODEL_PATH = os.path.join("model_storage", "dedicated_signal_model.keras")
MIN_TRAINING_ROWS = 2000
LABEL_EPSILON = float(os.getenv("LABEL_EPSILON", "0.0005"))

# ---------------------
# Custom Attention Layer
# ---------------------
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.W = self.add_weight(
            name="att_weight",
            shape=(input_shape[-1], 1),
            initializer="normal"
        )
        self.b = self.add_weight(
            name="att_bias",
            shape=(input_shape[1], 1),
            initializer="zeros"
        )
        super(AttentionLayer, self).build(input_shape)
    
    def call(self, x):
        e = tf.nn.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
        a = tf.nn.softmax(e, axis=1)
        output = tf.reduce_sum(x * a, axis=1)
        return output

# ---------------------
# Focal Loss (for class imbalance)
# ---------------------
def focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = 1e-8
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * tf.pow((1 - y_pred), gamma)
        loss = weight * cross_entropy
        return tf.reduce_mean(tf.reduce_sum(loss, axis=1))
    return focal_loss_fixed

# ---------------------
# Utility: Drop Highly Correlated Features
# ---------------------
def drop_highly_correlated_features(df, threshold=0.95):
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    return df.drop(columns=to_drop), to_drop

# ---------------------
# MLService Class
# ---------------------
class MLService:
    """
    MLService integrates advanced feature engineering and trains two models:
      - A multi-task model for both trend and signal prediction.
      - A dedicated signal model for improved signal prediction.
    
    The feature pipeline includes multi-timeframe resampling (5min, 15min, 60min, Daily),
    additional technical indicators (e.g., SMA20, SMA50, their difference, ATR change, volume change),
    robust scaling with correlation-based feature selection, and smoothed signal targets.
    """
    def __init__(
        self,
        lookback=120,
        signal_horizon=5,
        focal_gamma=2.0,
        focal_alpha=0.25,
        multi_task=True,
        batch_size=32
    ):
        self.lookback = lookback
        self.signal_horizon = signal_horizon
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        self.multi_task = multi_task
        self.batch_size = batch_size

        # Models: multi-task and dedicated signal.
        self.multi_task_model = None
        self.signal_model = None  # Dedicated signal model

        self.pca = None
        self.initialized = False
        self.running = True
        self.epochs = 20

        self.trend_model_ready = False
        self.signal_model_ready = False

        # Base feature set plus additional features.
        self.feature_cols = [
            "close", "returns", "rsi", "macd", "macd_diff",
            "bb_high", "bb_low", "bb_mavg", "atr",
            "mfi", "stoch", "obv", "vwap", "ema_20", "cci",
            "bb_width", "volatility_ratio", "vol_change",
            "rsi_lag1", "returns_rolling_mean", "returns_rolling_std",
            "cmf", "dmi_diff",
            "rsi_atr_interaction", "macd_diff_atr_interaction",
            "rsi_5m", "macd_5m", "atr_5m", "obv_5m",
            "sma_60", "sma_daily", "sma20", "sma50", "sma_diff",
            "atr_change"
        ]
        # This will store the final set of features after correlation-based selection.
        self.actual_feature_cols = None

    async def initialize(self):
        try:
            if self.multi_task:
                if os.path.exists(MULTI_TASK_MODEL_PATH):
                    logger.info("Found existing multi-task model on disk.")
                else:
                    logger.info("No existing multi-task model found; will create one on first training.")
            if os.path.exists(DEDICATED_SIGNAL_MODEL_PATH):
                logger.info("Found existing dedicated signal model on disk.")
            else:
                logger.info("No existing dedicated signal model found; will create one on first training.")
            self.initialized = True
        except Exception as e:
            logger.error("Could not initialize MLService", error=str(e))
            self.initialized = False

    async def schedule_daily_retrain(self):
        while self.running:
            await self.train_model()  # Multi-task training
            await self.train_dedicated_signal_model()  # Dedicated signal training
            await asyncio.sleep(86400)

    async def stop(self):
        self.running = False
        logger.info("MLService stopped")

    # ---------------------
    # Helper: Create Sequences
    # ---------------------
    def _make_sequences(self, features, labels, lookback):
        X, y = [], []
        for i in range(len(features) - lookback):
            X.append(features[i : i + lookback])
            y.append(labels[i + lookback])
        return np.array(X), np.array(y)

    # ---------------------
    # Multi-task Model Training
    # ---------------------
    async def train_model(self):
        if not self.initialized:
            logger.warning("MLService not initialized; cannot train multi-task model.")
            return

        logger.info("Training multi-task model with enhanced feature engineering.")
        query = """
            SELECT time, open, high, low, close, volume
            FROM candles
            ORDER BY time ASC
        """
        rows = await Database.fetch(query)
        if not rows:
            logger.warning("No candle data found for multi-task training.")
            return

        df = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close", "volume"])
        df["time"] = pd.to_datetime(df["time"])
        df.set_index("time", inplace=True)
        df = df.asfreq("1min")

        # Resample to 15-min (primary timeframe)
        df_15 = df.resample("15min").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum"
        }).ffill().dropna()

        # Multi-timeframe resamples: 5min, 60min, Daily
        df_5 = df.resample("5min").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum"
        }).ffill().dropna()
        df_60 = df.resample("60min").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum"
        }).ffill().dropna()
        df_daily = df.resample("D").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum"
        }).ffill().dropna()

        df_5_aligned = df_5.reindex(df_15.index, method="ffill")
        df_60_aligned = df_60.reindex(df_15.index, method="ffill")
        df_daily_aligned = df_daily.reindex(df_15.index, method="ffill")

        # Compute 15-min technical features
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
            high=df_15["high"], low=df_15["low"], close=df_15["close"], window=14)
        df_15["atr"] = atr_indicator.average_true_range()
        df_15["mfi"] = ta.volume.MFIIndicator(
            high=df_15["high"], low=df_15["low"],
            close=df_15["close"], volume=df_15["volume"], window=14).money_flow_index()
        stoch = ta.momentum.StochasticOscillator(
            high=df_15["high"], low=df_15["low"],
            close=df_15["close"], window=14, smooth_window=3)
        df_15["stoch"] = stoch.stoch()
        df_15["obv"] = ta.volume.OnBalanceVolumeIndicator(
            close=df_15["close"], volume=df_15["volume"]).on_balance_volume()
        df_15["vwap"] = ta.volume.VolumeWeightedAveragePrice(
            high=df_15["high"], low=df_15["low"],
            close=df_15["close"], volume=df_15["volume"]).volume_weighted_average_price()
        df_15["ema_20"] = ta.trend.EMAIndicator(
            close=df_15["close"], window=20).ema_indicator()
        df_15["cci"] = ta.trend.CCIIndicator(
            high=df_15["high"], low=df_15["low"],
            close=df_15["close"], window=20).cci()
        df_15["bb_width"] = df_15["bb_high"] - df_15["bb_low"]
        df_15["volatility_ratio"] = (df_15["high"] - df_15["low"]) / df_15["close"]
        df_15["rsi_lag1"] = df_15["rsi"].shift(1)
        df_15["returns_rolling_mean"] = df_15["returns"].rolling(window=3).mean()
        df_15["returns_rolling_std"] = df_15["returns"].rolling(window=3).std()
        df_15["cmf"] = ta.volume.ChaikinMoneyFlowIndicator(
            high=df_15["high"], low=df_15["low"],
            close=df_15["close"], volume=df_15["volume"], window=20).chaikin_money_flow()
        try:
            from ta.trend import DMIIndicator
            dmi = DMIIndicator(high=df_15["high"], low=df_15["low"],
                               close=df_15["close"], window=14)
            df_15["plus_di"] = dmi.plus_di()
            df_15["minus_di"] = dmi.minus_di()
            df_15["dmi_diff"] = df_15["plus_di"] - df_15["minus_di"]
        except (AttributeError, ImportError):
            df_15["dmi_diff"] = 0
        df_15["rsi_atr_interaction"] = df_15["rsi"] * df_15["atr"]
        df_15["macd_diff_atr_interaction"] = df_15["macd_diff"] * df_15["atr"]

        from ta.trend import IchimokuIndicator
        ichimoku = IchimokuIndicator(high=df_15["high"], low=df_15["low"],
                                     window1=9, window2=26, window3=52)
        df_15["tenkan_sen"] = ichimoku.ichimoku_conversion_line()
        df_15["kijun_sen"] = ichimoku.ichimoku_base_line()
        df_15["senkou_span_a"] = ichimoku.ichimoku_a()
        adx_indicator = ta.trend.ADXIndicator(high=df_15["high"], low=df_15["low"],
                                              close=df_15["close"], window=14)
        df_15["ADX"] = adx_indicator.adx()

        # Multi-timeframe features
        df_15["rsi_5m"] = ta.momentum.rsi(df_5_aligned["close"], window=14)
        macd_5m = ta.trend.MACD(close=df_5_aligned["close"],
                                window_slow=26, window_fast=12, window_sign=9)
        df_15["macd_5m"] = macd_5m.macd()
        atr_indicator_5m = ta.volatility.AverageTrueRange(
            high=df_5_aligned["high"], low=df_5_aligned["low"],
            close=df_5_aligned["close"], window=14)
        df_15["atr_5m"] = atr_indicator_5m.average_true_range()
        df_15["obv_5m"] = ta.volume.OnBalanceVolumeIndicator(
            close=df_5_aligned["close"], volume=df_5_aligned["volume"]).on_balance_volume()

        # Additional multi-timeframe SMA features:
        df_60_aligned["sma_60"] = df_60_aligned["close"].rolling(window=3, min_periods=1).mean()
        df_daily_aligned["sma_daily"] = df_daily_aligned["close"].rolling(window=3, min_periods=1).mean()
        df_15["sma_60"] = df_60_aligned["sma_60"]
        df_15["sma_daily"] = df_daily_aligned["sma_daily"]
        df_15["sma20"] = df_15["close"].rolling(window=20, min_periods=1).mean()
        df_15["sma50"] = df_15["close"].rolling(window=50, min_periods=1).mean()
        df_15["sma_diff"] = df_15["sma20"] - df_15["sma50"]
        df_15["atr_change"] = df_15["atr"].pct_change()
        df_15["vol_change"] = df_15["volume"].pct_change()

        df_15.dropna(inplace=True)

        # Use stored feature set from training.
        actual_cols = self.actual_feature_cols if self.actual_feature_cols is not None else self.feature_cols
        # For any expected column missing in inference, add it as zero.
        for col in self.feature_cols:
            if col in actual_cols and col not in df_15.columns:
                df_15[col] = 0.0
        # Reorder columns as per stored feature list.
        cols_to_use = [col for col in actual_cols if col in df_15.columns]
        if not cols_to_use:
            logger.warning("None of the expected features are present in inference data.")
            return None

        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
        df_15[cols_to_use] = scaler.fit_transform(df_15[cols_to_use])
        recent_slice = df_15.tail(self.lookback).copy()
        if len(recent_slice) < self.lookback:
            missing = self.lookback - len(recent_slice)
            pad_df = pd.DataFrame([recent_slice.iloc[0].values] * missing, columns=recent_slice.columns)
            recent_slice = pd.concat([pad_df, recent_slice], ignore_index=True)
        seq = recent_slice[cols_to_use].values
        seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        if self.pca is not None:
            flat_seq = seq.reshape(-1, seq.shape[1])
            flat_seq = self.pca.transform(flat_seq)
            seq = flat_seq.reshape(seq.shape[0], -1)
        seq = np.expand_dims(seq, axis=0)
        return seq

    # ---------------------
    # Multi-task Model Architecture
    # ---------------------
    def _build_multi_task_model(self, input_shape, num_classes=3):
        inputs = Input(shape=input_shape)
        conv = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(inputs)
        conv = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(conv)
        shortcut = Dense(32)(inputs)
        shared = Add()([conv, shortcut])
        x = Bidirectional(LSTM(64, return_sequences=True))(shared)
        x = Dropout(0.2)(x)
        x = Bidirectional(LSTM(32, return_sequences=True))(x)
        x = Dropout(0.2)(x)
        shared_rep = AttentionLayer()(x)
        
        # Trend branch
        trend_branch = Dense(32, activation="relu")(shared_rep)
        trend_output = Dense(num_classes, activation="softmax", name="trend",
                             kernel_regularizer=tf.keras.regularizers.l2(0.001))(trend_branch)
        
        # Enhanced Signal branch
        signal_branch = Dense(128, activation="relu")(shared_rep)
        signal_branch = BatchNormalization()(signal_branch)
        signal_branch = Dropout(0.4)(signal_branch)
        signal_branch = Dense(64, activation="relu")(signal_branch)
        signal_branch = BatchNormalization()(signal_branch)
        signal_branch = Dropout(0.3)(signal_branch)
        signal_branch = Dense(32, activation="relu")(signal_branch)
        signal_branch = BatchNormalization()(signal_branch)
        signal_output = Dense(num_classes, activation="softmax", name="signal",
                              kernel_regularizer=tf.keras.regularizers.l2(0.001))(signal_branch)
        
        model = Model(inputs=inputs, outputs=[trend_output, signal_output])
        optimizer = tf.keras.optimizers.experimental.AdamW(learning_rate=1e-3, weight_decay=1e-4)
        model.compile(
            loss={
                "trend": "categorical_crossentropy",
                "signal": focal_loss(gamma=self.focal_gamma, alpha=self.focal_alpha)
            },
            optimizer=optimizer,
            metrics=["accuracy"],
            loss_weights={"trend": 1.0, "signal": 3.0}
        )
        return model

    # ---------------------
    # Dedicated Signal Model Architecture
    # ---------------------
    def _build_dedicated_signal_model(self, input_shape, num_classes=3):
        inputs = Input(shape=input_shape)
        x = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(inputs)
        x = Dropout(0.2)(x)
        x = Bidirectional(LSTM(64, return_sequences=True))(x)
        x = Dropout(0.2)(x)
        x = Bidirectional(LSTM(32, return_sequences=True))(x)
        x = AttentionLayer()(x)
        x = Dense(128, activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)
        x = Dense(64, activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = Dense(32, activation="relu")(x)
        x = BatchNormalization()(x)
        outputs = Dense(num_classes, activation="softmax",
                        kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
        model = Model(inputs=inputs, outputs=outputs)
        optimizer = tf.keras.optimizers.experimental.AdamW(learning_rate=1e-3, weight_decay=1e-4)
        model.compile(loss=focal_loss(gamma=self.focal_gamma, alpha=self.focal_alpha),
                      optimizer=optimizer,
                      metrics=["accuracy"])
        return model

    # ---------------------
    # Dedicated Signal Model Training
    # ---------------------
    async def train_dedicated_signal_model(self):
        if not self.initialized:
            logger.warning("MLService not initialized; cannot train dedicated signal model.")
            return

        logger.info("Training dedicated signal model.")
        query = """
            SELECT time, open, high, low, close, volume
            FROM candles
            ORDER BY time ASC
        """
        rows = await Database.fetch(query)
        if not rows:
            logger.warning("No candle data found for dedicated signal training.")
            return

        df = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close", "volume"])
        df["time"] = pd.to_datetime(df["time"])
        df.set_index("time", inplace=True)
        df = df.asfreq("1min")
        df_15 = df.resample("15min").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum"
        }).ffill().dropna()

        df_5 = df.resample("5min").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum"
        }).ffill().dropna()
        df_60 = df.resample("60min").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum"
        }).ffill().dropna()
        df_daily = df.resample("D").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum"
        }).ffill().dropna()

        df_5_aligned = df_5.reindex(df_15.index, method="ffill")
        df_60_aligned = df_60.reindex(df_15.index, method="ffill")
        df_daily_aligned = df_daily.reindex(df_15.index, method="ffill")

        # Compute features (same pipeline as multi-task)
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
            high=df_15["high"], low=df_15["low"], close=df_15["close"], window=14)
        df_15["atr"] = atr_indicator.average_true_range()
        df_15["mfi"] = ta.volume.MFIIndicator(
            high=df_15["high"], low=df_15["low"],
            close=df_15["close"], volume=df_15["volume"], window=14).money_flow_index()
        stoch = ta.momentum.StochasticOscillator(
            high=df_15["high"], low=df_15["low"], close=df_15["close"],
            window=14, smooth_window=3)
        df_15["stoch"] = stoch.stoch()
        df_15["obv"] = ta.volume.OnBalanceVolumeIndicator(
            close=df_15["close"], volume=df_15["volume"]).on_balance_volume()
        df_15["vwap"] = ta.volume.VolumeWeightedAveragePrice(
            high=df_15["high"], low=df_15["low"],
            close=df_15["close"], volume=df_15["volume"]).volume_weighted_average_price()
        df_15["ema_20"] = ta.trend.EMAIndicator(
            close=df_15["close"], window=20).ema_indicator()
        df_15["cci"] = ta.trend.CCIIndicator(
            high=df_15["high"], low=df_15["low"],
            close=df_15["close"], window=20).cci()
        df_15["bb_width"] = df_15["bb_high"] - df_15["bb_low"]
        df_15["volatility_ratio"] = (df_15["high"] - df_15["low"]) / df_15["close"]
        df_15["rsi_lag1"] = df_15["rsi"].shift(1)
        df_15["returns_rolling_mean"] = df_15["returns"].rolling(window=3).mean()
        df_15["returns_rolling_std"] = df_15["returns"].rolling(window=3).std()
        df_15["cmf"] = ta.volume.ChaikinMoneyFlowIndicator(
            high=df_15["high"], low=df_15["low"],
            close=df_15["close"], volume=df_15["volume"], window=20).chaikin_money_flow()
        try:
            from ta.trend import DMIIndicator
            dmi = DMIIndicator(high=df_15["high"], low=df_15["low"],
                               close=df_15["close"], window=14)
            df_15["plus_di"] = dmi.plus_di()
            df_15["minus_di"] = dmi.minus_di()
            df_15["dmi_diff"] = df_15["plus_di"] - df_15["minus_di"]
        except (AttributeError, ImportError):
            df_15["dmi_diff"] = 0
        df_15["rsi_atr_interaction"] = df_15["rsi"] * df_15["atr"]
        df_15["macd_diff_atr_interaction"] = df_15["macd_diff"] * df_15["atr"]

        from ta.trend import IchimokuIndicator
        ichimoku = IchimokuIndicator(high=df_15["high"], low=df_15["low"],
                                     window1=9, window2=26, window3=52)
        df_15["tenkan_sen"] = ichimoku.ichimoku_conversion_line()
        df_15["kijun_sen"] = ichimoku.ichimoku_base_line()
        df_15["senkou_span_a"] = ichimoku.ichimoku_a()
        adx_indicator = ta.trend.ADXIndicator(
            high=df_15["high"], low=df_15["low"],
            close=df_15["close"], window=14)
        df_15["ADX"] = adx_indicator.adx()

        df_15["rsi_5m"] = ta.momentum.rsi(df_5_aligned["close"], window=14)
        macd_5m = ta.trend.MACD(close=df_5_aligned["close"],
                                window_slow=26, window_fast=12, window_sign=9)
        df_15["macd_5m"] = macd_5m.macd()
        atr_indicator_5m = ta.volatility.AverageTrueRange(
            high=df_5_aligned["high"], low=df_5_aligned["low"],
            close=df_5_aligned["close"], window=14)
        df_15["atr_5m"] = atr_indicator_5m.average_true_range()
        df_15["obv_5m"] = ta.volume.OnBalanceVolumeIndicator(
            close=df_5_aligned["close"], volume=df_5_aligned["volume"]).on_balance_volume()

        df_60_aligned["sma_60"] = df_60_aligned["close"].rolling(window=3, min_periods=1).mean()
        df_daily_aligned["sma_daily"] = df_daily_aligned["close"].rolling(window=3, min_periods=1).mean()
        df_15["sma_60"] = df_60_aligned["sma_60"]
        df_15["sma_daily"] = df_daily_aligned["sma_daily"]
        df_15["sma20"] = df_15["close"].rolling(window=20, min_periods=1).mean()
        df_15["sma50"] = df_15["close"].rolling(window=50, min_periods=1).mean()
        df_15["sma_diff"] = df_15["sma20"] - df_15["sma50"]
        df_15["atr_change"] = df_15["atr"].pct_change()
        df_15["vol_change"] = df_15["volume"].pct_change()

        df_15.dropna(inplace=True)

        features_df, dropped_features = drop_highly_correlated_features(df_15[self.feature_cols], threshold=0.95)
        logger.info("Dropped highly correlated features (dedicated signal)", dropped_features=dropped_features)
        if self.actual_feature_cols is None:
            self.actual_feature_cols = features_df.columns.tolist()
        actual_cols = self.actual_feature_cols

        scaler = RobustScaler()
        features_scaled = scaler.fit_transform(features_df)
        df_15_scaled = pd.DataFrame(features_scaled, index=df_15.index, columns=actual_cols)
        df_15.update(df_15_scaled)

        df_15["future_return"] = (df_15["close"].shift(-self.signal_horizon) / df_15["close"]) - 1
        df_15["future_return_smooth"] = df_15["future_return"].rolling(window=3, min_periods=1).mean()
        conditions = [
            (df_15["future_return_smooth"] > LABEL_EPSILON),
            (df_15["future_return_smooth"] < -LABEL_EPSILON)
        ]
        df_15["signal_target"] = np.select(conditions, [1, 0], default=2)
        df_15.dropna(inplace=True)

        logger.info("Dedicated Signal label distribution", distribution=collections.Counter(df_15["signal_target"]))

        features = df_15[self.actual_feature_cols].values
        signal_labels = df_15["signal_target"].astype(np.int32).values
        X_signal, y_signal = self._make_sequences(features, signal_labels, self.lookback)
        if len(X_signal) < 1:
            logger.warning("Not enough data after sequence creation for dedicated signal model.")
            return

        total_samples_signal = len(X_signal)
        train_end_signal = int(total_samples_signal * 0.70)
        val_end_signal = int(total_samples_signal * 0.85)
        X_signal_train = X_signal[:train_end_signal]
        y_signal_train = y_signal[:train_end_signal]
        X_signal_val = X_signal[train_end_signal:val_end_signal]
        y_signal_val = y_signal[train_end_signal:val_end_signal]

        unique_signal = np.unique(y_signal_train)
        cw_signal = compute_class_weight("balanced", classes=unique_signal, y=y_signal_train)
        cw_signal_dict = dict(zip(unique_signal, cw_signal))
        weights_signal = np.array([cw_signal_dict[label] for label in y_signal_train])

        self.pca = PCA(n_components=10)
        def apply_pca(X):
            flat = X.reshape(-1, X.shape[2])
            flat_pca = self.pca.transform(flat)
            return flat_pca.reshape(X.shape[0], X.shape[1], -1)
        flat_train_signal = X_signal_train.reshape(-1, X_signal_train.shape[2])
        flat_train_signal_pca = self.pca.fit_transform(flat_train_signal)
        X_signal_train = flat_train_signal_pca.reshape(X_signal_train.shape[0], X_signal_train.shape[1], -1)
        X_signal_val = apply_pca(X_signal_val)

        y_signal_train_cat = to_categorical(y_signal_train, num_classes=3)
        y_signal_val_cat = to_categorical(y_signal_val, num_classes=3)

        X_signal_train = np.nan_to_num(X_signal_train, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        X_signal_val = np.nan_to_num(X_signal_val, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        y_signal_train_cat = np.nan_to_num(y_signal_train_cat, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        y_signal_val_cat = np.nan_to_num(y_signal_val_cat, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        input_shape = (self.lookback, X_signal_train.shape[2])
        self.signal_model = self._build_dedicated_signal_model(input_shape, num_classes=3)
        logger.info("Dedicated Signal Model Summary:")
        self.signal_model.summary(print_fn=logger.info)
        history = self.signal_model.fit(
            X_signal_train, y_signal_train_cat,
            validation_data=(X_signal_val, y_signal_val_cat),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=[EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)],
            verbose=1,
            sample_weight=weights_signal
        )
        self.signal_model.save(DEDICATED_SIGNAL_MODEL_PATH)
        self.signal_model_ready = True
        logger.info("Dedicated signal model training complete.")

    # ---------------------
    # Multi-task Model Architecture
    # ---------------------
    def _build_multi_task_model(self, input_shape, num_classes=3):
        inputs = Input(shape=input_shape)
        conv = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(inputs)
        conv = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(conv)
        shortcut = Dense(32)(inputs)
        shared = Add()([conv, shortcut])
        x = Bidirectional(LSTM(64, return_sequences=True))(shared)
        x = Dropout(0.2)(x)
        x = Bidirectional(LSTM(32, return_sequences=True))(x)
        x = Dropout(0.2)(x)
        shared_rep = AttentionLayer()(x)
        
        # Trend branch
        trend_branch = Dense(32, activation="relu")(shared_rep)
        trend_output = Dense(num_classes, activation="softmax", name="trend",
                             kernel_regularizer=tf.keras.regularizers.l2(0.001))(trend_branch)
        
        # Enhanced Signal branch
        signal_branch = Dense(128, activation="relu")(shared_rep)
        signal_branch = BatchNormalization()(signal_branch)
        signal_branch = Dropout(0.4)(signal_branch)
        signal_branch = Dense(64, activation="relu")(signal_branch)
        signal_branch = BatchNormalization()(signal_branch)
        signal_branch = Dropout(0.3)(signal_branch)
        signal_branch = Dense(32, activation="relu")(signal_branch)
        signal_branch = BatchNormalization()(signal_branch)
        signal_output = Dense(num_classes, activation="softmax", name="signal",
                              kernel_regularizer=tf.keras.regularizers.l2(0.001))(signal_branch)
        
        model = Model(inputs=inputs, outputs=[trend_output, signal_output])
        optimizer = tf.keras.optimizers.experimental.AdamW(learning_rate=1e-3, weight_decay=1e-4)
        model.compile(
            loss={
                "trend": "categorical_crossentropy",
                "signal": focal_loss(gamma=self.focal_gamma, alpha=self.focal_alpha)
            },
            optimizer=optimizer,
            metrics=["accuracy"],
            loss_weights={"trend": 1.0, "signal": 3.0}
        )
        return model

    # ---------------------
    # Dedicated Signal Model Architecture
    # ---------------------
    def _build_dedicated_signal_model(self, input_shape, num_classes=3):
        inputs = Input(shape=input_shape)
        x = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(inputs)
        x = Dropout(0.2)(x)
        x = Bidirectional(LSTM(64, return_sequences=True))(x)
        x = Dropout(0.2)(x)
        x = Bidirectional(LSTM(32, return_sequences=True))(x)
        x = AttentionLayer()(x)
        x = Dense(128, activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)
        x = Dense(64, activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = Dense(32, activation="relu")(x)
        x = BatchNormalization()(x)
        outputs = Dense(num_classes, activation="softmax",
                        kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
        model = Model(inputs=inputs, outputs=outputs)
        optimizer = tf.keras.optimizers.experimental.AdamW(learning_rate=1e-3, weight_decay=1e-4)
        model.compile(loss=focal_loss(gamma=self.focal_gamma, alpha=self.focal_alpha),
                      optimizer=optimizer,
                      metrics=["accuracy"])
        return model

    # ---------------------
    # Prediction Method: Prefer Dedicated Signal Model if available
    # ---------------------
    def predict_signal(self, recent_data: pd.DataFrame) -> str:
        if self.signal_model:
            data_seq = self._prepare_data_sequence(recent_data)
            if data_seq is None:
                return "Hold"
            preds = self.signal_model.predict(data_seq)
            signal_class = np.argmax(preds, axis=1)[0]
            signal_label = {0: "Sell", 1: "Buy", 2: "Hold"}.get(signal_class, "Hold")
            return signal_label
        elif self.multi_task_model:
            data_seq = self._prepare_data_sequence(recent_data)
            if data_seq is None:
                return "Hold"
            preds = self.multi_task_model.predict(data_seq)
            signal_class = np.argmax(preds[1], axis=1)[0]
            signal_label = {0: "Sell", 1: "Buy", 2: "Hold"}.get(signal_class, "Hold")
            return signal_label
        else:
            logger.warning("No model available for prediction; returning 'Hold'.")
            return "Hold"

    # ---------------------
    # Prepare Data Sequence for Inference
    # ---------------------
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
        df_5 = data.resample('5min').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).ffill().dropna()
        df_60 = data.resample('60min').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).ffill().dropna()
        df_daily = data.resample('D').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).ffill().dropna()
        df_5_aligned = df_5.reindex(df_15.index, method='ffill')
        df_60_aligned = df_60.reindex(df_15.index, method='ffill')
        df_daily_aligned = df_daily.reindex(df_15.index, method='ffill')
        if len(df_15) < self.lookback:
            logger.warning("Not enough 15-min bars for inference.")
            return None

        # Compute features as in training
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
            high=df_15["high"], low=df_15["low"], close=df_15["close"], window=14)
        df_15["atr"] = atr_indicator.average_true_range()
        df_15["mfi"] = ta.volume.MFIIndicator(
            high=df_15["high"], low=df_15["low"],
            close=df_15["close"], volume=df_15["volume"], window=14).money_flow_index()
        stoch = ta.momentum.StochasticOscillator(
            high=df_15["high"], low=df_15["low"], close=df_15["close"],
            window=14, smooth_window=3)
        df_15["stoch"] = stoch.stoch()
        df_15["obv"] = ta.volume.OnBalanceVolumeIndicator(
            close=df_15["close"], volume=df_15["volume"]).on_balance_volume()
        df_15["vwap"] = ta.volume.VolumeWeightedAveragePrice(
            high=df_15["high"], low=df_15["low"],
            close=df_15["close"], volume=df_15["volume"]).volume_weighted_average_price()
        df_15["ema_20"] = ta.trend.EMAIndicator(
            close=df_15["close"], window=20).ema_indicator()
        df_15["cci"] = ta.trend.CCIIndicator(
            high=df_15["high"], low=df_15["low"], close=df_15["close"], window=20).cci()
        df_15["bb_width"] = df_15["bb_high"] - df_15["bb_low"]
        df_15["volatility_ratio"] = (df_15["high"] - df_15["low"]) / df_15["close"]
        df_15["rsi_lag1"] = df_15["rsi"].shift(1)
        df_15["returns_rolling_mean"] = df_15["returns"].rolling(window=3).mean()
        df_15["returns_rolling_std"] = df_15["returns"].rolling(window=3).std()
        df_15["cmf"] = ta.volume.ChaikinMoneyFlowIndicator(
            high=df_15["high"], low=df_15["low"],
            close=df_15["close"], volume=df_15["volume"], window=20).chaikin_money_flow()
        try:
            from ta.trend import DMIIndicator
            dmi = DMIIndicator(high=df_15["high"], low=df_15["low"],
                               close=df_15["close"], window=14)
            df_15["plus_di"] = dmi.plus_di()
            df_15["minus_di"] = dmi.minus_di()
            df_15["dmi_diff"] = df_15["plus_di"] - df_15["minus_di"]
        except (AttributeError, ImportError):
            df_15["dmi_diff"] = 0
        df_15["rsi_atr_interaction"] = df_15["rsi"] * df_15["atr"]
        df_15["macd_diff_atr_interaction"] = df_15["macd_diff"] * df_15["atr"]

        from ta.trend import IchimokuIndicator
        ichimoku = IchimokuIndicator(high=df_15["high"], low=df_15["low"],
                                     window1=9, window2=26, window3=52)
        df_15["tenkan_sen"] = ichimoku.ichimoku_conversion_line()
        df_15["kijun_sen"] = ichimoku.ichimoku_base_line()
        df_15["senkou_span_a"] = ichimoku.ichimoku_a()
        adx_indicator = ta.trend.ADXIndicator(
            high=df_15["high"], low=df_15["low"], close=df_15["close"], window=14)
        df_15["ADX"] = adx_indicator.adx()
        df_15["rsi_5m"] = ta.momentum.rsi(df_5_aligned["close"], window=14)
        macd_5m = ta.trend.MACD(close=df_5_aligned["close"],
                                window_slow=26, window_fast=12, window_sign=9)
        df_15["macd_5m"] = macd_5m.macd()
        atr_indicator_5m = ta.volatility.AverageTrueRange(
            high=df_5_aligned["high"], low=df_5_aligned["low"],
            close=df_5_aligned["close"], window=14)
        df_15["atr_5m"] = atr_indicator_5m.average_true_range()
        df_15["obv_5m"] = ta.volume.OnBalanceVolumeIndicator(
            close=df_5_aligned["close"], volume=df_5_aligned["volume"]).on_balance_volume()

        # Use stored feature set from training (or fallback)
        actual_cols = self.actual_feature_cols if self.actual_feature_cols is not None else self.feature_cols
        # Ensure all expected features are present; add missing ones as 0.
        for col in self.feature_cols:
            if col in actual_cols and col not in df_15.columns:
                df_15[col] = 0.0
        cols_to_use = [col for col in actual_cols if col in df_15.columns]
        if not cols_to_use:
            logger.warning("None of the expected features are present in inference data.")
            return None

        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
        df_15[cols_to_use] = scaler.fit_transform(df_15[cols_to_use])
        recent_slice = df_15.tail(self.lookback).copy()
        if len(recent_slice) < self.lookback:
            missing = self.lookback - len(recent_slice)
            pad_df = pd.DataFrame([recent_slice.iloc[0].values] * missing, columns=recent_slice.columns)
            recent_slice = pd.concat([pad_df, recent_slice], ignore_index=True)
        seq = recent_slice[cols_to_use].values
        seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        if self.pca is not None:
            flat_seq = seq.reshape(-1, seq.shape[1])
            flat_seq = self.pca.transform(flat_seq)
            seq = flat_seq.reshape(seq.shape[0], -1)
        seq = np.expand_dims(seq, axis=0)
        return seq

    # ---------------------
    # End of Model Architectures and Training
    # ---------------------

# End of file
