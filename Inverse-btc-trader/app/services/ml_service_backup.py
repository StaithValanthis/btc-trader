# File: app/services/ml_service.py

import asyncio
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from datetime import datetime, timezone
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Layer, Conv1D, Bidirectional, Add
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
import ta
from structlog import get_logger
import collections
from sklearn.decomposition import PCA
from sklearn.utils.class_weight import compute_class_weight

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
        # x has shape [batch_size, time_steps, features]
        e = tf.nn.tanh(tf.tensordot(x, self.W, axes=1) + self.b)  # [batch_size, time_steps, 1]
        a = tf.nn.softmax(e, axis=1)  # [batch_size, time_steps, 1]
        output = tf.reduce_sum(x * a, axis=1)  # [batch_size, features]
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
# MLService class
# ---------------------
class MLService:
    """
    Multi-task learning for Trend and Signal prediction.

    - Configurable signal target generation (via signal_horizon)
    - Advanced feature engineering (including lag/rolling features,
      multiple technical indicators, interaction terms, volatility_ratio)
    - PCA for dimensionality reduction
    - Focal loss & computed class weights for class imbalance
    - Multi-task model that shares a residual Conv1D block & bidirectional LSTMs
    - Automated hyperparameter tuning via KerasTuner
    - Optional SHAP-based feature importance
    - Async train_model so it can be scheduled (schedule_daily_retrain)
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

        # If multi_task is True, we'll build one combined model
        self.multi_task_model = None
        # Or separate models
        self.trend_model = None
        self.signal_model = None

        self.pca = None
        self.initialized = False
        self.running = True
        self.epochs = 20

        self.trend_model_ready = False
        self.signal_model_ready = False

        # Full list of expected features
        self.feature_cols = [
            "close", "returns", "rsi", "macd", "macd_diff",
            "bb_high", "bb_low", "bb_mavg", "atr",
            "mfi", "stoch", "obv", "vwap", "ema_20", "cci",
            "bb_width", "volatility_ratio",
            "tenkan_sen", "kijun_sen", "senkou_span_a", "ADX",
            "rsi_lag1", "returns_rolling_mean", "returns_rolling_std",
            "cmf", "dmi_diff",
            "rsi_atr_interaction", "macd_diff_atr_interaction"
        ]

    # ---------------------------------
    # Async initialization
    # ---------------------------------
    async def initialize(self):
        try:
            if self.multi_task:
                if os.path.exists(MULTI_TASK_MODEL_PATH):
                    logger.info("Found existing multi-task model on disk.")
                else:
                    logger.info("No existing multi-task model found; will create a new one on first training.")
            else:
                if os.path.exists("signal_model.keras"):
                    logger.info("Found existing SignalModel on disk.")
                else:
                    logger.info("No existing SignalModel found; will create new one on first training.")
                if os.path.exists("trend_model.keras"):
                    logger.info("Found existing TrendModel on disk.")
                else:
                    logger.info("No existing TrendModel found; will create new one on first training.")
            self.initialized = True
        except Exception as e:
            logger.error("Could not initialize MLService", error=str(e))
            self.initialized = False

    # ---------------------------------
    # Periodic retraining
    # ---------------------------------
    async def schedule_daily_retrain(self):
        """
        Calls train_model once a day while self.running is True.
        """
        while self.running:
            await self.train_model()
            await asyncio.sleep(86400)

    async def stop(self):
        self.running = False
        logger.info("MLService stopped")

    # ---------------------------------
    # The main async training method
    # ---------------------------------
    async def train_model(self):
        """
        The main asynchronous training function,
        invoked by schedule_daily_retrain or once at startup.
        """
        if not self.initialized:
            logger.warning("MLService not initialized; cannot train yet.")
            return

        logger.info("Retraining multi-task / separate models using advanced feature engineering (15-min data).")

        query = """
            SELECT time, open, high, low, close, volume
            FROM candles
            ORDER BY time ASC
        """
        rows = await Database.fetch(query)
        if not rows:
            logger.warning("No candle data found for training.")
            return

        df = pd.DataFrame(rows, columns=["time","open","high","low","close","volume"])
        df["time"] = pd.to_datetime(df["time"])
        df.set_index("time", inplace=True)
        df = df.asfreq("1min")

        # Resample to 15-min
        df_15 = df.resample("15min").agg({
            "open":"first",
            "high":"max",
            "low":"min",
            "close":"last",
            "volume":"sum"
        }).ffill().dropna()

        if len(df_15) < (MIN_TRAINING_ROWS // 15):
            logger.warning(f"Not enough 15-min data for training; got {len(df_15)} rows.")
            return

        # Basic feature engineering
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
        df_15["ema_20"] = ta.trend.EMAIndicator(
            close=df_15["close"], window=20
        ).ema_indicator()
        df_15["cci"] = ta.trend.CCIIndicator(
            high=df_15["high"], low=df_15["low"], close=df_15["close"], window=20
        ).cci()
        df_15["bb_width"] = df_15["bb_high"] - df_15["bb_low"]
        df_15["volatility_ratio"] = (df_15["high"] - df_15["low"]) / df_15["close"]
        df_15["rsi_lag1"] = df_15["rsi"].shift(1)
        df_15["returns_rolling_mean"] = df_15["returns"].rolling(window=3).mean()
        df_15["returns_rolling_std"] = df_15["returns"].rolling(window=3).std()

        # Additional indicators
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

        # Interaction terms
        df_15["rsi_atr_interaction"] = df_15["rsi"] * df_15["atr"]
        df_15["macd_diff_atr_interaction"] = df_15["macd_diff"] * df_15["atr"]

        # Ichimoku
        from ta.trend import IchimokuIndicator
        ichimoku = IchimokuIndicator(
            high=df_15["high"], low=df_15["low"], window1=9, window2=26, window3=52
        )
        df_15["tenkan_sen"] = ichimoku.ichimoku_conversion_line()
        df_15["kijun_sen"] = ichimoku.ichimoku_base_line()
        df_15["senkou_span_a"] = ichimoku.ichimoku_a()
        adx_indicator = ta.trend.ADXIndicator(
            high=df_15["high"], low=df_15["low"], close=df_15["close"], window=14
        )
        df_15["ADX"] = adx_indicator.adx()

        # Label generation for signal
        df_15["future_return"] = (df_15["close"].shift(-self.signal_horizon) / df_15["close"]) - 1
        conditions = [
            (df_15["future_return"] > LABEL_EPSILON),
            (df_15["future_return"] < -LABEL_EPSILON)
        ]
        df_15["signal_target"] = np.select(conditions, [1,0], default=2)
        df_15.dropna(inplace=True)

        # If "regime" in df_15, define a trend_target
        if "regime" in df_15.columns:
            regime_map = {"uptrending":0, "downtrending":1, "sideways":2}
            df_15["trend_target"] = df_15["regime"].map(regime_map)
        else:
            df_15["trend_target"] = 2
        df_15.dropna(inplace=True)

        logger.info("Signal label distribution", distribution=collections.Counter(df_15["signal_target"]))
        logger.info("Trend label distribution", distribution=collections.Counter(df_15["trend_target"]))

        for feature in self.feature_cols:
            if feature not in df_15.columns:
                df_15[feature] = 0

        actual_cols = [c for c in self.feature_cols if c in df_15.columns]
        df_15[actual_cols] = df_15[actual_cols].apply(pd.to_numeric, errors="coerce")
        df_15.dropna(subset=actual_cols, inplace=True)

        features = df_15[actual_cols].values
        trend_labels = df_15["trend_target"].astype(np.int32).values
        signal_labels = df_15["signal_target"].astype(np.int32).values

        # Create sequences
        X_trend, y_trend = self._make_sequences(features, trend_labels, self.lookback)
        X_signal, y_signal = self._make_sequences(features, signal_labels, self.lookback)

        if len(X_trend) < 1 or len(X_signal) < 1:
            logger.warning("Not enough data after sequence creation.")
            return

        # 70/15/15 splits
        total_samples_trend = len(X_trend)
        train_end_trend = int(total_samples_trend * 0.70)
        val_end_trend   = int(total_samples_trend * 0.85)

        X_trend_train = X_trend[:train_end_trend]
        y_trend_train = y_trend[:train_end_trend]
        X_trend_val   = X_trend[train_end_trend:val_end_trend]
        y_trend_val   = y_trend[train_end_trend:val_end_trend]
        X_trend_test  = X_trend[val_end_trend:]
        y_trend_test  = y_trend[val_end_trend:]

        total_samples_signal = len(X_signal)
        train_end_signal = int(total_samples_signal * 0.70)
        val_end_signal   = int(total_samples_signal * 0.85)

        X_signal_train = X_signal[:train_end_signal]
        y_signal_train = y_signal[:train_end_signal]
        X_signal_val   = X_signal[train_end_signal:val_end_signal]
        y_signal_val   = y_signal[train_end_signal:val_end_signal]
        X_signal_test  = X_signal[val_end_signal:]
        y_signal_test  = y_signal[val_end_signal:]

        # Class weights
        unique_trend = np.unique(y_trend_train)
        cw_trend = compute_class_weight("balanced", classes=unique_trend, y=y_trend_train)
        cw_trend_dict = dict(zip(unique_trend, cw_trend))
        weights_trend = np.array([cw_trend_dict[label] for label in y_trend_train])

        unique_signal = np.unique(y_signal_train)
        cw_signal = compute_class_weight("balanced", classes=unique_signal, y=y_signal_train)
        cw_signal_dict = dict(zip(unique_signal, cw_signal))
        weights_signal = np.array([cw_signal_dict[label] for label in y_signal_train])

        # PCA
        self.pca = PCA(n_components=10)
        def apply_pca(X):
            flat = X.reshape(-1, X.shape[2])
            flat_pca = self.pca.transform(flat)
            return flat_pca.reshape(X.shape[0], X.shape[1], -1)

        # Fit PCA on the trend training set
        flat_train_trend = X_trend_train.reshape(-1, X_trend_train.shape[2])
        flat_train_trend_pca = self.pca.fit_transform(flat_train_trend)
        X_trend_train = flat_train_trend_pca.reshape(X_trend_train.shape[0], X_trend_train.shape[1], -1)

        X_trend_val = apply_pca(X_trend_val)
        X_trend_test = apply_pca(X_trend_test)
        X_signal_train = apply_pca(X_signal_train)
        X_signal_val = apply_pca(X_signal_val)
        X_signal_test = apply_pca(X_signal_test)

        # to_categorical
        y_trend_train_cat = to_categorical(y_trend_train, num_classes=3)
        y_trend_val_cat   = to_categorical(y_trend_val,   num_classes=3)
        y_signal_train_cat= to_categorical(y_signal_train,num_classes=3)
        y_signal_val_cat  = to_categorical(y_signal_val,  num_classes=3)

        # Replace NaNs with 0
        X_trend_train = np.nan_to_num(X_trend_train, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        X_trend_val   = np.nan_to_num(X_trend_val,   nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        y_trend_train_cat = np.nan_to_num(y_trend_train_cat, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        y_trend_val_cat   = np.nan_to_num(y_trend_val_cat,   nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        X_signal_train = np.nan_to_num(X_signal_train, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        X_signal_val   = np.nan_to_num(X_signal_val,   nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        y_signal_train_cat = np.nan_to_num(y_signal_train_cat, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        y_signal_val_cat   = np.nan_to_num(y_signal_val_cat,   nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        logger.info(f"X_trend_train.shape={X_trend_train.shape}, dtype={X_trend_train.dtype}")
        logger.info(f"X_signal_train.shape={X_signal_train.shape}, dtype={X_signal_train.dtype}")

        # Build or reuse the multi-task model
        input_shape = (self.lookback, X_trend_train.shape[2])
        if self.multi_task:
            multi_task_model = self._build_multi_task_model(input_shape, num_classes=3)
            history = multi_task_model.fit(
                X_trend_train, {"trend": y_trend_train_cat, "signal": y_signal_train_cat},
                validation_data=(X_trend_val, {"trend": y_trend_val_cat, "signal": y_signal_val_cat}),
                epochs=self.epochs,
                batch_size=self.batch_size,
                callbacks=[EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)],
                verbose=1,
                sample_weight={"trend": weights_trend, "signal": weights_signal}
            )
            multi_task_model.save(MULTI_TASK_MODEL_PATH)
            self.multi_task_model = multi_task_model
            self.trend_model_ready  = True
            self.signal_model_ready = True
        else:
            # Build or reuse separate models
            if self.trend_model is None:
                self.trend_model = self._build_trend_model(input_shape, num_classes=3)
            if self.signal_model is None:
                self.signal_model = self._build_signal_model(input_shape, num_classes=3)

            history_trend = self.trend_model.fit(
                X_trend_train, y_trend_train_cat,
                validation_data=(X_trend_val, y_trend_val_cat),
                epochs=self.epochs,
                batch_size=self.batch_size,
                callbacks=[EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)],
                verbose=1,
                sample_weight=weights_trend
            )
            self.trend_model.save("trend_model.keras")
            self.trend_model_ready = True

            history_signal = self.signal_model.fit(
                X_signal_train, y_signal_train_cat,
                validation_data=(X_signal_val, y_signal_val_cat),
                epochs=self.epochs,
                batch_size=self.batch_size,
                callbacks=[EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)],
                verbose=1,
                sample_weight=weights_signal
            )
            self.signal_model.save("signal_model.keras")
            self.signal_model_ready = True

        logger.info("Training complete.")

    # ---------------------------------
    # Build multi-task model
    # ---------------------------------
    def _build_multi_task_model(self, input_shape, num_classes=3):
        """
        Builds a default multi-task model with a residual Conv1D block, 
        bidirectional LSTMs, an attention layer, then 2 output heads 
        (trend & signal).
        """
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

        trend_output = Dense(num_classes, activation="softmax", name="trend",
                             kernel_regularizer=tf.keras.regularizers.l2(0.001))(shared_rep)
        signal_output = Dense(num_classes, activation="softmax", name="signal",
                              kernel_regularizer=tf.keras.regularizers.l2(0.001))(shared_rep)

        model = Model(inputs=inputs, outputs=[trend_output, signal_output])
        optimizer = tf.keras.optimizers.experimental.AdamW(learning_rate=1e-3, weight_decay=1e-4)
        model.compile(
            loss={
                "trend": "categorical_crossentropy",
                "signal": focal_loss(gamma=self.focal_gamma, alpha=self.focal_alpha)
            },
            optimizer=optimizer,
            metrics=["accuracy"],
            loss_weights={"trend": 1.0, "signal": 1.0}
        )
        return model

    def _build_multi_task_model_tunable(self, hp):
        """
        This is used by the `tune_model` method to auto-build 
        a multi-task model with hyperparameters from keras_tuner.
        """
        inputs = Input(shape=(self.lookback, len(self.feature_cols)))
        conv_filters = hp.Int("conv_filters", min_value=16, max_value=64, step=16, default=32)
        kernel_size = hp.Choice("kernel_size", values=[3,5], default=3)
        conv = Conv1D(filters=conv_filters, kernel_size=kernel_size, activation='relu', padding='same')(inputs)
        conv = Conv1D(filters=conv_filters, kernel_size=kernel_size, activation='relu', padding='same')(conv)
        shortcut = Dense(conv_filters)(inputs)
        shared = Add()([conv, shortcut])
        lstm_units1 = hp.Int("lstm_units1", min_value=32, max_value=128, step=32, default=64)
        lstm_units2 = hp.Int("lstm_units2", min_value=16, max_value=64, step=16, default=32)
        x = Bidirectional(LSTM(lstm_units1, return_sequences=True))(shared)
        dropout_rate = hp.Float("dropout_rate", 0.1, 0.5, step=0.1, default=0.2)
        x = Dropout(dropout_rate)(x)
        x = Bidirectional(LSTM(lstm_units2, return_sequences=True))(x)
        x = Dropout(dropout_rate)(x)
        shared_rep = AttentionLayer()(x)
        trend_output = Dense(3, activation="softmax", name="trend",
                             kernel_regularizer=tf.keras.regularizers.l2(0.001))(shared_rep)
        signal_output = Dense(3, activation="softmax", name="signal",
                              kernel_regularizer=tf.keras.regularizers.l2(0.001))(shared_rep)

        model = Model(inputs=inputs, outputs=[trend_output, signal_output])

        learning_rate = hp.Float("learning_rate", 1e-4, 1e-2, sampling="log", default=1e-3)
        weight_decay = hp.Float("weight_decay", 1e-5, 1e-3, sampling="log", default=1e-4)
        optimizer = tf.keras.optimizers.experimental.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
        focal_gamma = hp.Float("focal_gamma", 1.5, 3.0, step=0.5, default=self.focal_gamma)
        focal_alpha = hp.Float("focal_alpha", 0.15, 0.35, step=0.05, default=self.focal_alpha)

        model.compile(
            loss={
                "trend": "categorical_crossentropy",
                "signal": focal_loss(gamma=focal_gamma, alpha=focal_alpha)
            },
            optimizer=optimizer,
            metrics=["accuracy"],
            loss_weights={"trend": 1.0, "signal": 1.0}
        )
        return model

    def tune_model(self, X_train, Y_train, X_val, Y_val,
                   weights_trend, weights_signal,
                   max_trials=10, executions_per_trial=1):
        """
        Automatic hyperparameter search for the multi-task model
        using KerasTuner's RandomSearch.
        """
        tuner = kt.RandomSearch(
            self._build_multi_task_model_tunable,
            objective="val_loss",
            max_trials=max_trials,
            executions_per_trial=executions_per_trial,
            directory="kt_dir",
            project_name="ml_service_tuning"
        )
        tuner.search(
            X_train, Y_train,
            validation_data=(X_val, Y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=[EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)],
            sample_weight={"trend": weights_trend, "signal": weights_signal}
        )
        best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
        best_model = tuner.hypermodel.build(best_hp)
        best_model.fit(
            X_train, Y_train,
            validation_data=(X_val, Y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=[EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)],
            sample_weight={"trend": weights_trend, "signal": weights_signal}
        )
        best_model.save(MULTI_TASK_MODEL_PATH)
        self.multi_task_model = best_model
        self.trend_model_ready  = True
        self.signal_model_ready = True
        return tuner.results_summary()

    def analyze_feature_importance(self, X_sample):
        """
        If shap is installed, performs KernelExplainer analysis
        on the multi-task model's shared representation.
        """
        if shap is None:
            logger.warning("SHAP not installed; skipping feature importance.")
            return
        if not self.multi_task_model:
            logger.warning("Multi-task model not built; skipping SHAP analysis.")
            return
        # Create an intermediate model for the shared representation
        intermediate_model = Model(
            inputs=self.multi_task_model.input,
            outputs=self.multi_task_model.get_layer(index=-3).output
        )
        explainer = shap.KernelExplainer(intermediate_model.predict, X_sample[:10])
        shap_values = explainer.shap_values(X_sample[:10])
        # shap_values is a list [trend, signal], each shape is different
        # we focus on signal shap for example
        avg_shap = np.mean(np.abs(shap_values[1]), axis=(0,1))
        feature_importance = dict(zip(self.feature_cols, avg_shap))
        logger.info("SHAP Feature Importance for signal head", **feature_importance)
        return feature_importance

    def predict_signal(self, recent_data: pd.DataFrame) -> str:
        """
        Predict next move as [Buy, Sell, or Hold], optionally constrained by the trend
        from the multi-task model (or separate models).
        """
        if self.multi_task:
            if not self.multi_task_model:
                logger.warning("Multi-task model not ready; returning 'Hold'.")
                return "Hold"
            data_seq = self._prepare_data_sequence(recent_data)
            if data_seq is None:
                return "Hold"
            preds = self.multi_task_model.predict(data_seq)
            trend_pred, signal_pred = preds
        else:
            if not self.trend_model or not self.signal_model:
                logger.warning("ML models not ready; returning 'Hold'.")
                return "Hold"
            data_seq = self._prepare_data_sequence(recent_data)
            if data_seq is None:
                return "Hold"
            trend_pred  = self.trend_model.predict(data_seq)
            signal_pred = self.signal_model.predict(data_seq)

        trend_class  = np.argmax(trend_pred,  axis=1)[0]
        signal_class = np.argmax(signal_pred, axis=1)[0]

        trend_label = {0:"uptrending", 1:"downtrending", 2:"sideways"}.get(trend_class, "sideways")
        signal_label= {0:"Sell", 1:"Buy", 2:"Hold"}.get(signal_class, "Hold")

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
        """
        Prepare data for predict_signal. Takes ~180 lines of code in train_model,
        condensed for inference context only.
        """
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
        macd = ta.trend.MACD(close=df_15["close"], window_slow=26, window_fast=12, window_sign=9)
        df_15["macd"] = macd.macd()
        df_15["macd_diff"] = macd.macd_diff()

        boll = ta.volatility.BollingerBands(df_15["close"], window=20, window_dev=2)
        df_15["bb_high"] = boll.bollinger_hband()
        df_15["bb_low"]  = boll.bollinger_lband()
        df_15["bb_mavg"] = boll.bollinger_mavg()

        atr_indicator = ta.volatility.AverageTrueRange(
            high=df_15["high"], low=df_15["low"], close=df_15["close"], window=14
        )
        df_15["atr"] = atr_indicator.average_true_range()

        df_15["mfi"] = ta.volume.MFIIndicator(
            high=df_15["high"], low=df_15["low"], close=df_15["close"],
            volume=df_15["volume"], window=14
        ).money_flow_index()

        stoch = ta.momentum.StochasticOscillator(
            high=df_15["high"], low=df_15["low"], close=df_15["close"],
            window=14, smooth_window=3
        )
        df_15["stoch"] = stoch.stoch()
        df_15["obv"]   = ta.volume.OnBalanceVolumeIndicator(
            close=df_15["close"], volume=df_15["volume"]
        ).on_balance_volume()

        df_15["vwap"]  = ta.volume.VolumeWeightedAveragePrice(
            high=df_15["high"], low=df_15["low"], close=df_15["close"], volume=df_15["volume"]
        ).volume_weighted_average_price()
        df_15["ema_20"] = ta.trend.EMAIndicator(df_15["close"], window=20).ema_indicator()
        df_15["cci"]    = ta.trend.CCIIndicator(
            high=df_15["high"], low=df_15["low"], close=df_15["close"], window=20
        ).cci()
        df_15["bb_width"] = df_15["bb_high"] - df_15["bb_low"]

        df_15["rsi_lag1"] = df_15["rsi"].shift(1)
        df_15["returns_rolling_mean"] = df_15["returns"].rolling(window=3).mean()
        df_15["returns_rolling_std"]  = df_15["returns"].rolling(window=3).std()

        df_15["cmf"] = ta.volume.ChaikinMoneyFlowIndicator(
            high=df_15["high"],
            low=df_15["low"],
            close=df_15["close"],
            volume=df_15["volume"],
            window=20
        ).chaikin_money_flow()

        try:
            from ta.trend import DMIIndicator
            dmi = DMIIndicator(
                high=df_15["high"], low=df_15["low"], close=df_15["close"], window=14
            )
            df_15["plus_di"] = dmi.plus_di()
            df_15["minus_di"]= dmi.minus_di()
            df_15["dmi_diff"]= df_15["plus_di"] - df_15["minus_di"]
        except (AttributeError, ImportError):
            df_15["dmi_diff"] = 0

        from ta.trend import IchimokuIndicator
        ichimoku = IchimokuIndicator(
            high=df_15["high"], low=df_15["low"], window1=9, window2=26, window3=52
        )
        df_15["tenkan_sen"] = ichimoku.ichimoku_conversion_line()
        df_15["kijun_sen"]  = ichimoku.ichimoku_base_line()
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

        # Must have at least `lookback` rows
        if len(df_15) < self.lookback:
            logger.warning("Not enough 15-min bars for inference.")
            return None

        recent_slice = df_15.tail(self.lookback).copy()
        # If still less than lookback, pad
        if len(recent_slice) < self.lookback:
            missing = self.lookback - len(recent_slice)
            pad_df = pd.DataFrame(
                [recent_slice.iloc[0].values] * missing,
                columns=recent_slice.columns
            )
            recent_slice = pd.concat([pad_df, recent_slice], ignore_index=True)

        seq = recent_slice[actual_cols].values
        seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        if self.pca is not None:
            flat_seq = seq.reshape(-1, seq.shape[1])
            flat_seq = self.pca.transform(flat_seq)
            seq = flat_seq.reshape(seq.shape[0], -1)

        seq = np.expand_dims(seq, axis=0)
        return seq

    def _make_sequences(self, features, labels, lookback):
        """
        Create sequences of length `lookback` from the features
        and produce aligned labels for the final time step.
        """
        X, y = [], []
        for i in range(len(features) - lookback):
            X.append(features[i : i + lookback])
            y.append(labels[i + lookback])
        return np.array(X), np.array(y)

    def train_and_evaluate(self):
        asyncio.run(self.train_model())