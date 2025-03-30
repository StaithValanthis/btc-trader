# File: app/services/ml_service.py

import asyncio
import os
import gc
import pandas as pd
import numpy as np
import tensorflow as tf
from datetime import datetime, timezone
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (LSTM, Dense, Dropout, Input, Layer, Conv1D,
                                     Bidirectional, BatchNormalization)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
import ta
from structlog import get_logger
import collections
from sklearn.decomposition import PCA
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import confusion_matrix, f1_score

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
# Custom DMI Calculation
# ---------------------
def calculate_dmi(high, low, close, window=14):
    """Calculate DMI (Directional Movement Index) values.
    
    Returns:
        dmi: Difference between plus DI and minus DI.
        plus_di: Plus Directional Indicator.
        minus_di: Minus Directional Indicator.
    """
    up_move = high.diff()
    down_move = low.diff().abs()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    plus_di = 100 * plus_dm.rolling(window=window).sum() / tr.rolling(window=window).sum()
    minus_di = 100 * minus_dm.rolling(window=window).sum() / tr.rolling(window=window).sum()
    dmi = plus_di - minus_di
    return dmi, plus_di, minus_di

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
# Utility Functions
# ---------------------
def drop_highly_correlated_features(df, threshold=0.95):
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    return df.drop(columns=to_drop), to_drop

def train_val_test_split(X, y, train_frac=0.7, val_frac=0.15):
    total = len(X)
    train_end = int(total * train_frac)
    val_end = int(total * (train_frac + val_frac))
    return X[:train_end], y[:train_end], X[train_end:val_end], y[train_end:val_end], X[val_end:], y[val_end:]

def walk_forward_validation(X, y, build_model, num_folds=3):
    fold_size = int(len(X) / num_folds)
    losses = []
    for i in range(num_folds - 1):
        train_end = fold_size * (i + 1)
        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[train_end:train_end + fold_size], y[train_end:train_end + fold_size]
        model = build_model()
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=10,
            batch_size=32,
            callbacks=[EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)],
            verbose=0
        )
        losses.append(history.history["val_loss"][-1])
    return np.mean(losses)

# ---------------------
# Periodic Garbage Collection Task
# ---------------------
async def periodic_gc(interval_seconds: int = 300):
    while True:
        collected = gc.collect()
        logger.info("Garbage collection complete", objects_collected=collected)
        await asyncio.sleep(interval_seconds)

# ---------------------
# MLService Class
# ---------------------
class MLService:
    """
    Trains two separate models:
      - Trend model: Uses composite features from 10‑min and 20‑min resampled data.
      - Signal model: Uses features computed on 5‑min candles.
    
    Retraining is scheduled periodically.
    """
    def __init__(
        self,
        lookback=120,
        signal_horizon=5,
        focal_gamma=2.0,
        focal_alpha=0.25,
        batch_size=32,
        ensemble_size=3,
        use_tuned_trend_model=True,
        use_tuned_signal_model=True
    ):
        self.lookback = lookback
        self.signal_horizon = signal_horizon
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        self.batch_size = batch_size
        self.ensemble_size = ensemble_size

        self.use_tuned_trend_model = use_tuned_trend_model
        self.use_tuned_signal_model = use_tuned_signal_model

        self.trend_model = None
        self.signal_models = []
        self.pca = None
        self.initialized = False
        self.running = True
        self.epochs = 20

        self.trend_model_ready = False
        self.signal_model_ready = False

        self.trend_feature_cols = ["sma_diff", "adx", "dmi_diff"]
        self.signal_feature_cols = ["close", "returns", "rsi", "macd_diff", "obv", "vwap", "mfi", "bb_width", "atr"]
        self.actual_trend_cols = None
        self.actual_signal_cols = None

        self._trend_tuning_data = None
        self._signal_tuning_data = None
        self._signal_input_shape = None

    async def initialize(self):
        try:
            logger.info("Initializing MLService...")
            self.initialized = True
            logger.info("Starting initial training for trend model...")
            if self.use_tuned_trend_model:
                await self.tune_trend_model()
            else:
                await self.train_trend_model()
            logger.info("Starting initial training for signal model...")
            if self.use_tuned_signal_model:
                await self.tune_signal_model()
            else:
                await self.train_signal_ensemble(n_models=self.ensemble_size)
            logger.info("Initial training complete.")
        except Exception as e:
            logger.error("Error during initial training", error=str(e))
            self.initialized = False

    async def schedule_daily_retrain(self):
        while self.running:
            logger.info("Starting retrain cycle (every 4 hours)...")
            if self.use_tuned_trend_model:
                await self.tune_trend_model()
            else:
                await self.train_trend_model()
            if self.use_tuned_signal_model:
                await self.tune_signal_model()
            else:
                await self.train_signal_ensemble(n_models=self.ensemble_size)
            logger.info("Retrain cycle complete. Sleeping for 4 hours...")
            await asyncio.sleep(14400)

    async def stop(self):
        self.running = False
        logger.info("MLService stopped.")

    def _make_sequences(self, features, labels, lookback):
        X, y = [], []
        for i in range(len(features) - lookback):
            X.append(features[i : i + lookback])
            y.append(labels[i + lookback])
        return np.array(X), np.array(y)

    # ---------------------
    # Trend Model Methods
    # ---------------------
    def _build_trend_model(self, input_shape, num_classes=3, dropout_rate=0.2, lstm_units=64):
        inputs = Input(shape=input_shape)
        x = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(inputs)
        x = Dropout(dropout_rate)(x)
        x = Bidirectional(LSTM(lstm_units, return_sequences=False))(x)
        x = Dropout(dropout_rate)(x)
        outputs = Dense(num_classes, activation='softmax')(x)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=tf.keras.optimizers.Adam(),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    async def tune_trend_model(self):
        logger.info("Tuning trend model: fetching data...")
        self._trend_tuning_data = await Database.fetch("""
            SELECT time, open, high, low, close, volume
            FROM candles
            ORDER BY time ASC
        """)
        df = pd.DataFrame(self._trend_tuning_data, columns=["time", "open", "high", "low", "close", "volume"])
        df["time"] = pd.to_datetime(df["time"])
        df.set_index("time", inplace=True)
        df = df.asfreq("1min")

        logger.debug("Trend model: before resampling", total_rows=len(df))
        df_10 = df.resample("10min").agg({
            "open": "first", "high": "max", "low": "min",
            "close": "last", "volume": "sum"
        }).ffill()
        df_20 = df.resample("20min").agg({
            "open": "first", "high": "max", "low": "min",
            "close": "last", "volume": "sum"
        }).ffill()

        df_10.dropna(how='all', inplace=True)
        df_20.dropna(how='all', inplace=True)
        logger.debug("Trend model: after resampling", rows_10=len(df_10), rows_20=len(df_20))

        df_20_aligned = df_20.reindex(df_10.index, method="ffill")
        for temp_df in [df_10, df_20_aligned]:
            temp_df["sma20"] = temp_df["close"].rolling(window=20, min_periods=1).mean()
            temp_df["sma50"] = temp_df["close"].rolling(window=50, min_periods=1).mean()
            temp_df["sma_diff"] = temp_df["sma20"] - temp_df["sma50"]
            temp_df["adx"] = ta.trend.ADXIndicator(
                high=temp_df["high"], low=temp_df["low"],
                close=temp_df["close"], window=14
            ).adx()
            # Use our custom DMI calculation
            try:
                dmi, plus_di, minus_di = calculate_dmi(temp_df["high"], temp_df["low"], temp_df["close"], window=14)
                temp_df["dmi_diff"] = plus_di - minus_di
            except Exception as e:
                logger.warning("Custom DMI calculation failed", error=str(e))
                temp_df["dmi_diff"] = 0.0

        composite = pd.DataFrame(index=df_10.index)
        composite["sma_diff"] = (df_10["sma_diff"] + df_20_aligned["sma_diff"]) / 2
        composite["adx"] = (df_10["adx"] + df_20_aligned["adx"]) / 2
        composite["dmi_diff"] = (df_10["dmi_diff"] + df_20_aligned["dmi_diff"]) / 2
        composite.ffill(inplace=True)
        composite.dropna(inplace=True)
        logger.debug("Trend composite final", shape=composite.shape)

        default_threshold = 0.3
        default_adx_threshold = 20
        composite["trend_target"] = np.where(
            (composite["sma_diff"] > default_threshold) & (composite["adx"] > default_adx_threshold), 0,
            np.where((composite["sma_diff"] < -default_threshold) & (composite["adx"] > default_adx_threshold), 1, 2)
        )

        self.actual_trend_cols = ["sma_diff", "adx", "dmi_diff"]
        scaler = RobustScaler()
        composite[self.actual_trend_cols] = scaler.fit_transform(composite[self.actual_trend_cols])
        logger.debug("Trend composite after scaling", shape=composite.shape)

        X, y = self._make_sequences(composite[self.actual_trend_cols].values, composite["trend_target"].values, self.lookback)
        logger.debug("Trend sequences", X_shape=X.shape, y_shape=y.shape)

        if len(X) < 1:
            logger.warning("Not enough data to build trend sequences after transformations.")
            return

        X_train, y_train, X_val, y_val, _ , _ = train_val_test_split(X, y)
        logger.info("Starting trend model tuning...")
        tuner = kt.RandomSearch(
            self.build_trend_model_tuner,
            objective="val_loss",
            max_trials=5,
            executions_per_trial=1,
            directory="kt_dir",
            project_name="trend_model_tuning"
        )
        await asyncio.to_thread(lambda: tuner.search(
            X_train, to_categorical(y_train, num_classes=3),
            validation_data=(X_val, to_categorical(y_val, num_classes=3)),
            epochs=10,
            batch_size=self.batch_size,
            callbacks=[EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)],
            verbose=0
        ))
        best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
        logger.info("Trend tuning complete. Best hyperparameters found:", best_hp.values)
        best_model = self.build_trend_model_tuner(best_hp)
        if self.use_tuned_trend_model:
            self.trend_model = best_model
            self.trend_model_ready = True
            best_model.save("trend_model.keras")
        return best_model

    def build_trend_model_tuner(self, hp):
        dropout_rate = hp.Float("dropout_rate", 0.1, 0.5, step=0.1, default=0.2)
        lstm_units = hp.Int("lstm_units", 32, 128, step=32, default=64)
        input_shape = (self.lookback, len(self.actual_trend_cols))
        model = self._build_trend_model(input_shape, num_classes=3, dropout_rate=dropout_rate, lstm_units=lstm_units)
        model.fit(
            np.zeros((10, self.lookback, len(self.actual_trend_cols))),
            to_categorical(np.zeros(10), num_classes=3),
            epochs=1,
            batch_size=self.batch_size,
            verbose=0
        )
        return model

    async def train_trend_model(self):
        await self.tune_trend_model()

    # ---------------------
    # Signal Model Methods
    # ---------------------
    def prepare_signal_features(self, df):
        df = df.copy()
        df.index = pd.to_datetime(df.index)
        logger.debug("Signal model: initial raw data shape", shape=df.shape)
        df_5 = df.resample("5min").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum"
        }).ffill()
        df_5.dropna(how='all', inplace=True)
        logger.debug("Signal model: after 5min resample", shape=df_5.shape)
        df_5["returns"] = df_5["close"].pct_change()
        df_5["rsi"] = ta.momentum.rsi(df_5["close"], window=14)
        df_5["macd_diff"] = ta.trend.MACD(df_5["close"], window_slow=26, window_fast=12, window_sign=9).macd_diff()
        df_5["obv"] = ta.volume.OnBalanceVolumeIndicator(close=df_5["close"], volume=df_5["volume"]).on_balance_volume()
        df_5["vwap"] = ta.volume.VolumeWeightedAveragePrice(
            high=df_5["high"], low=df_5["low"], close=df_5["close"], volume=df_5["volume"]
        ).volume_weighted_average_price()
        df_5["mfi"] = ta.volume.MFIIndicator(
            high=df_5["high"], low=df_5["low"], close=df_5["close"], volume=df_5["volume"], window=14
        ).money_flow_index()
        df_5.ffill(inplace=True)
        df_5.dropna(inplace=True)
        logger.debug("Signal model: after fillna", shape=df_5.shape)
        return df_5

    async def tune_signal_model(self):
        logger.info("Tuning signal model: fetching data...")

        async def _fetch_signal_data():
            query = """
                SELECT time, open, high, low, close, volume
                FROM candles
                ORDER BY time ASC
            """
            rows = await Database.fetch(query)
            df = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close", "volume"])
            df["time"] = pd.to_datetime(df["time"])
            df.set_index("time", inplace=True)
            df = df.asfreq("1min")
            return df

        self._signal_tuning_data = await _fetch_signal_data()
        df = self._signal_tuning_data.copy()
        logger.debug("Signal model: raw candle data shape", shape=df.shape)
        df_5 = df.resample("5min").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum"
        }).ffill()
        logger.debug("Signal model: after 5min resample", shape=df_5.shape)
        df_5["returns"] = df_5["close"].pct_change()
        df_5["rsi"] = ta.momentum.rsi(df_5["close"], window=14)
        df_5["macd_diff"] = ta.trend.MACD(df_5["close"], window_slow=26, window_fast=12, window_sign=9).macd_diff()
        boll = ta.volatility.BollingerBands(df_5["close"], window=20, window_dev=2.0)
        df_5["bb_width"] = boll.bollinger_hband() - boll.bollinger_lband()
        atr_indicator = ta.volatility.AverageTrueRange(
            high=df_5["high"], low=df_5["low"], close=df_5["close"], window=14
        )
        df_5["atr"] = atr_indicator.average_true_range()
        df_5["volatility_ratio"] = (df_5["high"] - df_5["low"]) / df_5["close"]
        df_5["cmf"] = ta.volume.ChaikinMoneyFlowIndicator(
            high=df_5["high"], low=df_5["low"], close=df_5["close"],
            volume=df_5["volume"], window=20
        ).chaikin_money_flow()
        df_5.ffill(inplace=True)
        df_5.dropna(inplace=True)
        logger.debug("Signal model: after indicators", shape=df_5.shape)
        df_5["future_return"] = (df_5["close"].shift(-self.signal_horizon) / df_5["close"]) - 1
        df_5["future_return_smooth"] = df_5["future_return"].rolling(window=3, min_periods=1).mean()
        conditions = [
            (df_5["future_return_smooth"] > LABEL_EPSILON),
            (df_5["future_return_smooth"] < -LABEL_EPSILON)
        ]
        df_5["signal_target"] = np.select(conditions, [1, 0], default=2)
        df_5.dropna(inplace=True)
        logger.debug("Signal model: after future_return calc", shape=df_5.shape)
        for col in self.signal_feature_cols:
            if col not in df_5.columns:
                df_5[col] = 0.0

        from sklearn.ensemble import RandomForestClassifier
        features_all = df_5[self.signal_feature_cols].copy()
        scaler = RobustScaler()
        features_scaled = scaler.fit_transform(features_all)
        features_all = pd.DataFrame(features_scaled, index=features_all.index, columns=self.signal_feature_cols)
        signal_labels = df_5["signal_target"].astype(np.int32).values
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(features_all, signal_labels)
        importances = rf.feature_importances_
        median_importance = np.median(importances)
        selected_indices = np.where(importances >= median_importance)[0]
        selected_features = [self.signal_feature_cols[i] for i in selected_indices]
        logger.info("Selected features for signal model:", selected_features=selected_features)
        self.actual_signal_cols = selected_features
        df_5[self.actual_signal_cols] = scaler.fit_transform(df_5[self.actual_signal_cols])
        features = df_5[self.actual_signal_cols].values
        y_labels = df_5["signal_target"].astype(np.int32).values
        X, y = self._make_sequences(features, y_labels, self.lookback)
        logger.debug("Signal sequences", X_shape=X.shape, y_shape=y.shape)
        if len(X) < 1:
            logger.warning("Not enough data after sequence creation for signal ensemble.")
            return
        X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(X, y)
        self._signal_input_shape = (X_train.shape[1], X_train.shape[2])
        logger.info("Starting signal model tuning...")
        tuner = kt.RandomSearch(
            self.build_signal_model_tuner,
            objective="val_loss",
            max_trials=5,
            executions_per_trial=1,
            directory="kt_dir",
            project_name="signal_model_tuning"
        )
        await asyncio.to_thread(lambda: tuner.search(
            X_train, to_categorical(y_train, num_classes=3),
            validation_data=(X_val, to_categorical(y_val, num_classes=3)),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=[EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)],
            verbose=0
        ))
        best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
        logger.info("Signal tuning complete. Best hyperparameters found:", best_hp.values)
        best_model = self.build_signal_model_tuner(best_hp)
        cv_loss = walk_forward_validation(
            X_train,
            to_categorical(y_train, num_classes=3),
            lambda: self.build_signal_model_tuner(best_hp),
            num_folds=3
        )
        logger.info("Signal model CV loss", cv_loss=cv_loss)
        if self.use_tuned_signal_model:
            self.signal_models = [best_model]
            best_model.save("signal_model.keras")
            self.signal_model_ready = True
        return best_model

    def build_signal_model_tuner(self, hp):
        dropout_rate = hp.Float("dropout_rate", 0.1, 0.5, step=0.1, default=0.2)
        lstm_units = hp.Int("lstm_units", 32, 128, step=32, default=64)
        learning_rate = hp.Float("learning_rate", 1e-4, 1e-2, sampling="log", default=1e-3)
        weight_decay = hp.Float("weight_decay", 1e-5, 1e-3, sampling="log", default=1e-4)
        input_shape = self._signal_input_shape
        model = self._build_dedicated_signal_model(input_shape, num_classes=3)
        optimizer = tf.keras.optimizers.experimental.AdamW(
            learning_rate=learning_rate,
            weight_decay=weight_decay
        )
        model.compile(
            loss=focal_loss(gamma=self.focal_gamma, alpha=self.focal_alpha),
            optimizer=optimizer,
            metrics=["accuracy"]
        )
        return model

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
        outputs = Dense(
            num_classes,
            activation="softmax",
            kernel_regularizer=tf.keras.regularizers.l2(0.001)
        )(x)
        model = Model(inputs=inputs, outputs=outputs)
        return model

    async def train_signal_ensemble(self, n_models=3):
        logger.info("Training dedicated signal ensemble for signal prediction.")
        query = """
            SELECT time, open, high, low, close, volume
            FROM candles
            ORDER BY time ASC
        """
        rows = await Database.fetch(query)
        if not rows:
            logger.warning("No candle data found for signal model training.")
            return
        df = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close", "volume"])
        df["time"] = pd.to_datetime(df["time"])
        df.set_index("time", inplace=True)
        df = df.asfreq("1min")
        df_5 = df.resample("5min").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum"
        }).ffill()
        df_5.dropna(how='all', inplace=True)
        df_5["returns"] = df_5["close"].pct_change()
        df_5["rsi"] = ta.momentum.rsi(df_5["close"], window=14)
        df_5["macd_diff"] = ta.trend.MACD(df_5["close"], window_slow=26, window_fast=12, window_sign=9).macd_diff()
        boll = ta.volatility.BollingerBands(df_5["close"], window=20, window_dev=2)
        df_5["bb_width"] = boll.bollinger_hband() - boll.bollinger_lband()
        atr_indicator = ta.volatility.AverageTrueRange(
            high=df_5["high"], low=df_5["low"], close=df_5["close"], window=14
        )
        df_5["atr"] = atr_indicator.average_true_range()
        df_5["volatility_ratio"] = (df_5["high"] - df_5["low"]) / df_5["close"]
        df_5["cmf"] = ta.volume.ChaikinMoneyFlowIndicator(
            high=df_5["high"], low=df_5["low"], close=df_5["close"],
            volume=df_5["volume"], window=20
        ).chaikin_money_flow()
        df_5.ffill(inplace=True)
        df_5.dropna(inplace=True)
        df_5["future_return"] = (df_5["close"].shift(-self.signal_horizon) / df_5["close"]) - 1
        df_5["future_return_smooth"] = df_5["future_return"].rolling(window=3, min_periods=1).mean()
        conditions = [
            (df_5["future_return_smooth"] > LABEL_EPSILON),
            (df_5["future_return_smooth"] < -LABEL_EPSILON)
        ]
        df_5["signal_target"] = np.select(conditions, [1, 0], default=2)
        df_5.dropna(inplace=True)
        logger.info("Signal label distribution", distribution=collections.Counter(df_5["signal_target"]))
        for col in self.signal_feature_cols:
            if col not in df_5.columns:
                df_5[col] = 0.0
        self.actual_signal_cols = self.signal_feature_cols
        scaler = RobustScaler()
        df_5[self.actual_signal_cols] = scaler.fit_transform(df_5[self.actual_signal_cols])
        features = df_5[self.actual_signal_cols].values
        signal_labels = df_5["signal_target"].astype(np.int32).values
        X, y = self._make_sequences(features, signal_labels, self.lookback)
        if len(X) < 1:
            logger.warning("Not enough data after sequence creation for signal ensemble.")
            return
        X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(X, y)
        flat_train = X_train.reshape(-1, X_train.shape[2])
        n_components = min(10, flat_train.shape[0], flat_train.shape[1])
        self.pca = PCA(n_components=n_components)
        flat_train_pca = self.pca.fit_transform(flat_train)
        X_train = flat_train_pca.reshape(X_train.shape[0], X_train.shape[1], -1)
        def apply_pca(X):
            flat = X.reshape(-1, X.shape[2])
            flat_pca = self.pca.transform(flat)
            return flat_pca.reshape(X.shape[0], X.shape[1], -1)
        X_val = apply_pca(X_val)
        X_test = apply_pca(X_test)
        unique = np.unique(y_train)
        cw = compute_class_weight("balanced", classes=unique, y=y_train)
        cw_dict = dict(zip(unique, cw))
        sample_weights = np.array([cw_dict[label] for label in y_train])
        self.signal_models = []
        for i in range(n_models):
            input_shape = (X_train.shape[1], X_train.shape[2])
            model = self._build_dedicated_signal_model(input_shape, num_classes=3)
            logger.info(f"Training signal model {i+1}...")
            model.fit(
                X_train, to_categorical(y_train, num_classes=3),
                validation_data=(X_val, to_categorical(y_val, num_classes=3)),
                epochs=self.epochs,
                batch_size=self.batch_size,
                callbacks=[EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)],
                verbose=1,
                sample_weight=sample_weights
            )
            self.signal_models.append(model)
        preds = [model.predict(X_test) for model in self.signal_models]
        avg_preds = np.mean(np.array(preds), axis=0)
        y_pred_class = np.argmax(avg_preds, axis=1)[0]
        cm = confusion_matrix(y_test, y_pred_class)
        f1 = f1_score(y_test, y_pred_class, average="weighted")
        logger.info("Signal Ensemble Evaluation", confusion_matrix=cm.tolist(), weighted_f1=f1)
        self.signal_model_ready = True
        return self.signal_models

    def _prepare_data_sequence(self, df: pd.DataFrame, feature_list: list, resample_period: str = '5min') -> np.ndarray:
        data = df.copy()
        data = data.asfreq('1min')
        df_resampled = data.resample(resample_period).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).ffill().dropna()
        for col in feature_list:
            if col not in df_resampled.columns:
                df_resampled[col] = 0.0
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
        df_resampled[feature_list] = scaler.fit_transform(df_resampled[feature_list])
        if len(df_resampled) < self.lookback:
            logger.warning("Not enough bars for inference.", current_rows=len(df_resampled))
            return None
        recent_slice = df_resampled.tail(self.lookback).copy()
        if len(recent_slice) < self.lookback:
            missing = self.lookback - len(recent_slice)
            pad_df = pd.DataFrame([recent_slice.iloc[0].values] * missing, columns=recent_slice.columns)
            recent_slice = pd.concat([pad_df, recent_slice], ignore_index=True)
        seq = recent_slice[feature_list].values
        seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        if self.pca is not None:
            flat_seq = seq.reshape(-1, seq.shape[1])
            flat_seq = self.pca.transform(flat_seq)
            seq = flat_seq.reshape(seq.shape[0], -1)
        seq = np.expand_dims(seq, axis=0)
        return seq

    def predict_trend(self, recent_data: pd.DataFrame) -> str:
        if self.ml_service.trend_model:
            df = recent_data.copy().asfreq("1min")
            df_10 = df.resample("10min").agg({
                "open": "first", "high": "max", "low": "min",
                "close": "last", "volume": "sum"
            }).ffill()
            df_20 = df.resample("20min").agg({
                "open": "first", "high": "max", "low": "min",
                "close": "last", "volume": "sum"
            }).ffill()
            if len(df_10) < self.ml_service.lookback or len(df_20) < self.ml_service.lookback:
                logger.warning("Not enough 10min or 20min data for trend inference.")
                return "Hold"
            df_20_aligned = df_20.reindex(df_10.index, method="ffill")
            for temp_df in [df_10, df_20_aligned]:
                temp_df["sma20"] = temp_df["close"].rolling(window=20, min_periods=1).mean()
                temp_df["sma50"] = temp_df["close"].rolling(window=50, min_periods=1).mean()
                temp_df["sma_diff"] = temp_df["sma20"] - temp_df["sma50"]
                temp_df["adx"] = ta.trend.ADXIndicator(
                    high=temp_df["high"], low=temp_df["low"],
                    close=temp_df["close"], window=14
                ).adx()
                # Use custom DMI calculation
                try:
                    dmi, plus_di, minus_di = calculate_dmi(temp_df["high"], temp_df["low"], temp_df["close"], window=14)
                    temp_df["dmi_diff"] = plus_di - minus_di
                except Exception as e:
                    logger.warning("Custom DMI calculation failed in predict_trend", error=str(e))
                    temp_df["dmi_diff"] = 0.0
            composite = pd.DataFrame(index=df_10.index)
            composite["sma_diff"] = (df_10["sma_diff"] + df_20_aligned["sma_diff"]) / 2
            composite["adx"] = (df_10["adx"] + df_20_aligned["adx"]) / 2
            composite["dmi_diff"] = (df_10["dmi_diff"] + df_20_aligned["dmi_diff"]) / 2
            composite.ffill(inplace=True)
            composite.dropna(inplace=True)
            if len(composite) < self.ml_service.lookback:
                logger.warning("Not enough composite rows for trend model lookback.")
                return "Hold"
            last_slice = composite.tail(self.ml_service.lookback).copy()
            from sklearn.preprocessing import RobustScaler
            used_cols = ["sma_diff", "adx", "dmi_diff"]
            scaler = RobustScaler()
            last_slice[used_cols] = scaler.fit_transform(last_slice[used_cols])
            seq = last_slice[used_cols].values
            seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            seq = np.expand_dims(seq, axis=0)
            preds = self.ml_service.trend_model.predict(seq)
            trend_class = np.argmax(preds, axis=1)[0]
            trend_label = {0: "Uptrending", 1: "Downtrending", 2: "Sideways"}.get(trend_class, "Sideways")
            return trend_label
        else:
            logger.warning("No trend model available; returning 'Sideways'.")
            return "Sideways"

    def predict_signal(self, recent_data: pd.DataFrame) -> str:
        if self.ml_service.signal_models and self.ml_service.signal_model_ready:
            data_seq = self._prepare_data_sequence(
                recent_data,
                self.ml_service.actual_signal_cols or self.ml_service.signal_feature_cols,
                resample_period="5min"
            )
            if data_seq is None:
                return "Hold"
            preds = [model.predict(data_seq) for model in self.ml_service.signal_models]
            avg_pred = np.mean(np.array(preds), axis=0)
            signal_class = np.argmax(avg_pred, axis=1)[0]
            signal_label = {0: "Sell", 1: "Buy", 2: "Hold"}.get(signal_class, "Hold")
            return signal_label
        else:
            logger.warning("No signal model ensemble available or not ready; returning 'Hold'.")
            return "Hold"

    async def stop(self):
        self.running = False
        logger.info("TradeService stopped.")


# ---------------------
# Periodic Garbage Collection Task
# ---------------------
async def periodic_gc(interval_seconds: int = 300):
    while True:
        collected = gc.collect()
        logger.info("Garbage collection complete", objects_collected=collected)
        await asyncio.sleep(interval_seconds)

if __name__ == "__main__":
    async def main():
        asyncio.create_task(periodic_gc(300))
        trade_service = TradeService()
        await trade_service.initialize()
        while True:
            await asyncio.sleep(60)
    asyncio.run(main())
