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

try:
    import keras_tuner as kt
except ImportError:
    raise ImportError("Please install keras-tuner via: pip install keras-tuner")

try:
    import shap
except ImportError:
    shap = None
    print("SHAP not installed; feature importance analysis will be skipped.")

from app.services.backfill_service import backfill_bybit_kline
from app.core.database import Database
from app.core.config import Config

logger = get_logger(__name__)

MIN_TRAINING_ROWS = 2000
LABEL_EPSILON = float(os.getenv("LABEL_EPSILON", "0.0005"))

# Confidence thresholds used in trade_service as well
SIGNAL_PROB_THRESHOLD = 0.6
TREND_PROB_THRESHOLD = 0.6

def exponential_smooth(series, alpha=0.3):
    return series.ewm(alpha=alpha).mean()

def median_filter(series, window=5):
    return series.rolling(window=window, center=True).median().bfill().ffill()

def add_lag_features(df, columns, lags=[1, 2, 3]):
    df_new = df.copy()
    for col in columns:
        for lag in lags:
            df_new[f"{col}_lag{lag}"] = df_new[col].shift(lag)
    df_new.dropna(inplace=True)
    return df_new

def auto_select_pca_components(X, variance_threshold=0.95):
    pca = PCA()
    pca.fit(X)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.searchsorted(cumulative_variance, variance_threshold) + 1
    n_components = max(n_components, 1)
    pca = PCA(n_components=n_components)
    return pca

def calculate_dmi(high, low, close, window=14):
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

class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        if input_shape[-1] <= 0 or input_shape[1] <= 0:
            raise ValueError(f"Invalid input shape for AttentionLayer: {input_shape}")
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

def focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = 1e-8
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * tf.pow((1 - y_pred), gamma)
        loss = weight * cross_entropy
        return tf.reduce_mean(tf.reduce_sum(loss, axis=1))
    return focal_loss_fixed

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

async def periodic_gc(interval_seconds: int = 300):
    while True:
        collected = gc.collect()
        logger.info("Garbage collection complete", objects_collected=collected)
        await asyncio.sleep(interval_seconds)

###############################################################################
# NEW: Data Repair Function
###############################################################################
def repair_candle_row(row):
    """
    Repairs a single candle row if the 'close' value is invalid.
    - If close <= 0, replace with previous valid close.
    - Clamp close between low and high.
    """
    # If close is non-positive, set it to the average of low and high (or use previous valid value)
    if row["close"] <= 0:
        row["close"] = (row["low"] + row["high"]) / 2.0
    # Clamp close to be between low and high
    row["close"] = max(row["low"], min(row["close"], row["high"]))
    return row

###############################################################################
# MLService Class
###############################################################################
class MLService:
    """
    Trains an ensemble of trend models using 4h data and a signal model ensemble using 15min data.
    The redesigned data pipeline cleans and repairs candle data so that technical indicators,
    including ATR, are computed on reliable data.
    """
    def __init__(
        self,
        lookback=60,
        signal_horizon=5,
        focal_gamma=2.0,
        focal_alpha=0.25,
        batch_size=32,
        ensemble_size=3,
        n_trend_models=2,
        use_tuned_trend_model=True,
        use_tuned_signal_model=True
    ):
        self.lookback = lookback
        self.signal_horizon = signal_horizon
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        self.batch_size = batch_size
        self.ensemble_size = ensemble_size
        self.n_trend_models = n_trend_models

        self.use_tuned_trend_model = use_tuned_trend_model
        self.use_tuned_signal_model = use_tuned_signal_model

        self.trend_models = []
        self.signal_models = []
        self.pca = None
        self.initialized = False
        self.running = True
        self.epochs = 20

        self.trend_model_ready = False
        self.signal_model_ready = False

        # For trend, 4h data
        self.trend_feature_cols = ["sma_diff", "adx", "dmi_diff", "close_lag1"]
        # For signals, 15min data
        self.signal_feature_cols = ["close", "returns", "rsi", "macd_diff",
                                    "obv", "vwap", "mfi", "bb_width", "atr", "close_lag1"]
        self.actual_trend_cols = None
        self.actual_signal_cols = None

        # Expected input shapes for inference
        self.trend_input_shape = None  # (lookback, number_of_trend_features)
        self.signal_input_shape = None  # (lookback, number_of_signal_features)

    def preprocess_dataframe(self, df):
        df = df.copy()
        # Convert key columns to float
        for col in ["open", "high", "low", "close", "volume"]:
            try:
                df[col] = df[col].astype(float)
            except Exception as e:
                logger.error("Error converting column %s to float: %s", col, str(e))
        logger.debug("Preprocessing: Stats before reindexing - open(min=%.2f, max=%.2f), high(min=%.2f, max=%.2f), low(min=%.2f, max=%.2f), close(min=%.2f, max=%.2f)",
                     df["open"].min(), df["open"].max(),
                     df["high"].min(), df["high"].max(),
                     df["low"].min(), df["low"].max(),
                     df["close"].min(), df["close"].max())
        # Reindex DataFrame to strict 1-minute frequency
        start = df.index.min()
        end = df.index.max()
        expected_minutes = int(((end - start).total_seconds() / 60)) + 1
        df = df.reindex(pd.date_range(start=start, end=end, freq="1min"))
        df.sort_index(inplace=True)
        df.ffill(inplace=True)
        logger.debug("After reindexing: rows=%d", len(df))
        for col in ["close", "volume"]:
            df[col] = exponential_smooth(df[col])
            df[col] = median_filter(df[col], window=5)
        df = add_lag_features(df, ["close"], lags=[1])
        # Repair rows that violate data integrity
        repaired_df = df.apply(repair_candle_row, axis=1)
        # Remove any remaining rows with non-positive close
        valid_df = repaired_df[repaired_df["close"] > 0]
        dropped = len(df) - len(valid_df)
        if dropped > 0:
            logger.warning("Dropped %d rows after repair due to non-positive close.", dropped)
        if len(valid_df) < 0.8 * expected_minutes:
            logger.warning("Preprocessed candle data has fewer rows (%d) than expected (%d)", len(valid_df), expected_minutes)
        return valid_df

    async def initialize(self):
        try:
            logger.info("Initializing MLService...")
            self.initialized = True
            logger.info("Starting initial training for trend models...")
            if self.use_tuned_trend_model:
                await self.tune_trend_models()
            else:
                await self.train_trend_models()
            logger.info("Starting initial training for signal models...")
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
                await self.tune_trend_models()
            else:
                await self.train_trend_models()
            if self.use_tuned_signal_model:
                await self.tune_signal_model()
            else:
                await self.train_signal_ensemble(n_models=self.ensemble_size)
            logger.info("Retrain cycle complete. Sleeping for 4 hours...")
            await asyncio.sleep(14400)

    async def tune_trend_models(self):
        logger.info("Tuning trend models: fetching data...")
        self._trend_tuning_data = await Database.fetch("""
            SELECT time, open, high, low, close, volume
            FROM candles
            ORDER BY time ASC
        """)
        df = pd.DataFrame(self._trend_tuning_data, columns=["time", "open", "high", "low", "close", "volume"])
        df["time"] = pd.to_datetime(df["time"])
        df.set_index("time", inplace=True)
        df = df.asfreq("1min")
        df = self.preprocess_dataframe(df)
        logger.info("Trend data time span: %s to %s", df.index.min(), df.index.max())
        df_4h = df.resample("4h").agg({
            "open": "first", "high": "max", "low": "min",
            "close": "last", "volume": "sum"
        }).ffill()
        df_4h["sma20"] = df_4h["close"].rolling(window=20, min_periods=1).mean()
        df_4h["sma50"] = df_4h["close"].rolling(window=50, min_periods=1).mean()
        df_4h["sma_diff"] = df_4h["sma20"] - df_4h["sma50"]
        df_4h["adx"] = ta.trend.ADXIndicator(
            high=df_4h["high"], low=df_4h["low"],
            close=df_4h["close"], window=14
        ).adx()
        try:
            dmi, plus_di, minus_di = calculate_dmi(df_4h["high"], df_4h["low"], df_4h["close"], window=14)
            df_4h["dmi_diff"] = plus_di - minus_di
        except Exception as e:
            logger.warning("Custom DMI calculation failed for trend model", error=str(e))
            df_4h["dmi_diff"] = 0.0
        df_4h["close_lag1"] = df_4h["close"].shift(1)
        df_4h.dropna(inplace=True)
        self.actual_trend_cols = ["sma_diff", "adx", "dmi_diff", "close_lag1"]
        self.trend_input_shape = (self.lookback, len(self.actual_trend_cols))
        scaler = RobustScaler()
        df_4h[self.actual_trend_cols] = scaler.fit_transform(df_4h[self.actual_trend_cols])
        default_threshold = 0.3
        default_adx_threshold = 20
        df_4h["trend_target"] = np.where(
            (df_4h["sma_diff"] > default_threshold) & (df_4h["adx"] > default_adx_threshold), 0,
            np.where((df_4h["sma_diff"] < -default_threshold) & (df_4h["adx"] > default_adx_threshold), 1, 2)
        )
        X, y = self._make_sequences(df_4h[self.actual_trend_cols].values, df_4h["trend_target"].values, self.lookback)
        if len(X) < 1:
            logger.warning("Not enough data to build trend sequences after transformations.")
            return
        X_train, y_train, X_val, y_val, _ , _ = train_val_test_split(X, y)
        logger.info("Starting trend model ensemble tuning using 4h data...")
        self.trend_models = []
        for i in range(self.n_trend_models):
            tuner = kt.RandomSearch(
                self.build_trend_model_tuner,
                objective="val_loss",
                max_trials=3,
                executions_per_trial=1,
                directory="kt_dir/trend",
                project_name=f"trend_model_tuning_{i}"
            )
            await asyncio.to_thread(lambda: tuner.search(
                X_train, to_categorical(y_train, num_classes=3),
                validation_data=(X_val, to_categorical(y_val, num_classes=3)),
                epochs=5,
                batch_size=self.batch_size,
                callbacks=[EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)],
                verbose=0
            ))
            best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
            model = self.build_trend_model_tuner(best_hp)
            model.fit(X_train, to_categorical(y_train, num_classes=3), epochs=5, batch_size=self.batch_size, verbose=0)
            self.trend_models.append(model)
        self.trend_model_ready = True

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

    async def train_trend_models(self):
        await self.tune_trend_models()

    async def tune_signal_model(self):
        logger.info("Tuning signal model: fetching data...")

        async def _fetch_signal_data():
            query = """
                SELECT time, open, high, low, close, volume
                FROM candles
                ORDER BY time ASC
            """
            rows = await Database.fetch(query)
            df_local = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close", "volume"])
            df_local["time"] = pd.to_datetime(df_local["time"])
            df_local.set_index("time", inplace=True)
            # Reindex to 1-minute frequency
            start = df_local.index.min()
            end = df_local.index.max()
            df_local = df_local.reindex(pd.date_range(start=start, end=end, freq="1min"))
            df_local.sort_index(inplace=True)
            df_local.ffill(inplace=True)
            return df_local

        self._signal_tuning_data = await _fetch_signal_data()
        df = self._signal_tuning_data.copy()
        for col in ["open", "high", "low", "close", "volume"]:
            try:
                df[col] = df[col].astype(float)
            except Exception as e:
                logger.error("Error converting column %s to float: %s", col, str(e))
        for col in ["close", "volume"]:
            df[col] = exponential_smooth(df[col])
            df[col] = median_filter(df[col], window=5)
        df = add_lag_features(df, ["close"], lags=[1])
        df_15 = df.resample("15min").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum"
        }).ffill()
        logger.info("Signal data time span: %s to %s", df_15.index.min(), df_15.index.max())
        df_15["returns"] = df_15["close"].pct_change()
        df_15["rsi"] = ta.momentum.rsi(df_15["close"], window=14)
        df_15["macd_diff"] = ta.trend.MACD(df_15["close"], window_slow=26, window_fast=12, window_sign=9).macd_diff()
        boll = ta.volatility.BollingerBands(df_15["close"], window=20, window_dev=2.0)
        df_15["bb_width"] = boll.bollinger_hband() - boll.bollinger_lband()
        atr_indicator = ta.volatility.AverageTrueRange(
            high=df_15["high"], low=df_15["low"], close=df_15["close"], window=14
        )
        df_15["atr"] = atr_indicator.average_true_range()
        if df_15["atr"].empty or df_15["atr"].iloc[-1] <= 0:
            logger.warning("Computed ATR non-positive in signal data. Applying fallback ATR.")
            fallback_atr = df_15["close"].pct_change().rolling(window=14).std().iloc[-1] * df_15["close"].iloc[-1]
            df_15["atr"] = fallback_atr if fallback_atr > 0 else 0.01 * df_15["close"].iloc[-1]
        df_15["volatility_ratio"] = (df_15["high"] - df_15["low"]) / df_15["close"]
        df_15["cmf"] = ta.volume.ChaikinMoneyFlowIndicator(
            high=df_15["high"], low=df_15["low"], close=df_15["close"],
            volume=df_15["volume"], window=20
        ).chaikin_money_flow()
        df_15.ffill(inplace=True)
        df_15.dropna(inplace=True)
        df_15["future_return"] = (df_15["close"].shift(-self.signal_horizon) / df_15["close"]) - 1
        df_15["future_return_smooth"] = df_15["future_return"].rolling(window=3, min_periods=1).mean()
        conditions = [
            (df_15["future_return_smooth"] > LABEL_EPSILON),
            (df_15["future_return_smooth"] < -LABEL_EPSILON)
        ]
        df_15["signal_target"] = np.select(conditions, [1, 0], default=2)
        df_15.dropna(inplace=True)

        all_needed_cols = list(set(self.signal_feature_cols + ["signal_target"]))
        df_15 = df_15.reindex(columns=df_15.columns.union(all_needed_cols), fill_value=0.0)
        logger.debug("Signal model DataFrame columns after union reindex", columns=df_15.columns.tolist())
        signal_labels = df_15["signal_target"].astype(np.int32).values
        for col in self.signal_feature_cols:
            if col not in df_15.columns:
                df_15[col] = 0.0
        self.actual_signal_cols = self.signal_feature_cols
        input_shape = (self.lookback, len(self.actual_signal_cols))
        self.signal_input_shape = input_shape
        scaler = RobustScaler()
        df_15[self.actual_signal_cols] = scaler.fit_transform(df_15[self.actual_signal_cols])
        features = df_15[self.actual_signal_cols].values
        X, y = self._make_sequences(features, signal_labels, self.lookback)
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
            directory="kt_dir/signal",
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
        return self.signal_models

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
        for col in ["close", "volume"]:
            df[col] = exponential_smooth(df[col])
            df[col] = median_filter(df[col], window=5)
        df = add_lag_features(df, ["close"], lags=[1])
        df_15 = df.resample("15min").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum"
        }).ffill()
        df_15["returns"] = df_15["close"].pct_change()
        df_15["rsi"] = ta.momentum.rsi(df_15["close"], window=14)
        df_15["macd_diff"] = ta.trend.MACD(df_15["close"], window_slow=26, window_fast=12, window_sign=9).macd_diff()
        boll = ta.volatility.BollingerBands(df_15["close"], window=20, window_dev=2)
        df_15["bb_width"] = boll.bollinger_hband() - boll.bollinger_lband()
        atr_indicator = ta.volatility.AverageTrueRange(
            high=df_15["high"], low=df_15["low"], close=df_15["close"], window=14
        )
        df_15["atr"] = atr_indicator.average_true_range()
        df_15["volatility_ratio"] = (df_15["high"] - df_15["low"]) / df_15["close"]
        df_15["cmf"] = ta.volume.ChaikinMoneyFlowIndicator(
            high=df_15["high"], low=df_15["low"], close=df_15["close"],
            volume=df_15["volume"], window=20
        ).chaikin_money_flow()
        df_15.ffill(inplace=True)
        df_15.dropna(inplace=True)
        df_15["future_return"] = (df_15["close"].shift(-self.signal_horizon) / df_15["close"]) - 1
        df_15["future_return_smooth"] = df_15["future_return"].rolling(window=3, min_periods=1).mean()
        conditions = [
            (df_15["future_return_smooth"] > LABEL_EPSILON),
            (df_15["future_return_smooth"] < -LABEL_EPSILON)
        ]
        df_15["signal_target"] = np.select(conditions, [1, 0], default=2)
        df_15.dropna(inplace=True)
        logger.info("Signal label distribution", distribution=collections.Counter(df_15["signal_target"]))
        all_needed_cols = list(set(self.signal_feature_cols + ["signal_target"]))
        df_15 = df_15.reindex(columns=df_15.columns.union(all_needed_cols), fill_value=0.0)
        logger.debug("Signal model DataFrame columns after union reindex", columns=df_15.columns.tolist())
        signal_labels = df_15["signal_target"].astype(np.int32).values
        for col in self.signal_feature_cols:
            if col not in df_15.columns:
                df_15[col] = 0.0
        self.actual_signal_cols = self.signal_feature_cols
        scaler = RobustScaler()
        df_15[self.actual_signal_cols] = scaler.fit_transform(df_15[self.actual_signal_cols])
        features = df_15[self.actual_signal_cols].values
        X, y = self._make_sequences(features, signal_labels, self.lookback)
        if len(X) < 1:
            logger.warning("Not enough data after sequence creation for signal ensemble.")
            return
        X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(X, y)
        flat_train = X_train.reshape(-1, X_train.shape[2])
        n_components = min(10, flat_train.shape[0], flat_train.shape[1])
        if flat_train.shape[1] <= 0:
            logger.error("Flat train has zero features; cannot perform PCA.")
            return
        self.pca = auto_select_pca_components(flat_train, variance_threshold=0.95)
        flat_train_pca = self.pca.fit_transform(flat_train)
        X_train = flat_train_pca.reshape(X_train.shape[0], X_train.shape[1], -1)
        def apply_pca(X_data):
            flat = X_data.reshape(-1, X_data.shape[2])
            flat_pca = self.pca.transform(flat)
            return flat_pca.reshape(X_data.shape[0], X_data.shape[1], -1)
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
        avg_pred = np.mean(np.array(preds), axis=0)
        signal_class = np.argmax(avg_pred, axis=1)[0]
        cm = confusion_matrix(y_test, signal_class)
        f1 = f1_score(y_test, signal_class, average="weighted")
        logger.info("Signal Ensemble Evaluation", confusion_matrix=cm.tolist(), weighted_f1=f1)
        self.signal_model_ready = True
        return self.signal_models

    def _make_sequences(self, features, labels, lookback):
        X, y = [], []
        for i in range(len(features) - lookback):
            X.append(features[i : i + lookback])
            y.append(labels[i + lookback])
        return np.array(X), np.array(y)

    def preprocess_dataframe(self, df):
        df = df.copy()
        # Convert numeric columns explicitly to float and log stats
        for col in ["open", "high", "low", "close", "volume"]:
            try:
                df[col] = df[col].astype(float)
            except Exception as e:
                logger.error("Error converting column %s to float: %s", col, str(e))
        logger.debug("Preprocessing: Column stats before reindexing - open(min=%.2f, max=%.2f), high(min=%.2f, max=%.2f), low(min=%.2f, max=%.2f), close(min=%.2f, max=%.2f)",
                     df["open"].min(), df["open"].max(),
                     df["high"].min(), df["high"].max(),
                     df["low"].min(), df["low"].max(),
                     df["close"].min(), df["close"].max())
        # Reindex DataFrame to 1-minute frequency
        start = df.index.min()
        end = df.index.max()
        expected_minutes = int(((end - start).total_seconds() / 60)) + 1
        df = df.reindex(pd.date_range(start=start, end=end, freq="1min"))
        df.sort_index(inplace=True)
        df.ffill(inplace=True)
        logger.debug("Preprocessing: After reindexing, rows=%d", len(df))
        for col in ["close", "volume"]:
            df[col] = exponential_smooth(df[col])
            df[col] = median_filter(df[col], window=5)
        df = add_lag_features(df, ["close"], lags=[1])
        # Instead of dropping rows immediately, repair them if possible.
        repaired_df = df.apply(repair_candle_row, axis=1)
        # Keep only rows with valid close values.
        valid_df = repaired_df[repaired_df["close"] > 0]
        dropped = len(df) - len(valid_df)
        if dropped > 0:
            logger.warning("Dropped %d rows due to data integrity issues (after repair).", dropped)
        if len(valid_df) < 0.8 * expected_minutes:
            logger.warning("Preprocessed candle data has fewer rows (%d) than expected (%d)", len(valid_df), expected_minutes)
        return valid_df

if __name__ == "__main__":
    async def main():
        asyncio.create_task(periodic_gc(300))
        ml_service = MLService()
        await ml_service.initialize()
        while True:
            await asyncio.sleep(60)
    asyncio.run(main())
