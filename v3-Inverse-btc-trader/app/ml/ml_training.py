# File: v2-Inverse-btc-trader/app/ml/ml_training.py

"""
Handles ML model building and advanced training for both trend and signal models.
- Keras Tuner examples (RandomSearch, etc.)
- Build & save model artifacts
"""

import os
import numpy as np
import pandas as pd
import keras_tuner as kt
from structlog import get_logger
from typing import Tuple, Optional

import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Conv1D, LSTM, Dense, Dropout, Bidirectional, Input, BatchNormalization
)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import RobustScaler
from sklearn.utils.class_weight import compute_class_weight

from app.ml.ml_utils import focal_loss, AttentionLayer
from app.ml.feature_engineering import compute_15min_features

logger = get_logger(__name__)

TREND_MODEL_PATH = os.path.join("model_storage", "trend_model.keras")
SIGNAL_MODEL_PATH = os.path.join("model_storage", "signal_model.keras")
LABEL_EPSILON = float(os.getenv("LABEL_EPSILON", "0.0005"))

def make_sequences(features: np.ndarray, labels: np.ndarray, lookback: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates sequences of length `lookback` from the features.

    Args:
        features (np.ndarray): Feature matrix of shape (N, d).
        labels (np.ndarray): Array of labels of length N.
        lookback (int): Sequence length.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (X, y) where
            X.shape = (N-lookback, lookback, d),
            y.shape = (N-lookback,).
    """
    X, y = [], []
    for i in range(len(features) - lookback):
        X.append(features[i : i+lookback])
        y.append(labels[i+lookback])
    return np.array(X), np.array(y)

def train_val_test_split(
    X: np.ndarray,
    y: np.ndarray,
    train_frac: float=0.7,
    val_frac: float=0.15
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Splits data into train/val/test sets (70/15/15 by default).

    Args:
        X (np.ndarray): Feature sequences.
        y (np.ndarray): Labels.
        train_frac (float): Fraction for training.
        val_frac (float): Fraction for validation.

    Returns:
        6-tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    total = len(X)
    train_end = int(total * train_frac)
    val_end = int(total * (train_frac + val_frac))

    return (
        X[:train_end], y[:train_end],
        X[train_end:val_end], y[train_end:val_end],
        X[val_end:], y[val_end:]
    )

def build_trend_model_tuner(hp) -> Model:
    """
    Build a Keras model for trend classification with hyperparameters from Keras Tuner.
    """
    dropout_rate = hp.Float("dropout_rate", 0.1, 0.5, step=0.1, default=0.2)
    lstm_units = hp.Int("lstm_units", 32, 128, step=32, default=64)

    input_shape = (hp.get("lookback"), hp.get("num_features"))
    inputs = Input(shape=input_shape)
    x = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(inputs)
    x = Dropout(dropout_rate)(x)
    x = Bidirectional(LSTM(lstm_units, return_sequences=False))(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(3, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def build_signal_model_tuner(hp) -> Model:
    """
    Build a Keras model for signal classification with hyperparameters from Keras Tuner.
    """
    dropout_rate = hp.Float("dropout_rate", 0.1, 0.5, step=0.1, default=0.2)
    lstm_units = hp.Int("lstm_units", 32, 128, step=32, default=64)
    learning_rate = hp.Float("learning_rate", 1e-4, 1e-2, sampling="log", default=1e-3)
    weight_decay = hp.Float("weight_decay", 1e-5, 1e-3, sampling="log", default=1e-4)

    input_shape = (hp.get("lookback"), hp.get("num_features"))
    inputs = Input(shape=input_shape)
    x = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(inputs)
    x = Dropout(dropout_rate)(x)
    x = Bidirectional(LSTM(lstm_units, return_sequences=True))(x)
    x = Dropout(dropout_rate)(x)
    x = AttentionLayer()(x)
    x = Dense(64, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    outputs = Dense(3, activation="softmax")(x)
    model = Model(inputs=inputs, outputs=outputs)

    optimizer = tf.keras.optimizers.experimental.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
    model.compile(
        loss=focal_loss(gamma=2.0, alpha=0.25),
        optimizer=optimizer,
        metrics=["accuracy"]
    )
    return model

async def train_trend_model_advanced(
    df: pd.DataFrame,
    lookback: int = 120,
    max_trials: int = 5
) -> Model:
    """
    Advanced training approach for a "trend" model using Keras Tuner.

    Args:
        df (pd.DataFrame): Candle data (e.g. 30min or 60min) with columns for 'sma_diff','adx','dmi_diff' + 'trend_target'
        lookback (int): Sequence length
        max_trials (int): Max tuner search trials

    Returns:
        A trained Keras model.
    """

    # Example columns:
    trend_features = ["sma_diff","adx","dmi_diff"]
    if any(col not in df.columns for col in trend_features + ["trend_target"]):
        raise ValueError("DataFrame missing required columns for trend model")

    df.dropna(subset=trend_features + ["trend_target"], inplace=True)
    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler()
    df[trend_features] = scaler.fit_transform(df[trend_features])

    X_, y_ = make_sequences(df[trend_features].values, df["trend_target"].values, lookback)
    if len(X_) < 10:
        raise ValueError("Not enough data after sequence creation to train trend model")

    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(X_, y_)
    # Set up Tuner
    tuner = kt.RandomSearch(
        build_trend_model_tuner,
        objective="val_loss",
        max_trials=max_trials,
        executions_per_trial=1,
        directory="kt_dir",
        project_name="trend_model_tuning"
    )

    # Provide custom hp values for shape
    tuner.search_space_summary()
    tuner.search(
        X_train, to_categorical(y_train, 3),
        validation_data=(X_val, to_categorical(y_val, 3)),
        callbacks=[EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)],
        epochs=10,
        batch_size=32,
        verbose=0,
        # pass hyperparameters for shape
        hp=None,
        initial_hyperparameters={"lookback": lookback, "num_features": len(trend_features)}
    )

    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    logger.info("Trend model best HP", hp=best_hp.values)
    best_model = build_trend_model_tuner(best_hp)
    # Rebuild with shapes
    best_hp_values = best_hp.values
    best_hp_values["lookback"] = lookback
    best_hp_values["num_features"] = len(trend_features)
    best_model = build_trend_model_tuner(best_hp)
    best_model.fit(
        X_train, to_categorical(y_train, 3),
        validation_data=(X_val, to_categorical(y_val, 3)),
        callbacks=[EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)],
        epochs=10,
        batch_size=32,
        verbose=1
    )
    best_model.save(TREND_MODEL_PATH)
    return best_model


async def train_signal_model_advanced(
    df: pd.DataFrame,
    lookback: int = 120,
    max_trials: int = 5
) -> Model:
    """
    Advanced training for the "signal" model using Keras Tuner.

    Args:
        df (pd.DataFrame): 15min candle data with columns for features + 'signal_target'
        lookback (int): Sequence length
        max_trials (int): Max tuner search trials

    Returns:
        A trained Keras model for signal classification.
    """

    # Example feature columns:
    feat_cols = ["close","returns","rsi","macd_diff","bb_width","atr"]
    if any(col not in df.columns for col in feat_cols + ["signal_target"]):
        raise ValueError("DataFrame missing required columns for signal model")

    df.dropna(subset=feat_cols + ["signal_target"], inplace=True)
    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler()
    df[feat_cols] = scaler.fit_transform(df[feat_cols])

    X_, y_ = make_sequences(df[feat_cols].values, df["signal_target"].values, lookback)
    if len(X_) < 10:
        raise ValueError("Not enough data to create sequences for signal model")

    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(X_, y_)
    tuner = kt.RandomSearch(
        build_signal_model_tuner,
        objective="val_loss",
        max_trials=max_trials,
        executions_per_trial=1,
        directory="kt_dir",
        project_name="signal_model_tuning"
    )
    tuner.search_space_summary()

    tuner.search(
        X_train, to_categorical(y_train, 3),
        validation_data=(X_val, to_categorical(y_val, 3)),
        callbacks=[EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)],
        epochs=10,
        batch_size=32,
        verbose=0,
        hp=None,
        initial_hyperparameters={"lookback": lookback, "num_features": len(feat_cols)}
    )

    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    logger.info("Signal model best HP", hp=best_hp.values)
    best_model = build_signal_model_tuner(best_hp)
    best_model.fit(
        X_train, to_categorical(y_train, 3),
        validation_data=(X_val, to_categorical(y_val, 3)),
        callbacks=[EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)],
        epochs=10,
        batch_size=32,
        verbose=1
    )
    best_model.save(SIGNAL_MODEL_PATH)
    return best_model
