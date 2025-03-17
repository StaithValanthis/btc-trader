# File: v2-Inverse-btc-trader/app/ml/ml_inference.py

"""
Handles loading saved models and performing inference (trend / signal predictions).
Also includes warm-start logic to skip re-training if model files are found.
"""

import os
import numpy as np
import pandas as pd
from structlog import get_logger
from typing import List, Optional
from tensorflow.keras.models import load_model
from sklearn.preprocessing import RobustScaler

from app.ml.ml_utils import AttentionLayer, focal_loss
from app.ml.feature_engineering import compute_15min_features
from app.ml.ml_training import (
    build_trend_model_tuner,
    build_signal_model_tuner,
    TREND_MODEL_PATH,
    SIGNAL_MODEL_PATH
)

logger = get_logger(__name__)

class TrendModelService:
    """
    Loads or trains a trend model for 3-class classification: Up / Down / Sideways.
    Expects data columns like 'sma_diff','adx','dmi_diff'.
    """

    def __init__(self, lookback: int=120) -> None:
        self.model = None
        self.lookback = lookback
        self.ready = False
        # We might define an expected feature list if we want to replicate the training
        self.trend_features = ["sma_diff","adx","dmi_diff"]

    def load(self) -> bool:
        """
        Attempt to load a previously saved trend model.
        Returns True if load succeeded, False otherwise.
        """
        if os.path.exists(TREND_MODEL_PATH):
            try:
                self.model = load_model(TREND_MODEL_PATH)
                self.ready = True
                logger.info("Loaded existing trend model.")
                return True
            except Exception as e:
                logger.warning("Failed to load existing trend model", error=str(e))
        return False

    def predict_trend(self, df: pd.DataFrame) -> str:
        """
        Predict the trend (Up, Down, Sideways) from the loaded or trained model.

        Args:
            df (pd.DataFrame): DataFrame that must contain self.trend_features

        Returns:
            str: "Uptrending", "Downtrending", or "Sideways"
        """
        if not self.ready or self.model is None:
            logger.warning("Trend model not ready; defaulting to 'Sideways'.")
            return "Sideways"

        if len(df) < self.lookback:
            return "Sideways"

        from sklearn.preprocessing import RobustScaler
        sub_df = df[self.trend_features].copy().fillna(0)
        scaler = RobustScaler()
        scaled = scaler.fit_transform(sub_df)
        seq = scaled[-self.lookback:]
        seq = np.expand_dims(seq, axis=0)

        preds = self.model.predict(seq)
        trend_class = np.argmax(preds, axis=1)[0]
        mapping = {0: "Uptrending", 1: "Downtrending", 2: "Sideways"}
        return mapping.get(trend_class, "Sideways")


class SignalModelService:
    """
    Loads or trains a signal model for short-term classification: Buy / Sell / Hold
    Expects data columns like 'close','returns','rsi','macd_diff','bb_width','atr'
    aggregated to 15min.
    """

    def __init__(self, lookback: int=120) -> None:
        self.models = []   # could store multiple for ensemble
        self.lookback = lookback
        self.ready = False
        self.signal_features = ["close","returns","rsi","macd_diff","bb_width","atr"]

    def load(self) -> bool:
        """
        Attempt to load a previously saved signal model.
        Returns True if load succeeded, False otherwise.
        """
        if os.path.exists(SIGNAL_MODEL_PATH):
            try:
                loaded = load_model(
                    SIGNAL_MODEL_PATH,
                    custom_objects={"AttentionLayer": AttentionLayer, "focal_loss_fixed": focal_loss()}
                )
                self.models = [loaded]
                self.ready = True
                logger.info("Loaded existing signal model.")
                return True
            except Exception as e:
                logger.warning("Failed to load existing signal model", error=str(e))
        return False

    def predict_signal(self, df_1min: pd.DataFrame) -> str:
        """
        Predict short-term signal (Buy, Sell, Hold) from a loaded or trained model ensemble.

        Args:
            df_1min (pd.DataFrame): Candle data at 1min freq to be aggregated to 15min features.

        Returns:
            str: "Buy", "Sell", or "Hold"
        """
        if not self.ready or not self.models:
            logger.warning("Signal model not ready; defaulting to 'Hold'.")
            return "Hold"

        from app.ml.feature_engineering import compute_15min_features
        df_15 = compute_15min_features(df_1min)
        if len(df_15) < self.lookback:
            return "Hold"

        sub_data = df_15[self.signal_features].copy().fillna(0)
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
        scaled = scaler.fit_transform(sub_data)
        seq = scaled[-self.lookback:]
        seq = np.expand_dims(seq, axis=0)

        preds = [model.predict(seq) for model in self.models]
        avg_preds = np.mean(preds, axis=0)
        signal_class = np.argmax(avg_preds, axis=1)[0]
        mapping = {0: "Sell", 1: "Buy", 2: "Hold"}
        return mapping.get(signal_class, "Hold")
