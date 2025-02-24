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
import ta  # For ATR, RSI, MACD, Bollinger Bands, Ichimoku, etc.
from sklearn.utils import class_weight

from app.services.backfill_service import backfill_mexc_kline
from app.core.database import Database
from app.core.config import Config

logger = get_logger(__name__)

MODEL_PATH = os.path.join("model_storage", "lstm_model.keras")
MIN_TRAINING_ROWS = 2000  # Require at least 2000 rows for training

class MLService:
    """
    Trains an LSTM model to predict market direction.
    
    The model uses a variety of technical indicators, including:
      - Base indicators: close, returns, rsi, macd, macd_signal, macd_diff, bb_high, bb_low, bb_mavg, atr
      - Additional features: mfi, stoch, obv, vwap, ema_20, cci, bb_width, lag1_return
      - Ichimoku Cloud features: tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b
      - New moving average features: sma_10, ema_10, smma_10
    """
    def __init__(self, lookback=60):
        self.lookback = lookback
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

        logger.info("Retraining LSTM model with candle + ATR + Ichimoku + moving average data...")

        query = """
            SELECT time, open, high, low, close, volume
            FROM candles
            ORDER BY time ASC
        """
        rows = await Database.fetch(query)
        if not rows:
            logger.warning("No candle data found for training.")
            return

        if len(rows) < MIN_TRAINING_ROWS:
            logger.warning(f"Only {len(rows)} rows found. Initiating backfill to reach {MIN_TRAINING_ROWS} rows.")
            await backfill_mexc_kline(
                symbol="BTCUSD",
                interval=1,
                start_time_ms=1676000000000,  # example start time (adjust as needed)
                days_to_fetch=7
            )
            rows = await Database.fetch(query)
            if len(rows) < MIN_TRAINING_ROWS:
                logger.warning(f"Still only {len(rows)} rows after backfill. Skipping training.")
                return

        df = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close", "volume"])
        df.set_index("time", inplace=True)
        df.sort_index(inplace=True)

        # Base technical indicators
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
            high=df["high"], low=df["low"], close=df["close"], window=14
        )
        df["atr"] = atr_indicator.average_true_range()

        # Additional features
        df["mfi"] = ta.volume.MFIIndicator(high=df["high"], low=df["low"], close=df["close"], volume=df["volume"], window=14).money_flow_index()
        stoch = ta.momentum.StochasticOscillator(high=df["high"], low=df["low"], close=df["close"], window=14, smooth_window=3)
        df["stoch"] = stoch.stoch()
        df["obv"] = ta.volume.OnBalanceVolumeIndicator(close=df["close"], volume=df["volume"]).on_balance_volume()
        df["vwap"] = ta.volume.VolumeWeightedAveragePrice(high=df["high"], low=df["low"], close=df["close"], volume=df["volume"]).volume_weighted_average_price()
        df["ema_20"] = ta.trend.EMAIndicator(close=df["close"], window=20).ema_indicator()
        df["cci"] = ta.trend.CCIIndicator(high=df["high"], low=df["low"], close=df["close"], window=20).cci()
        df["bb_width"] = df["bb_high"] - df["bb_low"]
        df["lag1_return"] = df["returns"].shift(1)

        # Ichimoku Cloud features
        ichimoku = ta.trend.IchimokuIndicator(
            high=df["high"],
            low=df["low"],
            window1=9,
            window2=26,
            window3=52
        )
        df["tenkan_sen"] = ichimoku.ichimoku_conversion_line()
        df["kijun_sen"] = ichimoku.ichimoku_base_line()
        df["senkou_span_a"] = ichimoku.ichimoku_a()
        df["senkou_span_b"] = ichimoku.ichimoku_b()

        # New Moving Average features
        df["sma_10"] = ta.trend.SMAIndicator(close=df["close"], window=10).sma_indicator()
        df["ema_10"] = ta.trend.EMAIndicator(close=df["close"], window=10).ema_indicator()
        # SMMA is approximated using Wilder’s smoothing which is equivalent to EMA with alpha = 1/window
        df["smma_10"] = df["close"].ewm(alpha=1/10, adjust=False).mean()

        df.dropna(inplace=True)

        # Create classification target: 1 if next bar's return > 0, else 0.
        df["target"] = (df["returns"].shift(-1) > 0).astype(int)
        df.dropna(inplace=True)

        # Define feature lists
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

        features = df[feature_cols].values
        labels = df["target"].values

        X, y = self._make_sequences(features, labels, self.lookback)
        if len(X) < 1:
            logger.warning("Not enough data for training after sequence creation.")
            return

        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        # Compute class weights to balance long and short signals
        class_weights_arr = class_weight.compute_class_weight('balanced',
                                                              classes=np.unique(y_train),
                                                              y=y_train)
        cw = dict(enumerate(class_weights_arr))
        logger.info("Computed class weights", class_weights=cw)

        if self.model is None:
            input_shape = (self.lookback, len(feature_cols))
            self.model = self._build_model(input_shape=input_shape)

        es = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.epochs, batch_size=32,
            callbacks=[es],
            verbose=1,
            class_weight=cw
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

        # Moving Average features
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
        return "Buy" if pred[0][0] > 0.5 else "Sell"

    def _build_model(self, input_shape):
        model = Sequential()
        model.add(LSTM(64, input_shape=input_shape, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation="sigmoid"))
        model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
        return model

    def _make_sequences(self, features, labels, lookback):
        X, y = [], []
        for i in range(len(features) - lookback):
            X.append(features[i:i+lookback])
            y.append(labels[i+lookback])
        return np.array(X), np.array(y)
