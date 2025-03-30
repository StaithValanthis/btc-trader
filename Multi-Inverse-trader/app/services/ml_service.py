import asyncio
import os
import shutil
import pandas as pd
import numpy as np
import tensorflow as tf
from datetime import datetime, timezone
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout, Conv1D,
    Bidirectional, BatchNormalization, Layer
)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
import ta
from structlog import get_logger
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight

# Attempt to import keras_tuner
try:
    import keras_tuner as kt
except ImportError:
    raise ImportError("Please install keras-tuner via: pip install keras-tuner")

try:
    import shap
except ImportError:
    shap = None
    print("SHAP not installed; skipping feature importance analysis")

from app.core.database import Database
from app.core.config import Config
from app.services.backfill_service import backfill_bybit_kline

logger = get_logger(__name__)

LABEL_EPSILON = float(os.getenv("LABEL_EPSILON", "0.0005"))

# -------------------------------------------------------------------
# 1) CUSTOM NO-CHECKPOINT TUNER
# -------------------------------------------------------------------
class NoCheckpointRandomSearch(kt.RandomSearch):
    """
    A Tuner subclass that disables ephemeral checkpoint creation/loading,
    preventing 'checkpoint_temp' merges and 'file not found' errors.
    """
    def _save_model(self, trial_id, model, step=0):
        pass  # no-op

    def _checkpoint_trial(self, trial):
        pass  # no-op

    def _try_restore_best_weights(self, trial):
        pass  # no-op

# -------------------------------------------------------------------
# 2) ATTENTION LAYER
# -------------------------------------------------------------------
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def build(self, input_shape):
        self.W = self.add_weight(
            "att_weight",
            shape=(input_shape[-1], 1),
            initializer="normal"
        )
        self.b = self.add_weight(
            "att_bias",
            shape=(input_shape[1], 1),
            initializer="zeros"
        )
        super().build(input_shape)
    def call(self, x):
        e = tf.nn.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
        a = tf.nn.softmax(e, axis=1)
        return tf.reduce_sum(x * a, axis=1)

# -------------------------------------------------------------------
# 3) FOCAL LOSS (if needed)
# -------------------------------------------------------------------
def focal_loss(gamma=2.0, alpha=0.25):
    def loss_fn(y_true, y_pred):
        epsilon = 1e-8
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * tf.pow((1 - y_pred), gamma)
        return tf.reduce_mean(tf.reduce_sum(weight * cross_entropy, axis=1))
    return loss_fn

# -------------------------------------------------------------------
# 4) TRAIN/VAL/TEST SPLIT
# -------------------------------------------------------------------
def train_val_test_split(X, y, train_frac=0.7, val_frac=0.15):
    total = len(X)
    train_end = int(total * train_frac)
    val_end = int(total * (train_frac + val_frac))
    return (X[:train_end], y[:train_end],
            X[train_end:val_end], y[train_end:val_end],
            X[val_end:], y[val_end:])

# -------------------------------------------------------------------
# 5) CREATE LSTM SEQUENCES
# -------------------------------------------------------------------
def make_sequences(features, labels, lookback):
    X, y = [], []
    for i in range(len(features) - lookback):
        X.append(features[i : i + lookback])
        y.append(labels[i + lookback])
    return np.array(X), np.array(y)

# -------------------------------------------------------------------
# 6) MLService
# -------------------------------------------------------------------
class MLService:
    """
    MLService fetches data from 'candles' for each symbol, 
    builds & tunes 'trend' (30/60-min) and 'signal' (15-min) models,
    uses a custom Tuner that never saves ephemeral checkpoints,
    and provides daily retraining plus 'predict_trend/signal'.
    """

    def __init__(
        self,
        symbol: str,
        lookback=120,
        signal_horizon=5,
        focal_gamma=2.0,
        focal_alpha=0.25,
        batch_size=32,
        ensemble_size=3,
        use_tuned_trend_model=True,
        use_tuned_signal_model=True
    ):
        self.symbol = symbol
        self.lookback = lookback
        self.signal_horizon = signal_horizon
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        self.batch_size = batch_size
        self.ensemble_size = ensemble_size
        self.use_tuned_trend_model = use_tuned_trend_model
        self.use_tuned_signal_model = use_tuned_signal_model

        self.running = True
        self.initialized = False
        self.epochs = 10  # or 20, etc.

        # Final models
        self.trend_model = None
        self.signal_models = []
        self.trend_model_ready = False
        self.signal_model_ready = False

        # Feature columns
        self.trend_feature_cols = ["sma_diff", "adx", "dmi_diff"]
        self.signal_feature_cols = [
            "close", "returns", "rsi", "macd_diff",
            "obv", "vwap", "mfi", "bb_width", "atr"
        ]
        self.actual_trend_cols = None
        self.actual_signal_cols = None

        # If you do PCA or scaling
        self.pca = None

    async def initialize(self):
        """
        Called once at startup. Optionally backfill, then do initial training.
        """
        logger.info(f"Initializing MLService for symbol={self.symbol}")
        try:
            self.initialized = True
            # e.g. await maybe_backfill_candles(...) if needed

            # TUNE/Train trend
            logger.info(f"Starting trend model for {self.symbol}")
            if self.use_tuned_trend_model:
                await self.tune_trend_model()
            else:
                await self.train_trend_model()

            # TUNE/Train signal
            logger.info(f"Starting signal model for {self.symbol}")
            if self.use_tuned_signal_model:
                await self.tune_signal_model()
            else:
                await self.train_signal_ensemble(n_models=self.ensemble_size)

            logger.info(f"Initial training complete for {self.symbol}")
        except Exception as e:
            logger.error("Error initializing MLService", symbol=self.symbol, error=str(e))
            self.initialized = False

    async def schedule_daily_retrain(self):
        """
        Called in parallel or in main loop to retrain every 4 hours.
        """
        while self.running:
            logger.info(f"Retrain cycle for {self.symbol} begins...")
            try:
                if self.use_tuned_trend_model:
                    await self.tune_trend_model()
                else:
                    await self.train_trend_model()

                if self.use_tuned_signal_model:
                    await self.tune_signal_model()
                else:
                    await self.train_signal_ensemble(n_models=self.ensemble_size)

                logger.info(f"Retrain cycle done for {self.symbol}")
            except Exception as e:
                logger.error("Error in daily retrain", symbol=self.symbol, error=str(e))

            await asyncio.sleep(14400)  # 4 hours

    async def stop(self):
        self.running = False
        logger.info(f"MLService for {self.symbol} stopped")

    # -------------------------------------------------------------------
    # (A) TREND MODEL
    # -------------------------------------------------------------------
    def _build_trend_model_hyper(self, hp):
        """
        Keras Tuner hypermodel for 'trend' classification:
        Up(0), Down(1), Sideways(2).
        """
        dropout_rate = hp.Float("dropout_rate", 0.1, 0.5, step=0.1, default=0.2)
        lstm_units = hp.Int("lstm_units", 32, 128, step=32, default=64)
        if not self.actual_trend_cols:
            raise ValueError(f"No actual_trend_cols for trend model on {self.symbol}")

        input_shape = (self.lookback, len(self.actual_trend_cols))
        tf.keras.backend.clear_session()

        inp = Input(shape=input_shape, name="trend_input")
        # unique names
        x = Conv1D(32, 3, activation='relu', padding='same', name="trend_conv1d")(inp)
        x = Dropout(dropout_rate, name="trend_dropout_1")(x)
        x = Bidirectional(LSTM(lstm_units, return_sequences=False), name="trend_bilstm")(x)
        x = Dropout(dropout_rate, name="trend_dropout_2")(x)
        out = Dense(3, activation='softmax', name="trend_output")(x)

        model = Model(inp, out, name="trend_model")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        # minimal dummy run
        model.fit(
            np.zeros((2, self.lookback, len(self.actual_trend_cols))),
            to_categorical(np.zeros(2), num_classes=3),
            epochs=1, batch_size=1, verbose=0
        )
        return model

    async def tune_trend_model(self):
        """
        1) Query 'candles' data for self.symbol
        2) 30/60-min composite => sma_diff, adx, dmi_diff
        3) Build sequences
        4) Use NoCheckpointRandomSearch Tuner
        """
        logger.info(f"Tuning trend model for {self.symbol}")
        rows = await Database.fetch("""
            SELECT time, open, high, low, close, volume
            FROM candles
            WHERE symbol=$1
            ORDER BY time ASC
        """, self.symbol)
        if not rows or len(rows)<50:
            logger.warning(f"Not enough data for trend tuning on {self.symbol}")
            return

        df = pd.DataFrame(rows, columns=["time","open","high","low","close","volume"])
        df["time"] = pd.to_datetime(df["time"])
        df.set_index("time", inplace=True)
        df = df.asfreq("1min")

        df_30 = df.resample("30min").agg({
            "open":"first","high":"max","low":"min",
            "close":"last","volume":"sum"
        }).ffill().dropna()
        df_60 = df.resample("60min").agg({
            "open":"first","high":"max","low":"min",
            "close":"last","volume":"sum"
        }).ffill().dropna()
        df_60_aligned = df_60.reindex(df_30.index, method="ffill")

        for tmp in [df_30, df_60_aligned]:
            tmp["sma20"] = tmp["close"].rolling(20, min_periods=1).mean()
            tmp["sma50"] = tmp["close"].rolling(50, min_periods=1).mean()
            tmp["sma_diff"] = tmp["sma20"] - tmp["sma50"]
            adx = ta.trend.ADXIndicator(tmp["high"], tmp["low"], tmp["close"], window=14)
            tmp["adx"] = adx.adx()
            try:
                from ta.trend import DMIIndicator
                dmi = DMIIndicator(tmp["high"], tmp["low"], tmp["close"], 14)
                tmp["dmi_diff"] = dmi.plus_di() - dmi.minus_di()
            except:
                tmp["dmi_diff"] = 0.0

        composite = pd.DataFrame(index=df_30.index)
        composite["sma_diff"] = (df_30["sma_diff"]+df_60_aligned["sma_diff"])/2
        composite["adx"]      = (df_30["adx"]+df_60_aligned["adx"])/2
        composite["dmi_diff"] = (df_30["dmi_diff"]+df_60_aligned["dmi_diff"])/2
        composite.dropna(inplace=True)
        if composite.empty:
            logger.warning(f"No composite trend data for {self.symbol}")
            return

        # label up(0), down(1), sideways(2)
        default_thresh = 0.3
        default_adx_thresh = 20
        composite["trend_target"] = np.where(
            (composite["sma_diff"]>default_thresh) & (composite["adx"]>default_adx_thresh), 0,
            np.where(
                (composite["sma_diff"]< -default_thresh) & (composite["adx"]>default_adx_thresh),
                1, 2
            )
        )
        self.actual_trend_cols = ["sma_diff","adx","dmi_diff"]

        sc = RobustScaler()
        composite[self.actual_trend_cols] = sc.fit_transform(composite[self.actual_trend_cols])

        from app.services.ml_service import make_sequences, train_val_test_split  # or local function
        X,y = make_sequences(
            composite[self.actual_trend_cols].values,
            composite["trend_target"].values,
            self.lookback
        )
        if len(X)<1:
            logger.warning(f"No sequences for {self.symbol} (trend).")
            return
        X_train,y_train,X_val,y_val,_,_ = train_val_test_split(X,y)
        if X_train.shape[0]<1 or X_train.shape[2]<1:
            logger.warning(f"Invalid shape for {self.symbol} trend: {X_train.shape}")
            return

        base_dir = "kt_dir"
        project_name = f"trend_model_tuning_{self.symbol}"  # unique name
        tune_dir = os.path.join(base_dir, project_name)
        shutil.rmtree(tune_dir, ignore_errors=True)

        tuner = NoCheckpointRandomSearch(
            hypermodel=self._build_trend_model_hyper,
            objective="val_loss",
            max_trials=5,
            executions_per_trial=1,
            directory=base_dir,
            project_name=project_name,
            overwrite=True
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
        logger.info("Trend tuning complete", symbol=self.symbol, best_hp=best_hp.values)

        final_model = self._build_trend_model_hyper(best_hp)
        self.trend_model = final_model
        self.trend_model_ready = True
        logger.info(f"Trend model is ready for {self.symbol}")

    async def train_trend_model(self):
        # Fallback direct training if not tuning
        await self.tune_trend_model()

    # -------------------------------------------------------------------
    # (B) SIGNAL MODEL
    # -------------------------------------------------------------------
    def _build_signal_model_hyper(self, hp):
        dropout_rate = hp.Float("dropout_rate", 0.1, 0.5, step=0.1, default=0.2)
        lstm_units = hp.Int("lstm_units", 32, 128, step=32, default=64)
        learning_rate = hp.Float("learning_rate", 1e-4, 1e-2, sampling="log", default=1e-3)
        weight_decay = hp.Float("weight_decay", 1e-5, 1e-3, sampling="log", default=1e-4)

        if not self.actual_signal_cols:
            raise ValueError(f"No actual_signal_cols for signal model on {self.symbol}")

        input_shape = (self.lookback, len(self.actual_signal_cols))
        tf.keras.backend.clear_session()

        inp = Input(shape=input_shape, name="signal_input")
        x = Conv1D(32, 3, activation='relu', padding='same', name="signal_conv1d")(inp)
        x = Dropout(dropout_rate, name="signal_dropout_1")(x)
        x = Bidirectional(LSTM(lstm_units, return_sequences=True), name="signal_bi_lstm1")(x)
        x = Dropout(dropout_rate, name="signal_dropout_2")(x)
        x = Bidirectional(LSTM(32, return_sequences=True), name="signal_bi_lstm2")(x)
        x = AttentionLayer(name="signal_attention")(x)

        x = Dense(128, activation="relu", name="signal_dense1")(x)
        x = BatchNormalization(name="signal_bn1")(x)
        x = Dropout(0.4, name="signal_dropout_3")(x)
        x = Dense(64, activation="relu", name="signal_dense2")(x)
        x = BatchNormalization(name="signal_bn2")(x)
        x = Dropout(0.3, name="signal_dropout_4")(x)
        x = Dense(32, activation="relu", name="signal_dense3")(x)
        x = BatchNormalization(name="signal_bn3")(x)

        out = Dense(3, activation="softmax",
                    kernel_regularizer=tf.keras.regularizers.l2(0.001),
                    name="signal_output")(x)
        model = Model(inp, out, name="signal_model")

        from tensorflow.keras.optimizers.experimental import AdamW
        opt = AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
        model.compile(
            loss=focal_loss(gamma=self.focal_gamma, alpha=self.focal_alpha),
            optimizer=opt,
            metrics=["accuracy"]
        )
        # minimal dummy run
        model.fit(
            np.zeros((2, self.lookback, len(self.actual_signal_cols))),
            to_categorical(np.zeros(2), num_classes=3),
            epochs=1, batch_size=1, verbose=0
        )
        return model

    async def tune_signal_model(self):
        logger.info(f"Tuning signal model for {self.symbol}")
        rows = await Database.fetch("""
            SELECT time, open, high, low, close, volume
            FROM candles
            WHERE symbol=$1
            ORDER BY time ASC
        """, self.symbol)
        if not rows or len(rows)<50:
            logger.warning(f"Not enough data for signal tune on {self.symbol}")
            return

        df = pd.DataFrame(rows, columns=["time","open","high","low","close","volume"])
        df["time"] = pd.to_datetime(df["time"])
        df.set_index("time", inplace=True)
        df = df.asfreq("1min")

        df_15 = df.resample("15min").agg({
            "open":"first","high":"max","low":"min",
            "close":"last","volume":"sum"
        }).ffill().dropna()

        df_15["returns"] = df_15["close"].pct_change()
        df_15["rsi"] = ta.momentum.rsi(df_15["close"], 14)
        macd = ta.trend.MACD(df_15["close"], 26,12,9)
        df_15["macd_diff"] = macd.macd_diff()
        boll = ta.volatility.BollingerBands(df_15["close"], window=20, window_dev=2)
        df_15["bb_width"] = boll.bollinger_hband() - boll.bollinger_lband()

        atr_ind = ta.volatility.AverageTrueRange(
            high=df_15["high"], low=df_15["low"], close=df_15["close"], window=14
        )
        df_15["atr"] = atr_ind.average_true_range()

        df_15["future_return"] = (df_15["close"].shift(-self.signal_horizon)/df_15["close"]) -1
        df_15["future_return_smooth"] = df_15["future_return"].rolling(3, min_periods=1).mean()

        conditions = [
            (df_15["future_return_smooth"]>LABEL_EPSILON),
            (df_15["future_return_smooth"]< -LABEL_EPSILON)
        ]
        df_15["signal_target"] = np.select(conditions, [1,0], default=2)
        df_15.dropna(inplace=True)

        for c in self.signal_feature_cols:
            if c not in df_15.columns:
                df_15[c] = 0.0

        from sklearn.preprocessing import RobustScaler
        sc = RobustScaler()
        df_15[self.signal_feature_cols] = sc.fit_transform(df_15[self.signal_feature_cols])

        feats = df_15[self.signal_feature_cols].values
        labs  = df_15["signal_target"].values.astype(np.int32)

        X,y = make_sequences(feats, labs, self.lookback)
        if len(X)<1:
            logger.warning(f"No sequences for {self.symbol} (signal).")
            return

        X_train,y_train,X_val,y_val,_,_ = train_val_test_split(X,y)
        if X_train.shape[0]<1 or X_train.shape[2]<1:
            logger.warning(f"Invalid shape for signal on {self.symbol}: {X_train.shape}")
            return

        base_dir = "kt_dir"
        project_name = f"signal_model_tuning_{self.symbol}"  # unique name to avoid concurrency collision
        tune_dir = os.path.join(base_dir, project_name)
        shutil.rmtree(tune_dir, ignore_errors=True)

        tuner = NoCheckpointRandomSearch(
            hypermodel=self._build_signal_model_hyper,
            objective="val_loss",
            max_trials=5,
            executions_per_trial=1,
            directory=base_dir,
            project_name=project_name,
            overwrite=True
        )
        await asyncio.to_thread(lambda: tuner.search(
            X_train, to_categorical(y_train, num_classes=3),
            validation_data=(X_val, to_categorical(y_val, num_classes=3)),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=[EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)],
            verbose=0
        ))
        best_hp = tuner.get_best_hyperparameters(1)[0]
        logger.info("Signal tuning complete", symbol=self.symbol, best_hp=best_hp.values)

        final_model = self._build_signal_model_hyper(best_hp)
        self.signal_models = [final_model]
        self.signal_model_ready = True
        logger.info(f"Signal model is ready for {self.symbol}")

    async def train_signal_ensemble(self, n_models=3):
        """
        Fallback direct training if not using tuner. 
        e.g. build multiple models, average predictions, etc.
        """
        logger.info(f"Fallback: training signal ensemble for {self.symbol}, not fully implemented.")
        self.signal_models = []
        self.signal_model_ready = True

    # -------------------------------------------------------------------
    # PREDICTIONS
    # -------------------------------------------------------------------
    def predict_trend(self, recent_data: pd.DataFrame) -> str:
        if not self.trend_model_ready or not self.trend_model:
            logger.warning(f"No trend model for {self.symbol}")
            return "Hold"
        seq = self._prepare_data_sequence(recent_data, self.actual_trend_cols or self.trend_feature_cols)
        if seq is None:
            return "Hold"
        preds = self.trend_model.predict(seq)
        c = np.argmax(preds, axis=1)[0]
        return {0:"Uptrending",1:"Downtrending",2:"Sideways"}.get(c,"Sideways")

    def predict_signal(self, recent_data: pd.DataFrame) -> str:
        if not self.signal_model_ready or not self.signal_models:
            logger.warning(f"No signal models for {self.symbol}")
            return "Hold"
        seq = self._prepare_data_sequence(recent_data, self.actual_signal_cols or self.signal_feature_cols)
        if seq is None:
            return "Hold"
        preds = [m.predict(seq) for m in self.signal_models]
        avg = np.mean(np.array(preds), axis=0)
        c = np.argmax(avg, axis=1)[0]
        return {0:"Sell",1:"Buy",2:"Hold"}.get(c,"Hold")

    # -------------------------------------------------------------------
    # HELPER: data prep for inference
    # -------------------------------------------------------------------
    def _prepare_data_sequence(self, df: pd.DataFrame, feature_list: list, resample_period="15min"):
        """
        Resample, scale, pad to produce [1, lookback, #features].
        """
        if df.empty or not feature_list:
            return None

        data = df.asfreq("1min").copy()
        df_res = data.resample(resample_period).agg({
            "open":"first","high":"max","low":"min","close":"last","volume":"sum"
        }).ffill().dropna()

        for col in feature_list:
            if col not in df_res.columns:
                df_res[col] = 0.0

        sc = RobustScaler()
        df_res[feature_list] = sc.fit_transform(df_res[feature_list])

        if len(df_res) < self.lookback:
            logger.warning(f"Not enough bars for inference on {self.symbol}")
            return None

        recent_slice = df_res.tail(self.lookback).copy()
        if len(recent_slice)<self.lookback:
            missing = self.lookback - len(recent_slice)
            pad_df = pd.DataFrame([recent_slice.iloc[0].values]*missing, columns=recent_slice.columns)
            recent_slice = pd.concat([pad_df, recent_slice], ignore_index=True)

        seq = recent_slice[feature_list].values
        seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        if self.pca is not None:
            # If you do PCA, apply here
            flat = seq.reshape(-1, seq.shape[1])
            flat = self.pca.transform(flat)
            seq = flat.reshape(seq.shape[0], -1)

        seq = np.expand_dims(seq, axis=0)
        return seq
