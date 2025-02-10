import asyncio
import time
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from structlog import get_logger
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from app.core import Config, Database
from app.services.trade_service import TradeService
from app.ml.data_preprocessor import DataPreprocessor
from app.ml.lstm_model import LSTMModel
from app.utils.progress import ProgressBar

logger = get_logger(__name__)

class LSTMStrategy:
    def __init__(self, trade_service: TradeService):
        self.trade_service = trade_service
        self.preprocessor = DataPreprocessor(
            lookback=Config.MODEL_CONFIG['lookback_window'],
            prediction_window=5  # you can adjust
        )

        # Because we're merging 1m & 5m data, total features might be large.
        # We'll figure out the shape after we create the data the first time,
        # but let's keep a "max" placeholder for now. This can be replaced once we know.
        # A quick approach is to set something large, or do a dynamic build in `_initialize_model()`.
        self.input_shape = (Config.MODEL_CONFIG['lookback_window'], len(self.preprocessor.required_columns))

        # We create a placeholder model object:
        self.model = LSTMModel(self.input_shape)
        self.model_loaded = False

        self.analyzer = SentimentIntensityAnalyzer()
        self.last_retrain = None
        self.data_ready = False
        self.warmup_start_time = None
        self.last_log_time = time.time() - 300

        self.min_samples = Config.MODEL_CONFIG['min_training_samples']
        self.progress_bar = ProgressBar(total=self.min_samples)

    async def run(self):
        """Main strategy loop."""
        logger.info("Strategy thread started")
        while True:
            try:
                # Data readiness check
                if not await self._check_data_availability():
                    await asyncio.sleep(30)
                    continue

                # Periodic retraining (every 24h)
                if self._should_retrain():
                    await self.retrain_model()

                # Trading operations
                await self.execute_trades()
                await self.log_market_analysis()

                await asyncio.sleep(60)

            except Exception as e:
                logger.error("Strategy loop error", error=str(e))
                await asyncio.sleep(10)

    def _should_retrain(self):
        """Determine if retraining is needed (every 24 hours by default)."""
        if not self.last_retrain:
            return True
        return (time.time() - self.last_retrain) > Config.MODEL_CONFIG['retrain_interval']

    async def retrain_model(self):
        """Robust model retraining with atomic operations."""
        temp_path = None
        try:
            logger.info("Starting model retraining cycle")

            # 1. Data collection
            df_merged = await self.get_multi_timeframe_data()
            if len(df_merged) < self.min_samples:
                logger.warning("Skipping retrain - insufficient data",
                               available=len(df_merged),
                               required=self.min_samples)
                return

            # 2. Data preparation
            X, y = self.preprocessor.prepare_data(df_merged)
            if X.shape[0] == 0:
                logger.error("Data preparation failed (no training samples)")
                return

            # Rebuild the model if input shape changed
            if X.shape[1:] != self.model.model.input_shape[1:]:
                logger.info(f"Rebuilding LSTM model to match new shape {X.shape[1:]}")
                self.model = LSTMModel(X.shape[1:])

            # 2.5 (optional) Hyperparameter Tuning
            if Config.MODEL_CONFIG['enable_hyperparam_tuning']:
                logger.info("Running hyperparameter tuning...")
                self.model.hyperparameter_tuning(X, y)
                # Rebuild final best model or re-load from tuner. For brevity, we skip details here.

            # 3. Model training
            logger.info("Training model", samples=X.shape[0])
            self.model.train(
                X, y, 
                epochs=Config.MODEL_CONFIG['train_epochs'], 
                batch_size=Config.MODEL_CONFIG['batch_size']
            )

            # 4. Model saving
            temp_path = "lstm_model_temp.h5"
            final_path = "lstm_model.h5"
            self.model.save(temp_path)

            if not os.path.exists(temp_path) or os.path.getsize(temp_path) < 1024:
                raise IOError("Model save verification failed")

            # Atomic replacement
            if os.path.exists(final_path):
                os.remove(final_path)
            os.rename(temp_path, final_path)

            # 5. Reload to confirm
            self.model.load(final_path)
            self.model_loaded = True
            logger.info("Model update successful")

        except Exception as e:
            logger.error("Retraining cycle failed", error=str(e))
            self.model_loaded = False
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception as exc:
                    logger.error("Temp file cleanup failed", error=str(exc))
        finally:
            self.last_retrain = time.time()

    async def execute_trades(self):
        """Trade execution with model validation."""
        try:
            if not self.model_loaded:
                logger.warning("Skipping trades - model not loaded")
                return

            prediction = await self.get_prediction()
            if prediction is None or len(prediction) == 0:
                return

            current_price = await self.trade_service.get_current_price()
            if current_price is None:
                return

            position_size = self._calculate_position_size(current_price)

            # Simple threshold logic:
            if prediction[0] > current_price * 1.002:
                await self.trade_service.execute_trade(current_price, 'Buy', position_size)
            elif prediction[0] < current_price * 0.998:
                await self.trade_service.execute_trade(current_price, 'Sell', position_size)

        except Exception as e:
            logger.error("Trade execution error", error=str(e))

    async def log_market_analysis(self):
        """Periodic market analysis logging."""
        if time.time() - self.last_log_time >= 300:
            try:
                sentiment = await self.fetch_sentiment()
                prediction = await self.get_prediction()
                logger.info(
                    "Market Snapshot",
                    sentiment_score=sentiment,
                    last_prediction=float(prediction[0]) if (prediction is not None and len(prediction) > 0) else None,
                    model_confidence=float(np.std(prediction)) if (prediction is not None and len(prediction) > 0) else None
                )
                self.last_log_time = time.time()
            except Exception as e:
                logger.error("Market analysis failed", error=str(e))

    async def fetch_sentiment(self):
        """Fetch market sentiment data (placeholder)."""
        return np.random.uniform(-1, 1)

    async def _check_data_availability(self):
        """Check data readiness with progress tracking."""
        if not self.data_ready:
            if not self.warmup_start_time:
                self.warmup_start_time = time.time()
                logger.info("Warmup phase started")

            count_result = await Database.fetch("SELECT COUNT(*) FROM market_data")
            current_count = count_result[0]['count'] if count_result else 0

            self.progress_bar.update(current_count)

            if current_count >= self.min_samples:
                self.data_ready = True
                self.progress_bar.clear()
                logger.info("Warmup complete - Starting trading")
                return True
            return False
        return True

    async def get_prediction(self):
        """Use the model to generate a next price prediction from the merged dataset."""
        try:
            if not self.model_loaded:
                return None

            df_merged = await self.get_multi_timeframe_data()
            if df_merged.empty:
                return None

            X, _ = self.preprocessor.prepare_data(df_merged)
            if X.shape[0] == 0:
                return None

            return self.model.predict(X[-1:]).flatten()

        except Exception as e:
            logger.error("Prediction failed", error=str(e))
            return None

    def _calculate_position_size(self, current_price):
        """Basic risk-adjusted position sizing."""
        account_balance = 10000  # Replace with real data from exchange
        risk_percent = 0.02
        dollar_risk = account_balance * risk_percent
        return min(dollar_risk / current_price, Config.TRADING_CONFIG['position_size'])

    # -------------------------------------------------------------
    # Fetching 1m & 5m data from your database & merging
    # -------------------------------------------------------------
    async def get_multi_timeframe_data(self):
        """
        Fetch 1-min bars and 5-min bars from the last X hours/days,
        merge them, then run the feature calculations.
        """
        # Decide how far back to fetch based on rolling_window or entire history
        if Config.MODEL_CONFIG['use_rolling_window']:
            time_threshold = datetime.utcnow() - timedelta(hours=Config.MODEL_CONFIG['rolling_window_hours'])
        else:
            # Grab a large window if not using rolling
            time_threshold = datetime.utcnow() - timedelta(days=90)

        # 1. Fetch 1-minute bars
        records_1m = await Database.fetch('''
            SELECT 
                time_bucket('1 minute', time) AS bucket,
                FIRST(price, time) AS open,
                MAX(price) AS high,
                MIN(price) AS low,
                LAST(price, time) AS close,
                SUM(volume) AS volume
            FROM market_data
            WHERE time > $1
            GROUP BY bucket
            ORDER BY bucket ASC
        ''', time_threshold)

        df1m = pd.DataFrame([dict(r) for r in records_1m])
        if not df1m.empty:
            df1m['bucket'] = pd.to_datetime(df1m['bucket'])
            df1m.set_index('bucket', inplace=True)
            df1m.rename(columns={
                'open': 'open_1m',
                'high': 'high_1m',
                'low': 'low_1m',
                'close': 'close_1m',
                'volume': 'volume_1m'
            }, inplace=True)

        # 2. Fetch 5-minute bars
        records_5m = await Database.fetch('''
            SELECT 
                time_bucket('5 minutes', time) AS bucket,
                FIRST(price, time) AS open,
                MAX(price) AS high,
                MIN(price) AS low,
                LAST(price, time) AS close,
                SUM(volume) AS volume
            FROM market_data
            WHERE time > $1
            GROUP BY bucket
            ORDER BY bucket ASC
        ''', time_threshold)

        df5m = pd.DataFrame([dict(r) for r in records_5m])
        if not df5m.empty:
            df5m['bucket'] = pd.to_datetime(df5m['bucket'])
            df5m.set_index('bucket', inplace=True)
            df5m.rename(columns={
                'open': 'open_5m',
                'high': 'high_5m',
                'low': 'low_5m',
                'close': 'close_5m',
                'volume': 'volume_5m'
            }, inplace=True)

        if df1m.empty or df5m.empty:
            logger.warning("Not enough data in 1m or 5m timeframe.")
            return pd.DataFrame()

        # 3. Merge timeframes
        df_merged = self.preprocessor.merge_timeframes(df1m, df5m)

        # 4. Create indicators for each timeframe
        #    We'll do 1m indicators first, then 5m.
        self.preprocessor.create_features_for_1m(df_merged)
        self.preprocessor.create_features_for_5m(df_merged)

        if df_merged.empty:
            logger.warning("After indicator calculation, no data remains.")
            return pd.DataFrame()

        logger.info("Merged multi-timeframe data prepared", samples=len(df_merged))
        return df_merged
