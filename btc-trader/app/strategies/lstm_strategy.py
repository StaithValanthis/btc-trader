# File: app/strategies/lstm_strategy.py

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
            prediction_window=5
        )

        # We'll build a placeholder LSTM with a maximum shape.
        self.input_shape = (Config.MODEL_CONFIG['lookback_window'], len(self.preprocessor.required_columns))
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
                if not await self._check_data_availability():
                    await asyncio.sleep(30)
                    continue

                if self._should_retrain():
                    await self.retrain_model()

                await self.execute_trades()
                await self.log_market_analysis()

                await asyncio.sleep(60)

            except Exception as e:
                logger.error("Strategy loop error", error=str(e))
                await asyncio.sleep(10)

    def _should_retrain(self):
        if not self.last_retrain:
            return True
        return (time.time() - self.last_retrain) > Config.MODEL_CONFIG['retrain_interval']

    async def retrain_model(self):
        temp_path = None
        try:
            logger.info("Starting model retraining cycle")

            df_merged = await self.get_multi_timeframe_data()
            if len(df_merged) < self.min_samples:
                logger.warning("Skipping retrain - insufficient data",
                               available=len(df_merged),
                               required=self.min_samples)
                return

            X, y = self.preprocessor.prepare_data(df_merged)
            if X.shape[0] == 0:
                logger.error("Data preparation failed (no training samples)")
                return

            # If shape changed, rebuild the model
            if X.shape[1:] != self.model.model.input_shape[1:]:
                logger.info(f"Rebuilding LSTM model to match new shape {X.shape[1:]}")
                self.model = LSTMModel(X.shape[1:])

            if Config.MODEL_CONFIG['enable_hyperparam_tuning']:
                logger.info("Running hyperparameter tuning...")
                self.model.hyperparameter_tuning(X, y)

            logger.info("Training model", samples=X.shape[0])
            self.model.train(
                X, y,
                epochs=Config.MODEL_CONFIG['train_epochs'],
                batch_size=Config.MODEL_CONFIG['batch_size']
            )

            # Save
            temp_path = "lstm_model_temp.h5"
            final_path = "lstm_model.h5"
            self.model.save(temp_path)

            if not os.path.exists(temp_path) or os.path.getsize(temp_path) < 1024:
                raise IOError("Model save verification failed")

            if os.path.exists(final_path):
                os.remove(final_path)
            os.rename(temp_path, final_path)

            # Reload
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
            # Simple threshold logic
            if prediction[0] > current_price * 1.002:
                await self.trade_service.execute_trade(current_price, 'Buy', position_size)
            elif prediction[0] < current_price * 0.998:
                await self.trade_service.execute_trade(current_price, 'Sell', position_size)

        except Exception as e:
            logger.error("Trade execution error", error=str(e))

    async def log_market_analysis(self):
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
        return np.random.uniform(-1, 1)

    async def _check_data_availability(self):
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
        account_balance = 10000  # Replace with actual exchange data
        risk_percent = 0.02
        dollar_risk = account_balance * risk_percent
        return min(dollar_risk / current_price, Config.TRADING_CONFIG['position_size'])

    # ----------------------------------------------------------------
    # Updated Multi-Timeframe Data Fetch with 1m priority
    # ----------------------------------------------------------------
    async def get_multi_timeframe_data(self):
        """
        Fetch 1-minute bars from the last X hours.
        Attempt 5-minute bars. If 5m is empty, proceed with 1m only.
        Then run the 1m/5m indicators if data is present.
        """
        if Config.MODEL_CONFIG['use_rolling_window']:
            time_threshold = datetime.utcnow() - timedelta(hours=Config.MODEL_CONFIG['rolling_window_hours'])
        else:
            time_threshold = datetime.utcnow() - timedelta(days=90)

        # 1) Fetch 1-minute bars
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
        if df1m.empty:
            logger.warning("Not enough 1-minute data yet.")
            return pd.DataFrame()

        df1m['bucket'] = pd.to_datetime(df1m['bucket'])
        df1m.set_index('bucket', inplace=True)
        df1m.rename(columns={
            'open': 'open_1m',
            'high': 'high_1m',
            'low': 'low_1m',
            'close': 'close_1m',
            'volume': 'volume_1m'
        }, inplace=True)

        # 2) Attempt fetching 5-minute bars
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
        if df5m.empty:
            logger.warning("5-minute data not yet available; using only 1-minute bars.")
            # Calculate 1m indicators only
            self.preprocessor.create_features_for_1m(df1m)
            if df1m.empty:
                logger.warning("After 1m indicator calculation, no data remains.")
                return pd.DataFrame()
            logger.info("Returning 1-minute only data", samples=len(df1m))
            return df1m

        df5m['bucket'] = pd.to_datetime(df5m['bucket'])
        df5m.set_index('bucket', inplace=True)
        df5m.rename(columns={
            'open': 'open_5m',
            'high': 'high_5m',
            'low': 'low_5m',
            'close': 'close_5m',
            'volume': 'volume_5m'
        }, inplace=True)

        # 3) Merge 1m & 5m
        df_merged = self.preprocessor.merge_timeframes(df1m, df5m)

        # 4) Indicators for each timeframe
        self.preprocessor.create_features_for_1m(df_merged)
        self.preprocessor.create_features_for_5m(df_merged)

        if df_merged.empty:
            logger.warning("After indicator calculation, no data remains.")
            return pd.DataFrame()

        logger.info("Merged multi-timeframe data prepared", samples=len(df_merged))
        return df_merged
