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
        self.preprocessor = DataPreprocessor(lookback=Config.MODEL_CONFIG['lookback_window'])
        self.input_shape = (Config.MODEL_CONFIG['lookback_window'], len(self.preprocessor.required_columns))
        self.model = self._initialize_model()
        self.current_position = None  # Track open positions: None/"long"/"short"
        self.last_trade_time = None
        self.analyzer = SentimentIntensityAnalyzer()
        self.last_retrain = None
        self.data_ready = False
        self.warmup_start_time = None
        self.last_log_time = time.time() - 300
        self.model_loaded = False
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

                # Regular retraining
                if self._should_retrain():
                    await self.retrain_model()

                # Trading operations
                await self.execute_trades()
                await self.log_market_analysis()

                await asyncio.sleep(60)
            except Exception as e:
                logger.error("Strategy loop error", error=str(e))
                await asyncio.sleep(10)

    def _initialize_model(self):
        model_path = "model_storage/lstm_model.keras"
        try:
            if os.path.exists(model_path) and self._validate_model_file(model_path):
                logger.info("Loading validated model", path=model_path)
                model = LSTMModel(self.input_shape)
                model.load(model_path)
                self.model_loaded = True  # Ensure this is set
                logger.info("Model loaded successfully")
                return model
            else:
                logger.warning("No valid model found, initializing new model")
                self.model_loaded = False  # Explicit state
                return LSTMModel(self.input_shape)
        except Exception as e:
            logger.error("Model initialization failed", error=str(e))
            self.model_loaded = False
            return LSTMModel(self.input_shape)

    def _validate_model_file(self, path):
        """Check model file integrity and input shape compatibility."""
        try:
            if os.path.getsize(path) < 1024:
                logger.warning("Model file too small", size=os.path.getsize(path))
                return False

            temp_model = LSTMModel(self.input_shape)
            temp_model.load(path)
            if temp_model.model.input_shape[1:] != self.input_shape:
                logger.warning(
                    "Model shape mismatch",
                    expected=self.input_shape,
                    actual=temp_model.model.input_shape
                )
                return False

            return True
        except Exception as e:
            logger.warning("Model validation failed", error=str(e))
            return False

    async def _check_data_availability(self):
        """Check data readiness with fixed time window"""
        if not self.data_ready:
            if not self.warmup_start_time:
                # Initialize warmup parameters
                self.warmup_start_time = time.time()
                self.warmup_data_start = datetime.utcnow()  # Fixed start point
                logger.info("Warmup phase started")
                self.progress_bar = ProgressBar(total=self.min_samples)

            # Always use initial warmup start time for queries
            records = await Database.fetchval('''
                SELECT COUNT(DISTINCT time_bucket('1 minute', time)) 
                FROM market_data 
                WHERE time > $1
            ''', self.warmup_data_start)

            current_count = records or 0
            elapsed_time = time.time() - self.warmup_start_time

            # Update progress
            self.progress_bar.update(current_count)

            # Completion check
            if current_count >= self.min_samples or elapsed_time >= Config.MODEL_CONFIG['warmup_period']:
                self.data_ready = True
                self.progress_bar.update(self.min_samples)  # Force 100%
                logger.info("Warmup requirements met", 
                        samples=current_count,
                        duration=elapsed_time)
                return True

            # Log current state
            logger.info("Warmup status",
                    samples=current_count,
                    required=self.min_samples,
                    elapsed=f"{elapsed_time:.1f}s",
                    remaining=f"{Config.MODEL_CONFIG['warmup_period']-elapsed_time:.1f}s")
            return False
        
        return True

    async def get_historical_data(self):
        """Fetch and process historical data with buffer for feature creation."""
        try:
            # Fetch 2x min_samples to account for feature creation drops
            records = await Database.fetch('''
                SELECT 
                    time_bucket('1 minute', time) AS bucket,
                    FIRST(price, time) AS open,
                    MAX(price) AS high,
                    MIN(price) AS low,
                    LAST(price, time) AS close,
                    SUM(volume) AS volume
                FROM market_data
                GROUP BY bucket
                ORDER BY bucket DESC
                LIMIT $1
            ''', self.min_samples * 2)

            if not records:
                return pd.DataFrame()

            df = pd.DataFrame([dict(r) for r in records])
            df = df.astype({
                'open': 'float64',
                'high': 'float64',
                'low': 'float64',
                'close': 'float64',
                'volume': 'float64'
            })
            df['bucket'] = pd.to_datetime(df['bucket'])
            df.set_index('bucket', inplace=True)
            df.sort_index(ascending=True, inplace=True)

            df = self.preprocessor.create_features(df)
            logger.info("Feature creation complete", samples=len(df))

            if len(df) < self.min_samples:
                logger.warning("Insufficient data after feature creation",
                            initial=len(records),
                            processed=len(df))

            return df

        except Exception as e:
            logger.error("Data retrieval failed", error=str(e))
            return pd.DataFrame()
    
    async def retrain_model(self):
        """Robust model retraining with fallbacks."""
        try:
            df = await self.get_historical_data()
            logger.info("Retraining model", input_shape=self.input_shape, data_samples=len(df))
            if len(df) < Config.MODEL_CONFIG['min_training_samples']:
                logger.warning("Insufficient processed data",
                            available=len(df),
                            required=Config.MODEL_CONFIG['min_training_samples'])
                return

            # Adjust lookback window dynamically
            new_lookback = Config.MODEL_CONFIG['lookback_window']
            if len(df) < new_lookback * 2:
                new_lookback = min(30, len(df) // 2)
                logger.info("Adjusting lookback window", old=Config.MODEL_CONFIG['lookback_window'], new=new_lookback)
            
            # Reinitialize model with the correct input shape
            self.input_shape = (new_lookback, len(self.preprocessor.required_columns))
            self.model = LSTMModel(self.input_shape)  # Rebuild model

            # Prepare data with the new lookback window
            X, y = self.preprocessor.prepare_training_data(df)
            if X.shape[0] == 0:
                raise ValueError("Empty training data after preprocessing")

            self.model.train(X, y)
            self.model.save("model_storage/lstm_model.keras")
            self.model_loaded = True
            self.last_retrain = time.time()
            logger.info("Model retrained successfully", training_samples=X.shape[0])
        except Exception as e:
            logger.error("Retraining failed", error=str(e))
            self.model_loaded = False
                        
    async def get_prediction(self):
        """Generate predictions with shape validation."""
        try:
            if not self.model_loaded:
                logger.warning("Model not loaded, skipping prediction.")
                return None

            df = await self.get_historical_data()
            if df.empty:
                logger.warning("No historical data available for prediction.")
                return None

            # Prepare data with the correct lookback window
            X, _ = self.preprocessor.prepare_prediction_data(df)
            if X.shape[1:] != self.input_shape:
                logger.error(
                    "Prediction input shape mismatch",
                    expected=self.input_shape,
                    actual=X.shape[1:]
                )
                return None

            pred = self.model.predict(X)
            return pred.flatten()
        except Exception as e:
            logger.error("Prediction failed", error=str(e))
            return None
        
    async def execute_trades(self):
        """Execute trades based on model predictions and risk parameters"""
        try:
            if not self.model_loaded:
                logger.warning("Skipping trades - model not loaded")
                return

            # Check for existing position
            if self.current_position is not None:
                logger.info("Skipping trade - existing position", 
                            position=self.current_position)
                return

            # Get prediction and current price
            prediction = await self.get_prediction()
            current_price = await self.trade_service.get_current_price()
            
            if prediction is None or current_price is None:
                return

            # Calculate risk parameters based on direction
            if prediction > current_price * 1.02:  # Long signal
                await self._execute_trade(
                    side='Buy',
                    price=current_price,
                    stop_loss=current_price * 0.97,
                    take_profit=current_price * 1.05
                )

            elif prediction < current_price * 0.98:  # Short signal
                await self._execute_trade(
                    side='Sell',
                    price=current_price,
                    stop_loss=current_price * 1.03,
                    take_profit=current_price * 0.95
                )

        except Exception as e:
            logger.error("Trade execution error", error=str(e))

    async def _execute_trade(self, side: str, price: float, stop_loss: float, take_profit: float):
        """Implementation added here"""
        try:
            logger.info(
                "Attempting to execute trade",
                side=side,
                price=price,
                stop_loss=stop_loss,
                take_profit=take_profit
            )
            
            await self.trade_service.execute_trade(
                side=side,
                price=price,
                stop_loss=stop_loss,
                take_profit=take_profit
            )
            
            self.current_position = side.lower()
            self.last_trade_time = time.time()
            logger.info("Trade executed successfully", position=self.current_position)
            
        except Exception as e:
            logger.error("Trade execution failed", error=str(e))
            self.current_position = None
            raise


    def _should_retrain(self):
        """Determine if the model should be retrained based on time since last training."""
        if not self.last_retrain:
            logger.info("No previous retrain; should retrain now.")
            return True
        elapsed = time.time() - self.last_retrain
        logger.info("Time since last retrain", 
                    elapsed=elapsed, 
                    retrain_interval=Config.MODEL_CONFIG['retrain_interval'])
        return elapsed > Config.MODEL_CONFIG['retrain_interval']
    
    async def log_market_analysis(self):
        """Log market analysis including sentiment and predictions."""
        try:
            sentiment = await self.fetch_sentiment()
            prediction = await self.get_prediction()
            logger.info(
                "Market Snapshot",
                sentiment_score=sentiment,
                last_prediction=float(prediction[0]) if prediction is not None and len(prediction) else None,
                model_confidence=float(np.std(prediction)) if prediction is not None and len(prediction) else None
            )
        except Exception as e:
            logger.error("Market analysis failed", error=str(e))

    async def fetch_sentiment(self):
        """Fetch sentiment data (placeholder for real implementation)."""
        try:
            # Placeholder for actual sentiment analysis
            return np.random.uniform(-1, 1)
        except Exception as e:
            logger.error("Failed to fetch sentiment", error=str(e))
            return 0.0