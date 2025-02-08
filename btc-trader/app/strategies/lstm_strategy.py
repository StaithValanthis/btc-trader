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
from app.utils.progress import ProgressBar, progress_bar

logger = get_logger(__name__)

class LSTMStrategy:
    def __init__(self, trade_service: TradeService):
        self.trade_service = trade_service
        self.preprocessor = DataPreprocessor()
        self.input_shape = (Config.MODEL_CONFIG['lookback_window'], 12)  # 12 features
        self.model = self._initialize_model()
        self.analyzer = SentimentIntensityAnalyzer()
        self.last_retrain = None
        self.data_ready = False
        self.warmup_start_time = None
        self.last_log_time = time.time() - 300
        self.model_loaded = False
        self.min_samples = Config.MODEL_CONFIG['min_training_samples']
        self.progress_bar = ProgressBar(total=self.min_samples)

    def _initialize_model(self):
        """Initialize or load LSTM model with validation"""
        model_path = "lstm_model.h5"
        try:
            if os.path.exists(model_path):
                logger.info("Model file found", path=model_path)
                
                # Verify model compatibility
                if self._validate_model_file(model_path):
                    model = LSTMModel(self.input_shape)
                    model.load(model_path)
                    self.model_loaded = True
                    logger.info("Model loaded successfully")
                    return model
                
            logger.warning("Initializing new model")
            return LSTMModel(self.input_shape)
            
        except Exception as e:
            logger.error("Model initialization failed", error=str(e))
            return LSTMModel(self.input_shape)

    def _validate_model_file(self, path):
        """Check model file integrity and compatibility"""
        try:
            # Check file size
            if os.path.getsize(path) < 1024:
                logger.warning("Model file too small", size=os.path.getsize(path))
                return False
                
            # Check input shape compatibility
            temp_model = LSTMModel(self.input_shape)
            temp_model.load(path)
            if temp_model.model.input_shape[1:] != self.input_shape:
                logger.warning("Model shape mismatch", 
                              expected=self.input_shape,
                              actual=temp_model.model.input_shape)
                return False
                
            return True
            
        except Exception as e:
            logger.warning("Model validation failed", error=str(e))
            return False

    async def get_historical_data(self):
        """Fetch and prepare training data with validation"""
        try:
            # Get data from last 4 hours to ensure freshness
            time_threshold = datetime.utcnow() - timedelta(hours=4)
            
            records = await Database.fetch('''
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
                ORDER BY bucket DESC
                LIMIT $2
            ''', time_threshold, self.min_samples * 2)  # Get extra samples for cleaning
            
            if not records:
                logger.warning("No historical data available")
                return pd.DataFrame()
                
            # Create DataFrame with proper types
            df = pd.DataFrame([dict(r) for r in records])
            df = df.astype({
                'open': 'float64',
                'high': 'float64',
                'low': 'float64',
                'close': 'float64',
                'volume': 'float64'
            })
            
            # Set index and sort
            df['bucket'] = pd.to_datetime(df['bucket'])
            df.set_index('bucket', inplace=True)
            df.sort_index(ascending=True, inplace=True)
            
            logger.info("Raw data retrieved", samples=len(df))
            return df
            
        except Exception as e:
            logger.error("Data retrieval failed", error=str(e))
            return pd.DataFrame()

    async def _validate_data(self, df):
        """Validate data quality before processing"""
        if df.empty:
            return False
            
        # Check for required columns
        required = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required):
            logger.warning("Missing required columns", 
                          missing=set(required) - set(df.columns))
            return False
            
        # Check for sufficient data points
        if len(df) < self.min_samples:
            logger.warning("Insufficient data points", 
                          available=len(df),
                          required=self.min_samples)
            return False
            
        # Check for NaN values
        if df[required].isnull().any().any():
            logger.warning("Data contains NaN values")
            return False
            
        return True

    async def _check_data_availability(self):
        """Comprehensive data readiness check"""
        if not self.data_ready:
            if not self.warmup_start_time:
                self.warmup_start_time = time.time()
                logger.info("Warmup phase started")

            try:
                # Get and validate data
                df = await self.get_historical_data()
                if not await self._validate_data(df):
                    return False
                    
                # Prepare features
                features_df = self.preprocessor.create_features(df)
                if features_df.empty:
                    logger.warning("Feature engineering failed")
                    return False
                    
                # Track progress
                valid_samples = len(features_df)
                total_samples = await self._get_total_data_count()
                
                time_progress = ((time.time() - self.warmup_start_time) / 
                               Config.MODEL_CONFIG['warmup_period']) * 100
                data_progress = (valid_samples / self.min_samples) * 100
                overall_progress = min(data_progress, time_progress)
                
                # Update static progress bar
                self.progress_bar.update(valid_samples)
                
                if valid_samples >= self.min_samples and time_progress >= 100:
                    self.data_ready = True
                    self.progress_bar.clear()  # Clear the progress bar when done
                    logger.info("Warmup complete - Starting trading")
                    return True
                    
            except Exception as e:
                logger.error("Data check failed", error=str(e))
                self.progress_bar.clear()
                
            return False
        return True

    async def _get_total_data_count(self):
        """Get total raw data points in database"""
        result = await Database.fetch("SELECT COUNT(*) FROM market_data")
        return result[0]['count'] if result else 0

    async def retrain_model(self):
        """Robust model retraining with atomic operations"""
        temp_path = None
        try:
            logger.info("Starting model retraining cycle")
            
            # 1. Data Collection
            df = await self.get_historical_data()
            if len(df) < self.min_samples:
                logger.warning("Skipping retrain - insufficient data",
                              available=len(df),
                              required=self.min_samples)
                return
                
            # 2. Data Preparation
            X, y = self.preprocessor.prepare_data(df)
            if X.shape[0] == 0:
                logger.error("Data preparation failed")
                return
                
            # 3. Model Training
            logger.info("Training model", samples=X.shape[0])
            self.model.train(X, y)
            
            # 4. Model Saving
            temp_path = "lstm_model_temp.h5"
            final_path = "lstm_model.h5"
            
            self.model.save(temp_path)
            
            # Verify saved model
            if not os.path.exists(temp_path) or os.path.getsize(temp_path) < 1024:
                raise IOError("Model save verification failed")
                
            # Atomic replacement
            if os.path.exists(final_path):
                os.remove(final_path)
            os.rename(temp_path, final_path)
            
            # Final validation
            if self._validate_model_file(final_path):
                self.model_loaded = True
                logger.info("Model update successful")
            else:
                raise ValueError("Model validation failed after save")
                
        except Exception as e:
            logger.error("Retraining cycle failed", error=str(e))
            self.model_loaded = False
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception as e:
                    logger.error("Temp file cleanup failed", error=str(e))
        finally:
            self.last_retrain = time.time()

    async def execute_trades(self):
        """Trade execution with model validation"""
        try:
            if not self.model_loaded:
                logger.warning("Skipping trades - model not loaded")
                return
                
            prediction = await self.get_prediction()
            if prediction is None:
                return
                
            current_price = await self.trade_service.get_current_price()
            if current_price is None:
                return
                
            # Calculate position sizing
            position_size = self._calculate_position_size(current_price)
            
            if prediction > current_price * 1.002:
                await self.trade_service.execute_trade(current_price, 'Buy', position_size)
            elif prediction < current_price * 0.998:
                await self.trade_service.execute_trade(current_price, 'Sell', position_size)
                
        except Exception as e:
            logger.error("Trade execution error", error=str(e))

    def _calculate_position_size(self, current_price):
        """Risk-adjusted position sizing"""
        account_balance = 10000  # Should be fetched from exchange
        risk_percent = 0.02
        dollar_risk = account_balance * risk_percent
        return min(dollar_risk / current_price, Config.TRADING_CONFIG['position_size'])

    async def run(self):
        """Main strategy loop"""
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

    def _should_retrain(self):
        """Determine if retraining is needed"""
        if not self.last_retrain:
            return True
        return (time.time() - self.last_retrain) > Config.MODEL_CONFIG['retrain_interval']

    async def log_market_analysis(self):
        """Periodic market analysis logging"""
        if time.time() - self.last_log_time >= 300:
            try:
                sentiment = await self.fetch_sentiment()
                prediction = await self.get_prediction()
                logger.info(
                    "Market Snapshot",
                    sentiment_score=sentiment,
                    last_prediction=float(prediction[0]) if prediction else None,
                    model_confidence=float(np.std(prediction)) if prediction else None
                )
                self.last_log_time = time.time()
            except Exception as e:
                logger.error("Market analysis failed", error=str(e))

    async def fetch_sentiment(self):
        """Fetch market sentiment data"""
        # Implement actual sentiment API call here
        return np.random.uniform(-1, 1)

    async def get_prediction(self):
        """Get model prediction with validation"""
        try:
            if not self.model_loaded:
                return None
                
            df = await self.get_historical_data()
            if df.empty:
                return None
                
            X, _ = self.preprocessor.prepare_data(df)
            return self.model.predict(X[-1:]).flatten()
        except Exception as e:
            logger.error("Prediction failed", error=str(e))
            return None