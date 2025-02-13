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
        """Initialize or load LSTM model with validation."""
        model_path = "model_storage/lstm_model.keras"
        try:
            if os.path.exists(model_path) and self._validate_model_file(model_path):
                logger.info("Loading validated model", path=model_path)
                model = LSTMModel(self.input_shape)
                model.load(model_path)
                self.model_loaded = True
                logger.info("Model loaded successfully")
                return model
            else:
                logger.warning("No valid model found, initializing new model")
                return LSTMModel(self.input_shape)
        except Exception as e:
            logger.error("Model initialization failed", error=str(e))
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
        """Check data readiness by ensuring enough aggregated bars."""
        if not self.data_ready:
            if not self.warmup_start_time:
                self.warmup_start_time = time.time()
                logger.info("Warmup phase started")

            count_result = await Database.fetch(
                "SELECT COUNT(DISTINCT time_bucket('1 minute', time)) as count FROM market_data"
            )
            final_count = count_result[0]['count'] if count_result else 0

            self.progress_bar.update(final_count)

            data_progress = final_count / self.min_samples
            elapsed = time.time() - self.warmup_start_time
            time_progress = elapsed / Config.MODEL_CONFIG['warmup_period']

            overall_progress = min(data_progress, time_progress) * 100

            logger.info(
                "Warmup Progress",
                aggregator_rows=final_count,
                required_rows=self.min_samples,
                data_progress=f"{data_progress*100:.1f}%",
                time_elapsed=int(elapsed),
                time_required=Config.MODEL_CONFIG['warmup_period'],
                time_progress=f"{time_progress*100:.1f}%",
                overall_progress=f"{overall_progress:.1f}%"
            )

            if final_count >= self.min_samples and elapsed >= Config.MODEL_CONFIG['warmup_period']:
                self.data_ready = True
                self.progress_bar.clear()
                logger.info("Warmup complete - Starting trading")
                return True

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
            if len(df) < Config.MODEL_CONFIG['min_training_samples']:
                logger.warning("Insufficient processed data",
                               available=len(df),
                               required=Config.MODEL_CONFIG['min_training_samples'])
                return

            # Adjust lookback window if needed
            if len(df) < Config.MODEL_CONFIG['lookback_window'] * 2:
                new_lookback = min(30, len(df) // 2)
                logger.info("Adjusting lookback window",
                             old=Config.MODEL_CONFIG['lookback_window'],
                             new=new_lookback)
                self.model.input_shape = (new_lookback, 12)

            X, y = self.preprocessor.prepare_data(df)
            if X.shape[0] == 0:
                raise ValueError("Empty training data after preprocessing")

            self.model.train(X, y)
            self.model.save("model_storage/lstm_model.keras")
            self.model_loaded = True
            logger.info("Model retrained successfully",
                         training_samples=X.shape[0])
    
        except Exception as e:
            logger.error("Retraining failed", error=str(e))
            if os.path.exists("model_storage/lstm_model.keras"):
                self.model.load("model_storage/lstm_model.keras")
                self.model_loaded = True

        finally:
                self.last_retrain = time.time()  # Update here to mark retrain time
                
    async def get_prediction(self):
        """Use the model to generate a next price prediction from aggregator data."""
        try:
            if not self.model_loaded:
                return None

            df = await self.get_historical_data()
            if df.empty:
                return None

            # Use the new prepare_data method to get X and y.
            X, _ = self.preprocessor.prepare_data(df)
            # Log the shape for debugging purposes
            logger.info("Prediction input shape", shape=X.shape)

            # Expecting X to have shape (n_samples, 60, 12). Take the last sample for prediction.
            if X.shape[0] == 0:
                return None

            pred = self.model.predict(X[-1:]).flatten()
            return pred
        except Exception as e:
            logger.error("Prediction failed", error=str(e))
            return None


    async def execute_trades(self):
        """Execute trades with dynamic thresholds and SL/TP based on volatility."""
        try:
            prediction = await self.get_prediction()
            current_price = await self.trade_service.get_current_price()

            if prediction is None or current_price is None:
                return

            # Calculate dynamic thresholds based on volatility from historical close prices.
            df = await self.get_historical_data()
            # Calculate volatility as the standard deviation of percent changes
            volatility = df['close'].pct_change().std()
            
            # Define dynamic thresholds (you may adjust the multiplier as needed)
            buy_threshold = 1 + (volatility * 1.5)
            sell_threshold = 1 - (volatility * 1.5)

            # Determine stop loss and take profit levels.
            # For a Buy trade:
            #   SL is set below the current price by one volatility unit.
            #   TP is set above the current price by two volatility units.
            # For a Sell trade, the levels are reversed.
            if prediction > current_price * buy_threshold:
                stop_loss = current_price * (1 - volatility)   # e.g. if volatility is 2%, SL is 98% of current price.
                take_profit = current_price * (1 + volatility * 2) # e.g. TP is 104% if volatility is 2%
                await self._execute_trade('Buy', current_price, stop_loss=stop_loss, take_profit=take_profit)
            elif prediction < current_price * sell_threshold:
                stop_loss = current_price * (1 + volatility)   # For sell, a stop loss above current price.
                take_profit = current_price * (1 - volatility * 2) # Take profit below current price.
                await self._execute_trade('Sell', current_price, stop_loss=stop_loss, take_profit=take_profit)

        except Exception as e:
            logger.error("Trade execution failed", error=str(e))

    async def _execute_trade(self, side: str, price: float, stop_loss: float = None, take_profit: float = None):
        """
        Validate the calculated position size before trading.
        If the position size meets the minimum requirement, execute the trade
        with optional stop loss (SL) and take profit (TP) levels.
        
        :param side: 'Buy' or 'Sell'
        :param price: Current market price
        :param stop_loss: Optional stop loss level
        :param take_profit: Optional take profit level
        """
        position_size = self._calculate_position_size(price)
        if position_size >= self.trade_service.min_qty:
            logger.info(
                "Executing trade",
                side=side,
                price=price,
                position_size=position_size,
                stop_loss=stop_loss,
                take_profit=take_profit
            )
            await self.trade_service.execute_trade(
                price, side, position_size, stop_loss=stop_loss, take_profit=take_profit
            )
        else:
            logger.warning(
                "Position size below minimum",
                calculated=position_size,
                required=self.trade_service.min_qty
            )


    def _calculate_position_size(self, current_price):
        """Risk-based position sizing."""
        account_balance = 10000  # placeholder
        risk_percent = 0.02
        dollar_risk = account_balance * risk_percent
        return min(dollar_risk / current_price, Config.TRADING_CONFIG['position_size'])

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

    def _should_retrain(self):
        if not self.last_retrain:
            logger.info("No previous retrain; should retrain now.")
            return True
        elapsed = time.time() - self.last_retrain
        logger.info("Time since last retrain", elapsed=elapsed, retrain_interval=Config.MODEL_CONFIG['retrain_interval'])
        return elapsed > Config.MODEL_CONFIG['retrain_interval']


    async def log_market_analysis(self):
        """Periodic logging of sentiment & model predictions."""
        if time.time() - self.last_log_time >= 300:
            try:
                sentiment = await self.fetch_sentiment()
                prediction = await self.get_prediction()
                logger.info(
                    "Market Snapshot",
                    sentiment_score=sentiment,
                    last_prediction=float(prediction[0]) if prediction is not None and len(prediction) else None,
                    model_confidence=float(np.std(prediction)) if prediction is not None and len(prediction) else None
                )
                self.last_log_time = time.time()
            except Exception as e:
                logger.error("Market analysis failed", error=str(e))

    async def fetch_sentiment(self):
        """Placeholder for real sentiment data."""
        return np.random.uniform(-1, 1)