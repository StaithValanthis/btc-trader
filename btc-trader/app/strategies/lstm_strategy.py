import asyncio
import time
import pandas as pd
import numpy as np
from structlog import get_logger
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from app.core import Config, Database
from app.services.trade_service import TradeService
from app.ml.data_preprocessor import DataPreprocessor
from app.ml.lstm_model import LSTMModel
from app.utils.progress import progress_bar

logger = get_logger(__name__)

class LSTMStrategy:
    def __init__(self, trade_service: TradeService):
        self.trade_service = trade_service
        self.preprocessor = DataPreprocessor()
        input_shape = (Config.MODEL_CONFIG['lookback_window'], 12)  # 12 features including indicators
        self.model = LSTMModel(input_shape)
        self.analyzer = SentimentIntensityAnalyzer()
        self.last_retrain = None  # Ensure first training occurs immediately
        self.data_ready = False
        self.warmup_start_time = None
        self.last_log_time = time.time() - 300  # Ensure logging occurs every 5 minutes

    async def fetch_sentiment(self):
        """Fetch sentiment data from external API or local processing."""
        sentiment_score = np.random.uniform(-1, 1)  # Placeholder for actual sentiment data
        return sentiment_score

    async def get_sentiment_score(self, news_text):
        """Analyze market sentiment based on news or social media text."""
        return self.analyzer.polarity_scores(news_text)['compound']

    async def get_prediction(self):
        """Generate predictions using the LSTM model with all available indicators."""
        try:
            df = await self.get_historical_data()
            X, _ = self.preprocessor.prepare_data(df)  # Use all indicators for prediction
            return self.model.predict(X[-1:]).flatten() if len(X) > 0 else None
        except Exception as e:
            logger.error("Prediction failed", error=str(e))
            return None

    async def get_historical_data(self):
        """Fetch time-bucketed data using TimescaleDB."""
        records = await Database.fetch('''
            SELECT 
                time_bucket('1 minute', time) AS bucket,
                FIRST(price, time) AS open,
                MAX(price) AS high,
                MIN(price) AS low,
                LAST(price, time) AS close,
                SUM(volume) AS volume,
                AVG(rsi) AS rsi,
                AVG(macd) AS macd,
                AVG(signal) AS signal,
                AVG(volatility) AS volatility
            FROM market_data
            GROUP BY bucket
            ORDER BY bucket DESC
            LIMIT 1000
        ''')
        df = pd.DataFrame([dict(r) for r in records])
        return self.preprocessor.create_features(df)  # Ensure features are added before use

    async def log_market_analysis(self):
        """Log market sentiment and prediction every 5 minutes."""
        current_time = time.time()
        if current_time - self.last_log_time >= 300:
            sentiment_score = await self.fetch_sentiment()
            prediction = await self.get_prediction()
            logger.info(
                "Market Analysis Update",
                sentiment_score=sentiment_score,
                prediction=prediction
            )
            self.last_log_time = current_time

    async def _check_data_availability(self):
        """Check if enough data is available before training or trading."""
        if not self.data_ready:
            if not self.warmup_start_time:
                self.warmup_start_time = time.time()
                logger.info("Warmup phase started")
            
            count = await Database.fetch("SELECT COUNT(*) FROM market_data")
            data_count = count[0]['count']
            data_progress = min(1, data_count / Config.MODEL_CONFIG['min_training_samples'])
            elapsed = time.time() - self.warmup_start_time
            time_progress = min(1, elapsed / Config.MODEL_CONFIG['warmup_period'])
            
            overall_progress = min(data_progress, time_progress) * 100  # Ensure progress reaches 100% correctly
            time_remaining = max(0, (Config.MODEL_CONFIG['warmup_period'] - elapsed) / 60)
            progress_bar_str = progress_bar(overall_progress)
            
            logger.info(
                "Warmup Status",
                progress=progress_bar_str,
                data=f"{data_count}/{Config.MODEL_CONFIG['min_training_samples']}",
                time_remaining=f"{time_remaining:.1f} minutes"
            )
            
            if data_progress >= 1 and elapsed >= Config.MODEL_CONFIG['warmup_period']:
                self.data_ready = True
                logger.info("Warmup complete - Starting trading")
                await self.retrain_model()  # Ensure first training happens immediately
                return True
            return False
        return True

    async def retrain_model(self):
        """Retrain the LSTM model and log training status only when retraining occurs."""
        try:
            if not self.data_ready:
                return  # Ensure warmup is complete before retraining
            
            df = await self.get_historical_data()
            data_available = len(df)
            required_data = Config.MODEL_CONFIG['min_training_samples']
            
            if data_available < required_data:
                return  # Skip logging unless retraining actually happens
            
            current_time = time.time()
            if self.last_retrain is None or current_time - self.last_retrain >= 3600:  # First run or 1-hour interval
                logger.info("Starting model retraining...")
                X, y = self.preprocessor.prepare_data(df)
                if len(X) > 0 and len(y) > 0:
                    self.model.train(X, y)
                    self.last_retrain = time.time()
                    logger.info("Model retrained successfully")
                else:
                    logger.warning("Training data preparation failed - Not enough valid samples")
        except Exception as e:
            logger.error("Retraining failed", error=str(e))

    async def execute_trades(self):
        """Execute trades based on LSTM predictions using all indicators."""
        try:
            df = await self.get_historical_data()
            if df.empty:
                return
            
            prediction = await self.get_prediction()
            if prediction is None:
                return
            
            current_price = await self.trade_service.get_current_price()
            avg_prediction = np.mean(prediction)
            
            if avg_prediction > current_price and avg_prediction > df['close'].mean():
                await self.trade_service.execute_trade(current_price, 'Buy')
            elif avg_prediction < current_price and avg_prediction < df['close'].mean():
                await self.trade_service.execute_trade(current_price, 'Sell')
        except Exception as e:
            logger.error("Trade execution failed", error=str(e))

    async def run(self):
        """Run the trading strategy."""
        while True:
            try:
                await self._check_data_availability()
                if self.data_ready:
                    await self.retrain_model()
                    await self.execute_trades()
                    await self.log_market_analysis()  # Log sentiment and prediction every 5 minutes
                await asyncio.sleep(60)
            except Exception as e:
                logger.error("Strategy error", error=str(e))
                await asyncio.sleep(10)
