import asyncio
import time
import pandas as pd
import numpy as np
from structlog import get_logger
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
        input_shape = (Config.MODEL_CONFIG['lookback_window'], 10)  # 10 features
        self.model = LSTMModel(input_shape)
        self.last_retrain = 0
        self.data_ready = False
        self.warmup_start_time = None

    async def _check_data_availability(self):
        """Check data readiness with progress tracking."""
        if not self.data_ready:
            if not self.warmup_start_time:
                self.warmup_start_time = time.time()
                logger.info("Warmup phase started")

            # Get data progress
            count = await Database.fetch("SELECT COUNT(*) FROM market_data")
            data_count = count[0]['count']
            data_progress = data_count / Config.MODEL_CONFIG['min_training_samples']
            
            # Get time progress
            elapsed = time.time() - self.warmup_start_time
            time_progress = elapsed / Config.MODEL_CONFIG['warmup_period']
            
            # Calculate overall progress
            overall_progress = min(data_progress, time_progress) * 100
            time_remaining = max(0, (Config.MODEL_CONFIG['warmup_period'] - elapsed)/60)

            # Log progress
            logger.info(
                "Warmup Status",
                progress=progress_bar(overall_progress),
                data=f"{data_count}/{Config.MODEL_CONFIG['min_training_samples']}",
                time_remaining=f"{time_remaining:.1f} minutes"
            )

            if data_count >= Config.MODEL_CONFIG['min_training_samples'] and \
               elapsed >= Config.MODEL_CONFIG['warmup_period']:
                self.data_ready = True
                logger.info("Warmup complete - Starting trading")
                return True
            return False
        return True

    async def get_historical_data(self):
        """Fetch time-bucketed data using TimescaleDB."""
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
            LIMIT 1000
        ''')
        return pd.DataFrame([dict(r) for r in records])

    async def retrain_model(self):
        """Retrain the LSTM model."""
        try:
            if not await self._check_data_availability():
                return
                
            df = await self.get_historical_data()
            X, y = self.preprocessor.prepare_data(df)
            
            if len(X) > 0 and len(y) > 0:
                self.model.train(X, y)
                self.last_retrain = time.time()
                logger.info("Model retrained successfully")
                
        except Exception as e:
            logger.error("Retraining failed", error=str(e))

    async def get_prediction(self):
        """Generate predictions using the LSTM model."""
        try:
            df = await self.get_historical_data()
            X, _ = self.preprocessor.prepare_data(df)
            return self.model.predict(X[-1:]).flatten() if len(X) > 0 else None
        except Exception as e:
            logger.error("Prediction failed", error=str(e))
            return None

    async def execute_trades(self):
        """Execute trades based on LSTM predictions."""
        if not await self._check_data_availability():
            return

        if time.time() - self.last_retrain > Config.TRADING_CONFIG['retrain_interval']:
            await self.retrain_model()

        prediction = await self.get_prediction()
        if prediction is None:
            return

        current_price = await self.trade_service.get_current_price()
        avg_prediction = np.mean(prediction)
        
        if avg_prediction > current_price * 1.005:
            await self.trade_service.execute_trade(current_price, 'Buy')
        elif avg_prediction < current_price * 0.995:
            await self.trade_service.execute_trade(current_price, 'Sell')

    async def run(self):
        """Run the trading strategy."""
        while True:
            try:
                await self.execute_trades()
                await asyncio.sleep(60)
            except Exception as e:
                logger.error("Strategy error", error=str(e))
                await asyncio.sleep(10)