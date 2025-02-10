# app/strategies/lstm_strategy.py
import asyncio
import os
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from structlog import get_logger
from app.core import Database
from app.services.trade_service import TradeService
from app.ml.data_preprocessor import DataPreprocessor
from app.ml.lstm_model import LSTMModel
from app.core.config import Config

logger = get_logger(__name__)

class LSTMStrategy:
    def __init__(self, trade_service: TradeService):
        self.trade_service = trade_service
        self.preprocessor = DataPreprocessor()
        self.input_shape = (Config.MODEL_CONFIG['lookback_window'], 12)
        self.model = self._initialize_model()
        self.model_loaded = False
        self.last_retrain = None

    def _initialize_model(self):
        """Initialize or load LSTM model"""
        try:
            if os.path.exists("lstm_model.h5"):
                model = LSTMModel(self.input_shape)
                model.load("lstm_model.h5")
                self.model_loaded = True
                return model
            return LSTMModel(self.input_shape)
        except Exception as e:
            logger.error("Model init failed", error=str(e))
            return LSTMModel(self.input_shape)

    async def get_historical_data(self, hours=24):
        """Fetch historical market data"""
        try:
            time_threshold = datetime.utcnow() - timedelta(hours=hours)
            records = await Database.fetch(
                "SELECT time, price, volume FROM market_data WHERE time > $1",
                time_threshold
            )
            
            if not records:
                return pd.DataFrame()
                
            df = pd.DataFrame([dict(r) for r in records])
            df['time'] = pd.to_datetime(df['time'])
            return df.set_index('time').sort_index()
            
        except Exception as e:
            logger.error("Data retrieval failed", error=str(e))
            return pd.DataFrame()

    async def retrain_model(self):
        """Retrain the LSTM model with fresh data"""
        try:
            df = await self.get_historical_data(hours=24)
            if len(df) < Config.MODEL_CONFIG['min_training_samples']:
                return
                
            X, y = self.preprocessor.prepare_data(df)
            if X.shape[0] == 0:
                return
                
            self.model.train(X, y)
            self.model.save("lstm_model.h5")
            self.model_loaded = True
            self.last_retrain = time.time()
            
        except Exception as e:
            self.model_loaded = False
            logger.error("Retraining failed", error=str(e))

    async def get_prediction(self):
        """Get the next price prediction from the model"""
        try:
            df = await self.get_historical_data(hours=1)
            if df.empty:
                return None
                
            X, _ = self.preprocessor.prepare_data(df)
            return self.model.predict(X[-1:]).flatten()
        except Exception as e:
            logger.error("Prediction failed", error=str(e))
            return None

    async def run(self):
        """Main strategy loop"""
        logger.info("Strategy started")
        while True:
            try:
                if not self.model_loaded:
                    await self.retrain_model()
                    
                prediction = await self.get_prediction()
                current_price = await self.trade_service.get_current_price()
                
                if prediction and current_price:
                    if prediction > current_price * 1.002:
                        await self.execute_trade('Buy', current_price)
                    elif prediction < current_price * 0.998:
                        await self.execute_trade('Sell', current_price)
                        
                if time.time() - self.last_retrain > 6 * 3600:
                    await self.retrain_model()
                    
                await asyncio.sleep(60)
                
            except Exception as e:
                logger.error("Main loop error", error=str(e))
                await asyncio.sleep(10)

    async def execute_trade(self, side: str, price: float):
        """Execute a trade"""
        try:
            await self.trade_service.execute_trade(price, side)
        except Exception as e:
            logger.error("Trade execution failed", error=str(e))