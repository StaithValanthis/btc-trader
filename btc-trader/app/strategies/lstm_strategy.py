import asyncio
import time
import pandas as pd
import numpy as np
from structlog import get_logger
from app.core import Config, Database
from app.services.trade_service import TradeService
from app.ml.data_preprocessor import DataPreprocessor
from app.ml.lstm_model import LSTMModel

logger = get_logger(__name__)

class LSTMStrategy:
    def __init__(self, trade_service: TradeService):
        self.trade_service = trade_service
        self.preprocessor = DataPreprocessor()
        input_shape = (Config.MODEL_CONFIG['lookback_window'], 4)
        self.model = LSTMModel(input_shape)
        self.last_retrain = 0

    async def get_historical_data(self):
        records = await Database.fetch(
            "SELECT time, price, volume FROM market_data "
            "ORDER BY time DESC LIMIT 1000"
        )
        return pd.DataFrame([dict(r) for r in records])

    async def retrain_model(self):
        try:
            df = await self.get_historical_data()
            X, y = self.preprocessor.prepare_data(df)
            if len(X) > 0:
                self.model.train(X, y)
                self.last_retrain = time.time()
        except Exception as e:
            logger.error("Retraining failed", error=str(e))

    async def get_prediction(self):
        try:
            df = await self.get_historical_data()
            if len(df) < Config.MODEL_CONFIG['lookback_window']:
                return None
                
            X, _ = self.preprocessor.prepare_data(df)
            prediction = self.model.predict(X[-1:])
            return prediction.flatten()
        except Exception as e:
            logger.error("Prediction failed", error=str(e))
            return None

    async def execute_trades(self):
        if time.time() - self.last_retrain > Config.TRADING_CONFIG['retrain_interval']:
            await self.retrain_model()

        prediction = await self.get_prediction()
        if prediction is None:
            return

        current_price = await self.trade_service.get_current_price()
        avg_prediction = np.mean(prediction)
        
        if avg_prediction > current_price * 1.005:  # 0.5% upside
            await self.trade_service.execute_trade(current_price, 'Buy')
        elif avg_prediction < current_price * 0.995:  # 0.5% downside
            await self.trade_service.execute_trade(current_price, 'Sell')

    async def run(self):
        while True:
            try:
                await self.execute_trades()
                await asyncio.sleep(60)
            except Exception as e:
                logger.error("Strategy error", error=str(e))
                await asyncio.sleep(10)