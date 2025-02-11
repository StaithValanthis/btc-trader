import asyncio
import pandas as pd
import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from structlog import get_logger
from datetime import datetime, timezone
from pybit.unified_trading import HTTP
from app.core.database import Database
from app.core.config import Config

logger = get_logger(__name__)

class MLTradeService:
    def __init__(self):
        self.model = self.load_model()
        self.running = False
        self.session = HTTP(
            testnet=Config.BYBIT_CONFIG['testnet'],
            api_key=Config.BYBIT_CONFIG['api_key'],
            api_secret=Config.BYBIT_CONFIG['api_secret']
        )
    
    def load_model(self):
        """Load the trained LSTM model or train a new one."""
        try:
            return tf.keras.models.load_model("lstm_model.h5")
        except (OSError, FileNotFoundError):
            logger.warning("No existing LSTM model found, training a new one.")
            return self.train_model()
    
    async def get_market_data(self, limit: int = 100):
        """Fetch market data from database."""
        try:
            records = await Database.fetch('''
                SELECT time, price 
                FROM market_data 
                ORDER BY time DESC 
                LIMIT $1
            ''', limit)
            
            return pd.DataFrame([dict(record) for record in records])
        except Exception as e:
            logger.error("Failed to fetch market data", error=str(e))
            raise
    
    def train_model(self):
        """Train a new LSTM model."""
        data = asyncio.run(self.get_market_data(500))
        if data.empty:
            logger.error("Not enough data to train the model.")
            return None
        
        data['price_lag'] = data['price'].shift(1)
        data.dropna(inplace=True)
        
        X, y = self.prepare_lstm_data(data['price'].values)
        
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X, y, epochs=10, batch_size=16)
        
        model.save("lstm_model.h5")
        logger.info("LSTM model trained and saved.")
        return model
    
    def prepare_lstm_data(self, price_series, time_steps=10):
        """Prepare data for LSTM training."""
        X, y = [], []
        for i in range(len(price_series) - time_steps):
            X.append(price_series[i:i + time_steps])
            y.append(price_series[i + time_steps])
        return np.array(X).reshape(-1, time_steps, 1), np.array(y)
    
    async def predict_next_price(self):
        """Predict the next price using the trained LSTM model."""
        market_data = await self.get_market_data(10)
        if len(market_data) < 10:
            logger.warning("Not enough data for prediction.")
            return None
        
        latest_prices = market_data['price'].values.reshape(1, -1, 1)
        prediction = self.model.predict(latest_prices)[0][0]
        logger.info(f"Predicted next price (LSTM): {prediction}")
        return prediction
    
    async def execute_trade_if_profitable(self):
        """Decide whether to buy or sell based on predicted price."""
        predicted_price = await self.predict_next_price()
        if predicted_price is None:
            return
        
        market_data = await self.get_market_data(1)
        current_price = market_data['price'].iloc[-1]
        
        if predicted_price > current_price * 1.002:  # 0.2% threshold
            await self.execute_trade(current_price, "BUY")
        elif predicted_price < current_price * 0.998:
            await self.execute_trade(current_price, "SELL")
    
    async def get_open_orders(self):
        """Fetch current open orders."""
        try:
            response = await asyncio.to_thread(
                self.session.get_open_orders,
                category="linear",
                symbol="BTCUSDT"
            )
            return response.get('result', {}).get('list', [])
        except Exception as e:
            logger.error("Failed to fetch open orders", error=str(e))
            raise
    
    async def get_position_info(self):
        """Get current position information."""
        try:
            response = await asyncio.to_thread(
                self.session.get_positions,
                category="linear",
                symbol="BTCUSDT"
            )
            return response.get('result', {}).get('list', [])
        except Exception as e:
            logger.error("Failed to fetch position info", error=str(e))
            raise
    
    async def stop(self):
        self.running = False
        logger.warning("LSTM Trading Service Stopped")

class LSTMStrategy:
    def __init__(self, trade_service: TradeService):
        # ... existing code ...
        self.warmup_start_time = None  # Track warmup start time

    async def _check_data_availability(self):
        """Check data readiness with progress tracking."""
        if not self.data_ready:
            if not self.warmup_start_time:
                self.warmup_start_time = time.time()
                logger.info("Warmup phase started")

            # Data progress
            count = await Database.fetch("SELECT COUNT(*) FROM market_data")
            data_progress = count[0]['count'] / Config.MODEL_CONFIG['min_training_samples']
            
            # Time progress
            elapsed = time.time() - self.warmup_start_time
            time_progress = elapsed / Config.MODEL_CONFIG['warmup_period']
            
            # Combined progress
            overall_progress = min(data_progress, time_progress) * 100
            
            logger.info(
                "Warmup Progress",
                data=f"{count[0]['count']}/{Config.MODEL_CONFIG['min_training_samples']}",
                time_remaining=f"{int((Config.MODEL_CONFIG['warmup_period'] - elapsed)/60)}m",
                progress=f"{overall_progress:.1f}%"
            )
            
            if count[0]['count'] >= Config.MODEL_CONFIG['min_training_samples'] and \
               elapsed >= Config.MODEL_CONFIG['warmup_period']:
                self.data_ready = True
                logger.info("Warmup complete - Starting trading")
                return True
            return False
        return True