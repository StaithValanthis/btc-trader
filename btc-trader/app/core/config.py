# app/core/config.py
from dotenv import load_dotenv
import os
from structlog import get_logger

logger = get_logger(__name__)

load_dotenv()

class Config:
    DB_CONFIG = {
        'host': os.getenv('DB_HOST', 'postgres'),
        'database': os.getenv('DB_NAME', 'trading_bot'),
        'user': os.getenv('DB_USER', 'postgres'),
        'password': os.getenv('DB_PASSWORD', 'postgres')
    }
    
    BYBIT_CONFIG = {
        'api_key': os.getenv('BYBIT_API_KEY', ''),
        'api_secret': os.getenv('BYBIT_API_SECRET', ''),
        'testnet': os.getenv('BYBIT_TESTNET', 'false').lower() == 'true'
    }
    
    TRADING_CONFIG = {
        'symbol': os.getenv('SYMBOL', 'BTCUSDT'),
        'position_size': float(os.getenv('POSITION_SIZE', 0.001)),
        'max_leverage': float(os.getenv('MAX_LEVERAGE', 10)),
        'retrain_interval': 86400  # Retrain every 24 hours
    }
    
    MODEL_CONFIG = {
        'lookback_window': 60,
        'min_training_samples': 500,
        'train_epochs': 50,
        'batch_size': 32,
        'warmup_period': 1800,  # 30 minutes
        'retrain_interval': 86400  # Retrain every 24 hours
    }