import os
from dotenv import load_dotenv
from structlog import get_logger

logger = get_logger(__name__)

load_dotenv()

def get_env_variable(name, default):
    """Safely get environment variable with proper type conversion."""
    value = os.getenv(name)
    if value is None or value.strip() == '':
        logger.warning(f"Environment variable {name} is missing or empty, using default: {default}")
        return default
    try:
        return float(value)
    except ValueError:
        return value

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
        'position_size': get_env_variable('POSITION_SIZE', 0.001),
        'max_leverage': get_env_variable('MAX_LEVERAGE', 10),
    }
 
    MODEL_CONFIG = {
        'lookback_window': 60,
        'min_training_samples': 60,     # only need 15 bars to start
        'train_epochs': 20,            # reduce epochs for quicker training
        'batch_size': 16,
        'warmup_period': 1800,          # 15 minutes = 900s
        'retrain_interval': 86400,     # daily retraining
        'use_rolling_window': True,
        'rolling_window_hours': 1,  # 15 minutes
        'enable_hyperparam_tuning': False
    }    
    
    TIMESCALE_CONFIG = {
        'compression_interval': '7 days',
        'retention_period': '365 days',
        'chunk_time_interval': '1 day'
    }

logger.info("Loaded configuration", config={
    "DB_CONFIG": {
        k: "****" if "password" in k else v
        for k, v in Config.DB_CONFIG.items()
    },
    "BYBIT_CONFIG": {
        k: "****" if "secret" in k else v
        for k, v in Config.BYBIT_CONFIG.items()
    },
    "TRADING_CONFIG": Config.TRADING_CONFIG,
    "MODEL_CONFIG": Config.MODEL_CONFIG
})
