from dotenv import load_dotenv
import os
from structlog import get_logger  # Add this import

# Initialize logger
logger = get_logger(__name__)  # Add this line

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
    
    TIMESCALE_CONFIG = {
        'compression_interval': '7 days',
        'retention_period': '365 days',
        'chunk_time_interval': '1 day'
    }

# Log the configuration for debugging
logger.info("Loaded configuration", config={
    "DB_CONFIG": {k: "****" if "password" in k else v for k, v in Config.DB_CONFIG.items()},
    "BYBIT_CONFIG": {k: "****" if "secret" in k else v for k, v in Config.BYBIT_CONFIG.items()},
    "TRADING_CONFIG": Config.TRADING_CONFIG,
    "MODEL_CONFIG": Config.MODEL_CONFIG
})