# File: v2-Inverse-btc-trader/app/core/config.py

import os
from dotenv import load_dotenv
from structlog import get_logger

logger = get_logger(__name__)
load_dotenv()

def parse_bool(env_value: str, default: bool=False) -> bool:
    """
    Helper to parse boolean from environment string.
    """
    if env_value is None:
        return default
    return env_value.strip().lower() == 'true'

class Config:
    """
    Central configuration object, loads environment variables for DB, Bybit, etc.
    """
    DB_CONFIG = {
        'host': os.getenv('DB_HOST', 'postgres'),
        'database': os.getenv('DB_NAME', 'trading_bot'),
        'user': os.getenv('DB_USER', 'postgres'),
        'password': os.getenv('DB_PASSWORD', 'postgres')
    }

    BYBIT_CONFIG = {
        'api_key': os.getenv('BYBIT_API_KEY', ''),
        'api_secret': os.getenv('BYBIT_API_SECRET', ''),
        'testnet': parse_bool(os.getenv('BYBIT_TESTNET', 'false'))
    }

    TRADING_CONFIG = {
        'symbol': os.getenv('SYMBOL', 'BTCUSD'),
        'position_size': float(os.getenv('POSITION_SIZE', '1.0')),
        'category': os.getenv('BYBIT_CATEGORY', 'inverse')  # e.g., "inverse" or "linear"
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
    "TRADING_CONFIG": Config.TRADING_CONFIG
})
