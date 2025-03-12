import os
from dotenv import load_dotenv
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

    MEXC_CONFIG = {
        'api_key': os.getenv('MEXC_API_KEY', ''),
        'api_secret': os.getenv('MEXC_API_SECRET', ''),
        'testnet': os.getenv('MEXC_TESTNET', 'false').lower() == 'true'
    }

    TRADING_CONFIG = {
        'symbol': os.getenv('SYMBOL', 'BTCUSD'),
        'position_size': float(os.getenv('POSITION_SIZE', '1.0'))
    }

logger.info("Loaded configuration", config={
    "DB_CONFIG": {k: "****" if "password" in k else v for k, v in Config.DB_CONFIG.items()},
    "MEXC_CONFIG": {k: "****" if "secret" in k else v for k, v in Config.MEXC_CONFIG.items()},
    "TRADING_CONFIG": Config.TRADING_CONFIG
})
