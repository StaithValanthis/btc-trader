import os
from dotenv import load_dotenv
from structlog import get_logger

logger = get_logger(__name__)
load_dotenv()

def parse_bool(env_value, default=False):
    if env_value is None:
        return default
    return env_value.strip().lower() == 'true'

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
        'testnet': parse_bool(os.getenv('BYBIT_TESTNET', 'false'))
    }

    # For MEXC coin-m futures, the symbol is typically "BTCUSD". 
    # For USDT-m futures, it might be "BTC_USD". Adjust as needed.
    TRADING_CONFIG = {
        'symbol': os.getenv('SYMBOL', 'BTC_USD'),
        'position_size': float(os.getenv('POSITION_SIZE', '1.0'))
    }

    MEXC_CONFIG = {
        'api_key': os.getenv('MEXC_API_KEY', ''),
        'api_secret': os.getenv('MEXC_API_SECRET', ''),
        'base_url': os.getenv('MEXC_BASE_URL', 'https://contract.mexc.com')
    }

logger.info("Loaded configuration", config={
    "DB_CONFIG": Config.DB_CONFIG,
    "BYBIT_CONFIG": {k: ("****" if "key" in k.lower() or "secret" in k.lower() else v)
                     for k, v in Config.BYBIT_CONFIG.items()},
    "TRADING_CONFIG": Config.TRADING_CONFIG,
    "MEXC_CONFIG": {k: ("****" if "key" in k.lower() or "secret" in k.lower() else v)
                    for k, v in Config.MEXC_CONFIG.items()}
})
