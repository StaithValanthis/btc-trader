from dotenv import load_dotenv
import os

load_dotenv()

def get_env_variable(name, default):
    """Safely get environment variable with proper type conversion"""
    value = os.getenv(name)
    if value is None or value.strip() == '':
        return default
    try:
        return float(value)
    except ValueError:
        return default

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
        'max_leverage': get_env_variable('MAX_LEVERAGE', 10)
    }