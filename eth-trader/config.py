import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Bybit API
    BYBIT_API_KEY = os.getenv("BYBIT_API_KEY")
    BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET")
    BYBIT_BASE_URL = os.getenv("BYBIT_BASE_URL", "https://api.bybit.com")
    TICKER_ENDPOINT = f"{BYBIT_BASE_URL}/v2/public/tickers?symbol=ETHUSD"

    # Database settings (Tortoise ORM)
    POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
    POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "mysecretpassword")
    POSTGRES_DB = os.getenv("POSTGRES_DB", "trading_data")
    POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
    POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))

    # Trading settings
    CAPITAL = float(os.getenv("CAPITAL", "10000"))
    PRICE_WINDOW = 5
    PREDICTION_MARGIN = 10.0
    RETRAIN_INTERVAL = 50
    RISK_PERCENT = 0.01
    VOLATILITY_WINDOW = 20
    VOLATILITY_MULTIPLIER = 1.0
    DEFAULT_QUANTITY = 0.01

    # Indicator settings
    RSI_PERIOD = 14
    RSI_OVERSOLD = 30
    RSI_OVERBOUGHT = 70
    ATR_PERIOD = 14
    BB_WINDOW = 20
    BB_STD_MULTIPLIER = 2.0
    EMA_PERIOD = 20

    # Tortoise ORM configuration
    TORTOISE_ORM = {
        "connections": {
            "default": f"postgres://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}?min_size=1&max_size=10"
        },
        "apps": {
            "models": {
                "models": ["database"],
                "default_connection": "default",
            }
        }
    }
