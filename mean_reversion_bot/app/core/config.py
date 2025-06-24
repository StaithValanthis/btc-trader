import os
from dotenv import load_dotenv
from structlog import get_logger

logger = get_logger(__name__)
load_dotenv()

def parse_bool(val, default=False):
    if val is None: return default
    return val.strip().lower() in ("1","true","yes")

Config = {
    "DB": {
        "host":     os.getenv("DB_HOST", "postgres"),
        "database": os.getenv("DB_NAME", "trading_bot"),
        "user":     os.getenv("DB_USER", "postgres"),
        "password": os.getenv("DB_PASSWORD", "postgres"),
    },
    "BYBIT": {
        "api_key":   os.getenv("BYBIT_API_KEY",""),
        "api_secret":os.getenv("BYBIT_API_SECRET",""),
        "testnet":   parse_bool(os.getenv("BYBIT_TESTNET","false")),
        "category":  os.getenv("BYBIT_CATEGORY","linear"),
    },
    "TRADING": {
        "symbols":   os.getenv("SYMBOLS","BTCUSDT,ETHUSDT").split(","),
        "risk_pct":  float(os.getenv("RISK_PCT","0.02")),
        "SL_PCT":    float(os.getenv("SL_PCT","0.01")),
        "leverage":  int(os.getenv("LEVERAGE","5")),
        "BB_WINDOW": int(os.getenv("BB_WINDOW","20")),
        "BB_DEV":    float(os.getenv("BB_DEV","2.0")),
        "RSI_LONG":  int(os.getenv("RSI_LONG","30")),
        "RSI_SHORT": int(os.getenv("RSI_SHORT","70")),
    },
    "PARAM_RANGES": {
        "BB_WINDOW":  [int(x)   for x in os.getenv("RANGE_BB_WINDOW","10,20,30").split(",")],
        "BB_DEV":     [float(x) for x in os.getenv("RANGE_BB_DEV","1.5,2.0,2.5").split(",")],
        "RSI_LONG":   [int(x)   for x in os.getenv("RANGE_RSI_LONG","20,30").split(",")],
        "RSI_SHORT":  [int(x)   for x in os.getenv("RANGE_RSI_SHORT","70,80").split(",")],
        "SL_PCT":     [float(x) for x in os.getenv("RANGE_SL_PCT","0.005,0.01,0.02").split(",")],
        "MC_TRIALS":  int(os.getenv("MONTE_CARLO_TRIALS","100")),
        "MC_RANGE": {
            "window":   tuple(int(x)   for x in os.getenv("MC_RANGE_WINDOW","10,50").split(",")),
            "dev":      tuple(float(x) for x in os.getenv("MC_RANGE_DEV","1.0,3.0").split(",")),
            "rsi_long": tuple(int(x)   for x in os.getenv("MC_RANGE_RSI_LONG","10,40").split(",")),
            "rsi_short":tuple(int(x)   for x in os.getenv("MC_RANGE_RSI_SHORT","60,90").split(",")),
        }
    }
}

logger.info("Loaded config", Config)
