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
    # Database credentials
    DB_CONFIG = {
        'host':     os.getenv('DB_HOST', 'postgres'),
        'database': os.getenv('DB_NAME', 'trading_bot'),
        'user':     os.getenv('DB_USER', 'postgres'),
        'password': os.getenv('DB_PASSWORD', 'postgres')
    }

    # Bybit API + market type
    BYBIT_CONFIG = {
        'api_key':    os.getenv('BYBIT_API_KEY', ''),
        'api_secret': os.getenv('BYBIT_API_SECRET', ''),
        'testnet':    parse_bool(os.getenv('BYBIT_TESTNET', 'false')),
        'category':   os.getenv('BYBIT_CATEGORY', 'linear')  # linear = USDT‐margined
    }

    # Trading config: symbols + MR params + stop-loss + risk per trade
    TRADING_CONFIG = {
        'symbols':       os.getenv(
                            'SYMBOLS',
                            'BTCUSDT,ETHUSDT,XRPUSDT,BNBUSDT,SOLUSDT,TRXUSDT,DOGEUSDT,ADAUSDT,'
                            'HYPEUSDT,BCHUSDT,SUIUSDT,LEOUSDT,LINKUSDT,XLMUSDT,AVAXUSDT,TONUSDT,'
                            'SHIBUSDT,LTCUSDT,HBARUSDT,XMRUSDT,DOTUSDT,BGBUSDT,UNIUSDT,PIUSDT,'
                            'PEPEUSDT,AAVEUSDT,OKBUSDT,TAOUSDT,APTUSDT,CROUSDT,ICPUSDT,NEARUSDT,'
                            'ETCUSDT,ONDOUSDT,GTUSDT,MNTUSDT,MATICUSDT,KASUSDT,TRUMPUSDT,VETUSDT,'
                            'SKYUSDT,ENAUSDT,RENDERUSDT,ATOMUSDT,FETUSDT,FILUSDT,ALGOUSDT,WLDUSDT,'
                            'KCSUSDT,ARBUSDT'
                        ).split(','),
        # Risk settings
        'risk_pct':      float(os.getenv('RISK_PCT', '0.02')),   # 2% of equity per trade
        'SL_PCT':        float(os.getenv('SL_PCT', '0.01')),     # 1% stop-loss
        # Leverage
        'leverage':      int(os.getenv('LEVERAGE', '5')),       # default 5x
        # Other MR params
        'BB_WINDOW':     int(os.getenv('BB_WINDOW', '20')),
        'BB_DEV':        float(os.getenv('BB_DEV', '2.0')),
        'RSI_LONG':      int(os.getenv('RSI_LONG', '30')),
        'RSI_SHORT':     int(os.getenv('RSI_SHORT', '70')),
    }

    # Parameter ranges for scripts
    PARAM_RANGES = {
        'BB_WINDOW': [int(x)   for x in os.getenv('RANGE_BB_WINDOW',  '10,20,30').split(',')],
        'BB_DEV':    [float(x) for x in os.getenv('RANGE_BB_DEV',     '1.5,2.0,2.5').split(',')],
        'RSI_LONG':  [int(x)   for x in os.getenv('RANGE_RSI_LONG',   '20,30').split(',')],
        'RSI_SHORT': [int(x)   for x in os.getenv('RANGE_RSI_SHORT',  '70,80').split(',')],
        'SL_PCT':    [float(x) for x in os.getenv('RANGE_SL_PCT',     '0.005,0.01,0.02').split(',')],
        'MC_TRIALS': int(os.getenv('MONTE_CARLO_TRIALS', '100')),
        'MC_RANGE': {
            'window':    tuple(int(x)   for x in os.getenv('MC_RANGE_WINDOW',   '10,50').split(',')),
            'dev':       tuple(float(x) for x in os.getenv('MC_RANGE_DEV',      '1.0,3.0').split(',')),
            'rsi_long':  tuple(int(x)   for x in os.getenv('MC_RANGE_RSI_LONG', '10,40').split(',')),
            'rsi_short': tuple(int(x)   for x in os.getenv('MC_RANGE_RSI_SHORT','60,90').split(',')),
        }
    }

logger.info("Loaded configuration", config={
    "TRADING_CONFIG": Config.TRADING_CONFIG,
    "PARAM_RANGES":   Config.PARAM_RANGES
})
