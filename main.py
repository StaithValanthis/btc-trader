from trading_bot import BitcoinTrader
import os

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if __name__ == "__main__":
    trader = BitcoinTrader()
    trader.trading_loop()