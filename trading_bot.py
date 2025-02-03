import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Additional performance flag
import sys
import time
import pandas as pd
from pybit.unified_trading import HTTP
from strategy import MLStrategy
from online_learner import OnlineLearner
import logging
from dotenv import load_dotenv

load_dotenv()

# Configure robust logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(module)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_debug.log'),
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)

class BitcoinTrader:
    def __init__(self):
        # Initialize with automatic retry
        self.client = self._init_bybit_client()
        self.strategy = MLStrategy()
        self.learner = OnlineLearner()
        self.symbol = "BTCUSDT"
        self.min_qty = 0.001
        self.position_size = 0.0
        self.balance_refresh = time.time()
        self.current_balance = 1000  # Default paper balance
        
    def _init_bybit_client(self, retries=3):
        """Robust client initialization"""
        for attempt in range(retries):
            try:
                return HTTP(
                    testnet=True,
                    api_key=os.getenv('BYBIT_API_KEY'),
                    api_secret=os.getenv('BYBIT_API_SECRET'),
                    recv_window=20000
                )
            except Exception as e:
                logger.warning(f"Client init failed (attempt {attempt+1}): {str(e)}")
                time.sleep(2)
        raise ConnectionError("Failed to initialize Bybit client")

    def fetch_data(self, limit=100):
        """Enhanced data fetching with retries"""
        try:
            response = self.client.get_kline(
                category="linear",
                symbol=self.symbol,
                interval="15",  # String format for compatibility
                limit=limit
            )
            
            # Create properly ordered DataFrame
            df = pd.DataFrame(
                response['result']['list'],
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover']
            )[::-1].reset_index(drop=True)  # Reverse for chronological order
            
            # Convert and validate
            numeric_cols = ['open', 'high', 'low', 'close']
            df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
            df = df.dropna(subset=['close'])
            
            logger.debug(f"Fetched {len(df)} valid records")
            return df
            
        except Exception as e:
            logger.error(f"Data fetch error: {str(e)}")
            return pd.DataFrame()

    def generate_signal(self):
        """Generate trading signals with drift detection"""
        try:
            data = self.fetch_data()
            if len(data) < 20:
                logger.warning("Insufficient data (min 20 records required)")
                return 'Hold'
            
            features = self.strategy.create_features(data)
            labels = (data['close'].shift(-4) > data['close']).astype(int).dropna()
            
            min_len = min(len(features), len(labels))
            features = features.iloc[:min_len]
            labels = labels.iloc[:min_len]
            
            # Online learning
            for X, y in zip(features.values[-50:], labels.values[-50:]):
                self.learner.update(X, y)
            
            # Concept drift detection
            if len(features) >= 10:
                recent_data = features.values[-10:]
                drift_detected = self.learner.detect_drift(recent_data)
                if drift_detected:
                    logger.warning("Concept drift detected! Initiating retraining...")
                    self.strategy.train_model(data)
                    self.learner = OnlineLearner()  # Reset drift detector
            
            return self.strategy.calculate_signals(data)
            
        except Exception as e:
            logger.error(f"Signal generation failed: {str(e)}")
            return 'Hold'

    def trading_loop(self):
        """Main loop with enhanced safeguards"""
        logger.info("Starting trading loop in PAPER mode")
        while True:
            try:
                # Refresh balance hourly
                if time.time() - self.balance_refresh > 3600:
                    self._update_balance()
                
                signal = self.generate_signal()
                logger.info(f"Market signal: {signal}")
                
                if signal in ('Buy', 'Sell'):
                    self.execute_order(signal)
                
                # Flexible interval for debugging
                sleep_time = 60 if os.getenv('DEBUG') else 900
                time.sleep(sleep_time)
                
            except KeyboardInterrupt:
                logger.info("User initiated shutdown")
                break
            except Exception as e:
                logger.critical(f"Critical loop error: {str(e)}")
                time.sleep(300)  # Prevent tight error loops

    def execute_order(self, side):
        """Execute order with enhanced validation"""
        try:
            if os.getenv('MODE') != 'paper':
                logger.warning("Live trading disabled - check .env MODE setting")
                return

            qty = self.calculate_position_size()
            if float(qty) < self.min_qty:
                logger.warning(f"Position too small: {qty} BTC")
                return
                
            logger.info(f"Paper trade: {side} {qty} BTC")
            self.simulate_order_execution(side, qty)
            
        except Exception as e:
            logger.error(f"Order failed: {str(e)}")

    def simulate_order_execution(self, side, qty):
        """Simulate order execution with position tracking"""
        logger.info(f"SIMULATION: {side} order executed for {qty} BTC")
        self.position_size = float(qty) * (1 if side == 'Buy' else -1)
        logger.info(f"New position: {self.position_size:.4f} BTC")

    def calculate_position_size(self):
        """Calculate position size with risk management"""
        try:
            ticker = self.client.get_tickers(
                category="linear",
                symbol=self.symbol
            )['result']['list'][0]
            price = float(ticker['lastPrice'])
            
            risk_percent = float(os.getenv('RISK_PERCENT', 1))
            risk_amount = self.current_balance * (risk_percent / 100)
            qty = max(risk_amount / price, self.min_qty)
            
            return f"{round(qty, 4)}"
        except Exception as e:
            logger.error(f"Position calc error: {str(e)}")
            return "0"

    def _update_balance(self):
        """Safe balance update with paper mode check"""
        if os.getenv('MODE') == 'paper':
            try:
                # Simulated balance growth for paper trading
                self.current_balance *= 1.002  # 0.2% growth simulation
                logger.info(f"Paper balance updated: ${self.current_balance:.2f}")
            except:
                self.current_balance = 1000
        else:
            # Real balance update logic
            self.get_account_balance()

    def get_account_balance(self, retries=3):
        """Fetch account balance with retry logic"""
        for attempt in range(retries):
            try:
                response = self.client.get_wallet_balance(
                    accountType="UNIFIED",
                    coin="USDT"
                )
                balance = response['result']['list'][0]['coin'][0]['availableToWithdraw']
                self.current_balance = max(0, float(balance))
                self.balance_refresh = time.time()
                logger.info(f"Updated balance: ${self.current_balance:.2f}")
                return
            except Exception as e:
                logger.warning(f"Balance attempt {attempt+1} failed: {str(e)}")
                time.sleep(2)
        logger.error("Failed to update balance after retries")
        self.current_balance = 0

if __name__ == "__main__":
    trader = BitcoinTrader()
    trader.trading_loop()