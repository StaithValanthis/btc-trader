from fetch_data import BybitDataFetcher
from online_learner import OnlineLearner
from strategy import TradingStrategy
import numpy as np
import asyncio
from dotenv import load_dotenv
import os

load_dotenv()

class BitcoinTrader:
    def __init__(self):
        self.data_buffer = []
        self.initial_data = None  # Populate with real data
        self.learner = OnlineLearner()
        self.strategy = TradingStrategy(self.learner.model)
        self.api_key = os.getenv("BYBIT_API_KEY")
        self.api_secret = os.getenv("BYBIT_API_SECRET")

    async def run(self):
        # Collect initial data for x_ref
        await self.collect_initial_data()
        
        # Reinitialize learner with real x_ref
        self.learner = OnlineLearner(x_ref=self.initial_data)
        
        # Start trading loop
        def handle_price(price):
            self.data_buffer.append(price)
            if len(self.data_buffer) >= 50:
                X = self.compute_features(self.data_buffer)
                y = self._get_labels()
                self.learner.update(X, y)
                signal = self.strategy.generate_signal(X)
                self.execute_trade(signal)
                self.data_buffer.pop(0)

        fetcher = BybitDataFetcher(callback=handle_price)
        await fetcher.stream_market_data()

    async def collect_initial_data(self):
        # Fetch initial 100 data points
        initial_prices = await self.fetch_initial_prices()
        self.initial_data = self.compute_features(initial_prices)

    async def fetch_initial_prices(self):
        # Implement data collection logic (e.g., REST API call)
        return [...]  # Replace with real data

    def compute_features(self, prices: list) -> np.ndarray:
        window = np.array(prices[-50:])
        features = [
            np.mean(window),
            np.std(window),
            (window[-1] - window[0]) / window[0]
        ]
        return np.array(features)

    def _get_labels(self) -> np.ndarray:
        future_prices = self.data_buffer[1:]
        labels = (np.diff(future_prices) > 0).astype(int)
        return labels[:-1]

    def execute_trade(self, signal: str):
        # Replace with Bybit API calls
        print(f"Executing trade: {signal}")
        # Example (uncomment and use your API keys):
        # from pybit.unified_trading import HTTP
        # session = HTTP(
        #     api_key=self.api_key,
        #     api_secret=self.api_secret,
        #     testnet=self.testnet
        # )
        # if signal == "BUY":
        #     session.place_order(
        #         category="linear",
        #         symbol="BTCUSDT",
        #         side="Buy",
        #         orderType="Market",
        #         qty="0.001"
        #     )

async def main():
    trader = BitcoinTrader()
    await trader.run()

if __name__ == "__main__":
    asyncio.run(main())