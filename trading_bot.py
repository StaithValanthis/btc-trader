from fetch_data import BybitDataFetcher
from database import TimescaleDB
from datetime import datetime
import numpy as np
import asyncio
import json

class BitcoinTrader:
    def __init__(self):
        self.db = TimescaleDB()
        self.data_buffer = []
        self.window_size = 50  # For feature calculation

    async def run(self):
        fetcher = BybitDataFetcher(callback=self.handle_price)
        await fetcher.stream_market_data()

    def handle_price(self, price: float):
        # Record timestamp
        time = datetime.now()

        # Compute features
        features = self.compute_features(price)
        self.db.log_market_data(time, price, features)

        # Update model
        self.update_model(features, price)

        # Generate signal
        signal = self.generate_signal(features)
        self.db.log_trade(time, price, signal)

    def compute_features(self, price: float) -> dict:
        self.data_buffer.append(price)
        if len(self.data_buffer) < self.window_size:
            return {}

        window = np.array(self.data_buffer[-self.window_size:])
        features = {
            "moving_average": np.mean(window),
            "volatility": np.std(window),
            "momentum": (window[-1] - window[0]) / window[0]
        }
        return features

    def update_model(self, features: dict, price: float):
        if len(self.data_buffer) < self.window_size:
            return

        # Fetch historical data for training
        historical_data = self.db.get_historical_data(lookback_days=7)
        X, y = [], []

        for data in historical_data:
            time, price_hist, features_hist = data
            features_hist = json.loads(features_hist)
            X.append([
                features_hist["moving_average"],
                features_hist["volatility"],
                features_hist["momentum"]
            ])
            y.append(price_hist)

        # Train the model
        if len(X) > 0:
            self.trainer.train(np.array(X), np.array(y), epochs=5)

    def generate_signal(self, features: dict) -> str:
        if not features:
            return "HOLD"

        X = np.array([
            features["moving_average"],
            features["volatility"],
            features["momentum"]
        ]).reshape(1, -1)

        prediction = self.trainer.predict(X)[0][0]
        if prediction > 0.02:
            return "BUY"
        elif prediction < -0.02:
            return "SELL"
        else:
            return "HOLD"

if __name__ == "__main__":
    trader = BitcoinTrader()
    asyncio.run(trader.run())