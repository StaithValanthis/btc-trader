from pybit.unified_trading import WebSocket
import numpy as np
import asyncio

class BybitDataFetcher:
    def __init__(self, symbol='BTCUSDT', callback=None):
        self.symbol = symbol
        self.callback = callback
        self.ws = WebSocket(
            testnet=False,  # Set to True for testnet
            channel_type="linear"  # Use "linear" for USDT contracts
        )

    async def stream_market_data(self):
        """
        Stream real-time market data from Bybit.
        """
        def handle_message(message):
            """
            Handle incoming WebSocket messages.
            """
            if 'data' in message:
                price = float(message['data'][0]['last_price'])
                if self.callback:
                    self.callback(price)  # Send price to trading bot

        # Subscribe to the trade stream for the specified symbol
        self.ws.trade_stream(self.symbol, handle_message)

        # Keep the WebSocket connection alive
        while True:
            await asyncio.sleep(1)

    def compute_features(self, prices: list) -> np.ndarray:
        """
        Compute features for the ML model.
        """
        window = np.array(prices[-50:])  # Last 50 data points
        features = [
            np.mean(window),  # Moving average
            np.std(window),   # Volatility
            (window[-1] - window[0]) / window[0]  # Momentum
        ]
        return np.array(features)

async def main(callback):
    """
    Main function to start streaming data.
    """
    fetcher = BybitDataFetcher(symbol='BTCUSDT', callback=callback)
    await fetcher.stream_market_data()

# Example usage
if __name__ == "__main__":
    def print_price(price):
        print(f"Current Price: {price}")

    asyncio.run(main(print_price))