from pybit.unified_trading import WebSocket
import asyncio
import time

class BybitDataFetcher:
    def __init__(self, symbol='BTCUSDT', callback=None):
        self.symbol = symbol
        self.callback = callback
        self.ws = None
        self.retry_count = 0
        self.max_retries = 5

    async def stream_market_data(self):
        """
        Stream real-time market data from Bybit.
        """
        while self.retry_count < self.max_retries:
            try:
                self.ws = WebSocket(
                    testnet=False,
                    channel_type="linear"
                )

                def handle_message(message):
                    if 'data' in message:
                        price = float(message['data'][0]['last_price'])
                        if self.callback:
                            self.callback(price)

                # Subscribe to the trade stream
                self.ws.trade_stream(self.symbol, handle_message)

                # Keep the WebSocket connection alive
                while True:
                    await asyncio.sleep(1)

            except Exception as e:
                print(f"WebSocket error: {e}. Retrying... ({self.retry_count + 1}/{self.max_retries})")
                self.retry_count += 1
                time.sleep(5)  # Wait before retrying

        print("Max retries reached. Exiting...")

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