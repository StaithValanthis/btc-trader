from pybit.unified_trading import WebSocket
import asyncio
import time
import logging

logging.basicConfig(level=logging.INFO)

class BybitDataFetcher:
    def __init__(self, symbol='BTCUSDT', callback=None):
        self.symbol = symbol
        self.callback = callback
        self.ws = None
        self.retry_count = 0
        self.max_retries = 5
        self.reconnect_delay = 5  # seconds
        self.running = True

    async def _connect(self):
        """Establish WebSocket connection with retries"""
        while self.running and self.retry_count < self.max_retries:
            try:
                self.ws = WebSocket(
                    testnet=False,
                    channel_type="linear"
                )
                logging.info("WebSocket connection established")
                return True
            except Exception as e:
                logging.error(f"Connection failed: {e}")
                self.retry_count += 1
                await asyncio.sleep(self.reconnect_delay)
        return False

    async def _subscribe(self):
        """Subscribe to market data"""
        def handle_message(message):
            if 'data' in message:
                for trade in message['data']:
                    price = float(trade['p'])
                    if self.callback:
                        self.callback(price)

        self.ws.trade_stream(self.symbol, handle_message)

    async def stream_market_data(self):
        """Main WebSocket management loop"""
        while self.running:
            if await self._connect():
                try:
                    await self._subscribe()
                    while self.running:
                        await asyncio.sleep(1)
                except Exception as e:
                    logging.error(f"WebSocket error: {e}")
                finally:
                    await self._disconnect()
            else:
                logging.error("Max connection attempts reached")
                break
            await asyncio.sleep(self.reconnect_delay)

    async def _disconnect(self):
        """Cleanly close the connection"""
        if self.ws:
            try:
                await self.ws.close()
            except Exception as e:
                logging.error(f"Error closing connection: {e}")
            self.ws = None
            logging.info("WebSocket disconnected")

    def stop(self):
        """Graceful shutdown"""
        self.running = False