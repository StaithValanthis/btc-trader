from pybit.unified_trading import WebSocket
import asyncio

class BybitDataFetcher:
    def __init__(self, symbol='BTCUSDT', callback=None):
        self.symbol = symbol
        self.callback = callback
        self.ws = WebSocket(
            testnet=False,
            channel_type="linear",
            api_key="",  # Not needed for public data
            api_secret=""
        )

    async def stream_market_data(self):
        def handle_message(message):
            if 'topic' in message and message['topic'] == 'publicTrade.BTCUSDT':
                for trade in message['data']:
                    price = float(trade['p'])
                    if self.callback:
                        self.callback(price)

        # Subscribe to the public trade stream
        self.ws.trade_stream(
            symbol=self.symbol,
            callback=handle_message
        )

        while True:
            await asyncio.sleep(1)