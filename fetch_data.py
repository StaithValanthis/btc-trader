from pybit.unified_trading import WebSocket
import asyncio

class BybitDataFetcher:
    def __init__(self, symbol='BTCUSDT', callback=None):
        self.symbol = symbol
        self.callback = callback
        self.ws = WebSocket(
            testnet=False,
            channel_type="linear"
        )

    async def stream_market_data(self):
        def handle_message(message):
            if 'data' in message:
                price = float(message['data'][0]['last_price'])
                if self.callback:
                    self.callback(price)

        self.ws.trade_stream(self.symbol, handle_message)
        while True:
            await asyncio.sleep(1)