from bybit import BybitWebsocket
import os
import json
import asyncio
from database import Database
from datetime import datetime, timezone

class BybitMarketData:
    def __init__(self):
        self.ws = None
        self.api_key = os.getenv("BYBIT_API_KEY")
        self.api_secret = os.getenv("BYBIT_API_SECRET")
        self.symbol = "BTCUSD"

    async def _connect_websocket(self):
        self.ws = BybitWebsocket(
            ws_url="wss://stream.bybit.com/v5/public/linear",
            api_key=self.api_key,
            api_secret=self.api_secret
        )
        self.ws.subscribe(f"publicTrade.{self.symbol}")

    def _handle_message(self, msg):
        try:
            if 'topic' in msg and 'publicTrade' in msg['topic']:
                for trade in msg['data']:
                    price = float(trade['price'])
                    features = {
                        'size': trade['size'],
                        'side': trade['side'],
                        'trade_time': trade['trade_time_ms']
                    }
                    asyncio.create_task(Database.insert_market_data(price, features))
        except Exception as e:
            print(f"Error processing message: {e}")

    async def run(self):
        while True:
            try:
                if not self.ws:
                    await self._connect_websocket()
                
                message = self.ws.get_data()
                if message:
                    self._handle_message(message)
                await asyncio.sleep(0.1)
            except Exception as e:
                print(f"WebSocket error: {e}")
                self.ws = None
                await asyncio.sleep(5)