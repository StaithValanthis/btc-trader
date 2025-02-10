# app/core/bybit_client.py
import asyncio
import json
import websockets
from datetime import datetime, timezone
from structlog import get_logger
from app.core import Database, Config

logger = get_logger(__name__)

class BybitMarketData:
    def __init__(self):
        self.ws = None
        self.running = False
        self.reconnect_attempts = 0
        self.max_reconnects = 5

    async def initialize(self):
        """Initialize WebSocket connection"""
        await self._connect_websocket()

    async def _connect_websocket(self):
        """Establish WebSocket connection with retries"""
        try:
            self.ws = await websockets.connect(
                "wss://stream.bybit.com/v5/public/linear",
                ping_interval=30,
                ping_timeout=10
            )
            
            # Subscribe to trade stream
            await self.ws.send(json.dumps({
                "op": "subscribe",
                "args": [f"publicTrade.{Config.TRADING_CONFIG['symbol']}"]
            }))
            await self.ws.recv()  # Confirm subscription
            self.reconnect_attempts = 0
            logger.info("WebSocket connected")

        except Exception as e:
            logger.error("WebSocket connection failed", error=str(e))
            await self._reconnect()

    async def _process_message(self, trade_data):
        """Process incoming market data"""
        try:
            # Handle both single trades and batch trades
            trades = trade_data if isinstance(trade_data, list) else [trade_data]
            
            values = [
                (datetime.fromtimestamp(int(trade['T'])/1000, tz=timezone.utc),
                 float(trade['p']),
                 float(trade['v']))
                for trade in trades
            ]

            await Database.execute('''
                INSERT INTO market_data (time, price, volume)
                SELECT * FROM UNNEST($1::timestamptz[], $2::float8[], $3::float8[])
                ON CONFLICT (time, price, volume) DO NOTHING
            ''', [v[0] for v in values], [v[1] for v in values], [v[2] for v in values])

        except Exception as e:
            logger.error("Data processing error", error=str(e))

    async def run(self):
        """Main data ingestion loop"""
        self.running = True
        while self.running:
            try:
                message = await self.ws.recv()
                data = json.loads(message)
                if 'topic' in data and data['topic'].startswith('publicTrade'):
                    await self._process_message(data['data'])
            except websockets.ConnectionClosed:
                logger.warning("WebSocket connection closed")
                await self._reconnect()
            except Exception as e:
                logger.error("Message processing error", error=str(e))

    async def stop(self):
        """Clean shutdown procedure"""
        self.running = False
        if self.ws:
            await self.ws.close()
        logger.info("Market data service stopped")