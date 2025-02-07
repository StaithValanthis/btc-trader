import json
from pybit.unified_trading import WebSocket
import asyncio
from structlog import get_logger
from app.core.database import Database
from datetime import datetime, timezone
from app.core.config import Config

logger = get_logger(__name__)

class BybitMarketData:
    def __init__(self, symbol: str = Config.TRADING_CONFIG['symbol']):
        self.symbol = symbol
        self.ws = None
        self.running = False
        self.loop = asyncio.get_event_loop()

    async def _connect_websocket(self):
        try:
            self.ws = WebSocket(
                testnet=Config.BYBIT_CONFIG['testnet'],
                channel_type="linear",
                api_key=Config.BYBIT_CONFIG['api_key'],
                api_secret=Config.BYBIT_CONFIG['api_secret'],
                trace_logging=False  # Enable verbose WS logging
            )
            
            def handle_message(message):
                logger.debug("WebSocket message received", message=message)
                if 'data' in message and isinstance(message['data'], list):
                    for trade in message['data']:
                        asyncio.run_coroutine_threadsafe(
                            self._process_message(trade),
                            self.loop
                        )
            
            self.ws.trade_stream(
                symbol=self.symbol,
                callback=handle_message
            )
            logger.info("WebSocket connected successfully")
        except Exception as e:
            logger.error("WebSocket connection failed", error=str(e))
            raise

    async def _process_message(self, trade):
        try:
            await Database.execute('''
                INSERT INTO market_data (time, price, volume)
                VALUES ($1, $2, $3)
                ON CONFLICT (time) DO NOTHING
            ''', datetime.now(timezone.utc), 
               float(trade.get('p', 0)), 
               float(trade.get('v', 0)))
        except Exception as e:
            logger.error("Data processing failed", error=str(e))

    async def run(self):
        self.running = True
        while self.running:
            try:
                if not self.ws:
                    await self._connect_websocket()
                await asyncio.sleep(1)
            except Exception as e:
                logger.error("WebSocket error", error=str(e))
                self.ws = None
                await asyncio.sleep(5)

    async def stop(self):
        self.running = False
        if self.ws:
            self.ws.exit()
        logger.info("WebSocket stopped")