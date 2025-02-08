import json
import asyncio
import websockets
from structlog import get_logger
from datetime import datetime, timezone
from app.core.database import Database
from app.core.config import Config

logger = get_logger(__name__)

class BybitMarketData:
    def __init__(self, symbol: str = Config.TRADING_CONFIG['symbol']):
        self.symbol = symbol
        self.ws = None
        self.running = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        self.last_message_time = None

    async def _connect_websocket(self):
        try:
            self.ws = await websockets.connect(
                "wss://stream.bybit.com/v5/public/linear",
                ping_interval=30,
                ping_timeout=10,
                extra_headers={"User-Agent": "BTC-Trader/1.0"}
            )
            
            # Subscribe to correct trade stream format
            subscription = {
                "op": "subscribe",
                "args": [f"publicTrade.{self.symbol}"]
            }
            await self.ws.send(json.dumps(subscription))
            
            # Verify subscription response
            response = await self.ws.recv()
            logger.debug("Subscription response", response=json.loads(response))
            
            # Start listening task
            asyncio.create_task(self._listen())
            self.reconnect_attempts = 0
            logger.info("WebSocket connected successfully")
            
        except Exception as e:
            logger.error("WebSocket connection failed", error=str(e))
            await self._reconnect()

    async def _reconnect(self):
        if self.reconnect_attempts < self.max_reconnect_attempts:
            self.reconnect_attempts += 1
            wait_time = min(30, 2 ** self.reconnect_attempts)
            logger.warning(f"Reconnecting attempt {self.reconnect_attempts} in {wait_time}s...")
            await asyncio.sleep(wait_time)
            await self._connect_websocket()
        else:
            logger.error("Max reconnect attempts reached")
            self.running = False

    async def _listen(self):
        while self.running:
            try:
                message = await self.ws.recv()
                self.last_message_time = datetime.now(timezone.utc)
                data = json.loads(message)
                
                if 'topic' in data and data['topic'] == f"publicTrade.{self.symbol}":
                    await self._process_message(data['data'])
                else:
                    logger.debug("Received non-trade message", message=data)
                    
            except websockets.ConnectionClosed:
                logger.warning("WebSocket connection closed")
                await self._reconnect()
            except Exception as e:
                logger.error("Message processing error", error=str(e))

    async def _process_message(self, trade_data):
        try:
            if not isinstance(trade_data, list):
                trade_data = [trade_data]
                
            logger.debug("Processing trades", count=len(trade_data))
            
            for trade in trade_data:
                # Convert Bybit timestamp to UTC datetime
                trade_time = datetime.fromtimestamp(
                    int(trade['T']) / 1000, 
                    tz=timezone.utc
                )
                
                # Insert into database with error handling
                result = await Database.execute('''
                    INSERT INTO market_data (time, price, volume)
                    VALUES ($1, $2, $3)
                    ON CONFLICT (time) DO NOTHING
                ''', trade_time,
                   float(trade['p']),
                   float(trade['v']))
                
                if result == "INSERT 0 1":
                    logger.debug("Inserted new trade", 
                                time=trade_time.isoformat(),
                                price=trade['p'],
                                volume=trade['v'])
                else:
                    logger.debug("Skipped duplicate trade", time=trade_time.isoformat())
                    
        except Exception as e:
            logger.error("Trade processing failed", error=str(e))

    async def run(self):
        self.running = True
        await self._connect_websocket()
        # Start connection monitor
        asyncio.create_task(self._connection_monitor())

    async def _connection_monitor(self):
        while self.running:
            # Check for stale connection
            if self.last_message_time and \
               (datetime.now(timezone.utc) - self.last_message_time).total_seconds() > 60:
                logger.warning("No messages received for 60 seconds, reconnecting...")
                await self._reconnect()
            await asyncio.sleep(10)

    async def stop(self):
        self.running = False
        if self.ws:
            await self.ws.close()
        logger.info("WebSocket stopped")