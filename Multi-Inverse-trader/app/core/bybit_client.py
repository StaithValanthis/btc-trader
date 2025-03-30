import asyncio
import contextlib
import json
import websockets
from datetime import datetime, timezone
from structlog import get_logger
from app.core.database import Database
from app.core.config import Config

logger = get_logger(__name__)

class BybitMarketData:
    def __init__(self, symbols=None):
        # Accept a list of symbols; default to configuration value if None
        self.symbols = symbols if symbols is not None else Config.TRADING_CONFIG['symbols']
        self.ws = None
        self.running = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        self.last_message_time = None
        self.recv_lock = asyncio.Lock()
        self.message_queue = asyncio.Queue()
        # Choose the proper websocket endpoint based on testnet setting
        if Config.BYBIT_CONFIG['testnet']:
            self.websocket_url = "wss://stream-testnet.bybit.com/v5/public/inverse"
        else:
            self.websocket_url = "wss://stream.bybit.com/v5/public/inverse"
        self._listener_task = None
        self._process_task = None
        self._monitor_task = None

    async def run(self):
        self.running = True
        await self._connect_websocket()
        if self.running and self.ws:
            self._listener_task = asyncio.create_task(self._listener(), name="BybitListener")
            self._process_task = asyncio.create_task(self._process_messages(), name="BybitProcessMsg")
            self._monitor_task = asyncio.create_task(self._connection_monitor(), name="BybitConnMon")

    async def stop(self):
        self.running = False
        if self._listener_task:
            self._listener_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._listener_task
            self._listener_task = None
        if self._process_task:
            self._process_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._process_task
            self._process_task = None
        if self._monitor_task:
            self._monitor_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._monitor_task
            self._monitor_task = None
        if self.ws:
            try:
                await self.ws.close()
            except Exception as e:
                logger.warning("Error closing websocket", error=str(e))
            self.ws = None
        logger.info("Market data service stopped")

    async def _connect_websocket(self):
        try:
            logger.info("Connecting to Bybit WebSocket...", url=self.websocket_url)
            self.ws = await websockets.connect(
                self.websocket_url,
                ping_interval=30,
                ping_timeout=10,
                extra_headers={"User-Agent": "BTC-Trader/1.0"}
            )
            # Subscribe to all symbols (e.g., "publicTrade.BTCUSD", "publicTrade.SOLUSD", …)
            subscription = {
                "op": "subscribe",
                "args": [f"publicTrade.{symbol}" for symbol in self.symbols]
            }
            await self.ws.send(json.dumps(subscription))
            response = await self.ws.recv()
            logger.debug("Subscription response", response=json.loads(response))
            self.reconnect_attempts = 0
            logger.info("WebSocket connected successfully")
        except Exception as e:
            logger.error("WebSocket connection failed", error=str(e))
            await self._reconnect()

    async def _reconnect(self):
        if self.ws:
            try:
                await self.ws.close()
            except Exception as e:
                logger.warning("Error closing old websocket", error=str(e))
            self.ws = None
        if self._listener_task:
            self._listener_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._listener_task
            self._listener_task = None
        if not self.running:
            return
        if self.reconnect_attempts < self.max_reconnect_attempts:
            self.reconnect_attempts += 1
            wait_time = min(30, 2 ** self.reconnect_attempts)
            logger.warning("Reconnecting attempt", attempt=self.reconnect_attempts, wait_time=wait_time)
            await asyncio.sleep(wait_time)
            await self._connect_websocket()
            if self.ws and self.running:
                self._listener_task = asyncio.create_task(self._listener(), name="BybitListener")
        else:
            logger.error("Max reconnect attempts reached. Stopping MarketData.")
            self.running = False

    async def _listener(self):
        while self.running and self.ws:
            async with self.recv_lock:
                try:
                    message = await self.ws.recv()
                    await self.message_queue.put(message)
                except Exception as e:
                    logger.error("Error receiving message", error=str(e))
                    await self._reconnect()

    async def _process_messages(self):
        while self.running:
            message = await self.message_queue.get()
            try:
                data = json.loads(message)
                self.last_message_time = datetime.now(timezone.utc)
                # Determine symbol from topic (e.g., "publicTrade.BTCUSD")
                topic = data.get('topic', '')
                if topic.startswith("publicTrade."):
                    symbol = topic.split(".")[1]
                    await self._process_trades(data.get('data'), symbol)
                else:
                    logger.debug("Received non-trade message", message=data)
            except Exception as e:
                logger.error("Error processing message", error=str(e))
            finally:
                self.message_queue.task_done()

    async def _process_trades(self, trade_data, symbol):
        if not self.running or Database._pool is None:
            return
        if not isinstance(trade_data, list):
            trade_data = [trade_data]
        inserted_count = 0
        for trade in trade_data:
            trade_id = trade['i']
            trade_time = datetime.fromtimestamp(int(trade['T']) / 1000, tz=timezone.utc)
            try:
                await Database.execute('''
                    INSERT INTO market_data (symbol, time, trade_id, price, volume)
                    VALUES ($1, $2, $3, $4, $5)
                    ON CONFLICT (symbol, time, trade_id) DO NOTHING
                ''', symbol, trade_time, trade_id, float(trade['p']), float(trade['v']))
                inserted_count += 1
            except Exception as e:
                logger.error("Failed to insert trade", error=str(e), trade_id=trade_id)
        if inserted_count > 0:
            logger.debug("Inserted new trades", count=inserted_count)
            
    async def _connection_monitor(self):
        while self.running:
            if self.last_message_time:
                delta = (datetime.now(timezone.utc) - self.last_message_time).total_seconds()
                if delta > 60:
                    logger.warning("No trade messages in 60s, reconnecting...")
                    await self._reconnect()
            await asyncio.sleep(10)
