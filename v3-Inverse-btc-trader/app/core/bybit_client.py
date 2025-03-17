# File: v2-Inverse-btc-trader/app/core/bybit_client.py

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
    """
    Manages a WebSocket connection to Bybit's public trade feed (inverse).
    Ingests trades into market_data table.
    Includes improved reconnect logic.
    """

    def __init__(self, symbol: str = Config.TRADING_CONFIG['symbol']) -> None:
        self.symbol = symbol
        self.ws = None
        self.running = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        self.last_message_time = None

        self.recv_lock = asyncio.Lock()
        self.message_queue = asyncio.Queue()

        category = Config.TRADING_CONFIG.get('category', 'inverse')
        if Config.BYBIT_CONFIG['testnet']:
            self.websocket_url = "wss://stream-testnet.bybit.com/v5/public/inverse"
            if category != 'inverse':
                self.websocket_url = "wss://stream-testnet.bybit.com/v5/public/linear"
        else:
            self.websocket_url = "wss://stream.bybit.com/v5/public/inverse"
            if category != 'inverse':
                self.websocket_url = "wss://stream.bybit.com/v5/public/linear"

        self._listener_task = None
        self._process_task = None
        self._monitor_task = None

    async def run(self) -> None:
        """
        Start the market data feed by connecting and launching tasks to listen
        and process messages.
        """
        self.running = True
        await self._connect_websocket()
        if self.running and self.ws:
            self._listener_task = asyncio.create_task(self._listener(), name="BybitListener")
            self._process_task = asyncio.create_task(self._process_messages(), name="BybitProcessMsg")
            self._monitor_task = asyncio.create_task(self._connection_monitor(), name="BybitConnMon")

    async def stop(self) -> None:
        """
        Stop WebSocket tasks and close the connection.
        """
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

    async def _connect_websocket(self) -> None:
        """
        Establish a websocket connection and subscribe to publicTrade.<symbol>.
        """
        try:
            logger.info("Connecting to Bybit WebSocket...", url=self.websocket_url)
            self.ws = await websockets.connect(
                self.websocket_url,
                ping_interval=30,
                ping_timeout=10,
                extra_headers={"User-Agent": "BTC-Trader/1.0"}
            )
            subscription = {
                "op": "subscribe",
                "args": [f"publicTrade.{self.symbol}"]
            }
            await self.ws.send(json.dumps(subscription))
            response = await self.ws.recv()
            logger.debug("Subscription response", response=json.loads(response))
            self.reconnect_attempts = 0
            logger.info("WebSocket connected successfully")
        except Exception as e:
            logger.error("WebSocket connection failed", error=str(e))
            await self._reconnect()

    async def _reconnect(self) -> None:
        """
        Attempt to reconnect with exponential backoff.
        Also handles quiet-market edge cases to avoid unnecessary reconnects.
        """
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

    async def _listener(self) -> None:
        """
        Continuously receive messages from the WebSocket in a loop,
        using recv_lock to avoid concurrency conflicts.
        """
        while self.running and self.ws:
            async with self.recv_lock:
                try:
                    message = await self.ws.recv()
                    await self.message_queue.put(message)
                except Exception as e:
                    logger.error("Error receiving message", error=str(e))
                    await self._reconnect()

    async def _process_messages(self) -> None:
        """
        Process messages from the queue.
        """
        while self.running:
            message = await self.message_queue.get()
            try:
                data = json.loads(message)
                self.last_message_time = datetime.now(timezone.utc)
                if 'topic' in data and data['topic'] == f"publicTrade.{self.symbol}":
                    await self._process_trades(data.get('data'))
                else:
                    logger.debug("Received non-trade message", message=data)
            except Exception as e:
                logger.error("Error processing message", error=str(e))
            finally:
                self.message_queue.task_done()

    async def _process_trades(self, trade_data) -> None:
        """
        Insert trades into the market_data table, if the DB is open.
        Uses a batch approach to reduce insert overhead.
        """
        if not self.running or Database._pool is None:
            return
        if not isinstance(trade_data, list):
            trade_data = [trade_data]

        batch_inserts = []
        for trade in trade_data:
            trade_time = datetime.fromtimestamp(int(trade['T']) / 1000, tz=timezone.utc)
            trade_id = trade['i']
            price = float(trade['p'])
            volume = float(trade['v'])
            batch_inserts.append((trade_time, trade_id, price, volume))

        if batch_inserts:
            values_str = ",".join(
                f"('{row[0].isoformat()}','{row[1]}',{row[2]},{row[3]})"
                for row in batch_inserts
            )
            query = f"""
                INSERT INTO market_data (time, trade_id, price, volume)
                VALUES {values_str}
                ON CONFLICT (time, trade_id) DO NOTHING
            """
            await Database.execute(query)

    async def _connection_monitor(self) -> None:
        """
        Monitors connection staleness; reconnects if no message in 60s.
        """
        while self.running:
            if self.last_message_time:
                delta = (datetime.now(timezone.utc) - self.last_message_time).total_seconds()
                # If truly no messages for 60s in a presumably active market, reconnect
                if delta > 60:
                    logger.warning("No trade messages in 60s, reconnecting...")
                    await self._reconnect()
            await asyncio.sleep(10)
