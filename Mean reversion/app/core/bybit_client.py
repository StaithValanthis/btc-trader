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
    def __init__(self):
        self.symbols = Config.TRADING_CONFIG['symbols']
        self.ws = None
        self.running = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        self.last_message_time = None
        self.recv_lock = asyncio.Lock()
        self.message_queue = asyncio.Queue()

        # Use the category from config (linear or inverse)
        base = "stream-testnet" if Config.BYBIT_CONFIG['testnet'] else "stream"
        cat  = Config.BYBIT_CONFIG['category']   # now 'linear' for USDT‐perps
        self.websocket_url = f"wss://{base}.bybit.com/v5/public/{cat}"

        self._listener_task = None
        self._process_task = None
        self._monitor_task = None

    async def run(self):
        self.running = True
        await self._connect_websocket()
        if self.ws:
            self._listener_task = asyncio.create_task(self._listener())
            self._process_task = asyncio.create_task(self._process_messages())
            self._monitor_task = asyncio.create_task(self._connection_monitor())

    async def stop(self):
        self.running = False
        for task in (self._listener_task, self._process_task, self._monitor_task):
            if task:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        if self.ws:
            await self.ws.close()
            self.ws = None
        logger.info("MarketData stopped")

    async def _connect_websocket(self):
        try:
            logger.info("Connecting WS", url=self.websocket_url)
            self.ws = await websockets.connect(
                self.websocket_url,
                ping_interval=30,
                ping_timeout=10,
                extra_headers={"User-Agent": "MeanRevBot/1.0"}
            )
            sub = {
                "op": "subscribe",
                "args": [f"publicTrade.{s}" for s in self.symbols]
            }
            await self.ws.send(json.dumps(sub))
            resp = await self.ws.recv()
            logger.debug("Sub resp", response=json.loads(resp))
            self.reconnect_attempts = 0
            logger.info("WebSocket connected")
        except Exception as e:
            logger.error("WS connect failed", error=str(e))
            await self._reconnect()

    async def _reconnect(self):
        if self.ws:
            await self.ws.close()
            self.ws = None
        if not self.running:
            return
        if self.reconnect_attempts < self.max_reconnect_attempts:
            self.reconnect_attempts += 1
            wait = min(30, 2 ** self.reconnect_attempts)
            logger.warning("Reconnecting", attempt=self.reconnect_attempts, wait=wait)
            await asyncio.sleep(wait)
            await self._connect_websocket()
            if self.ws:
                self._listener_task = asyncio.create_task(self._listener())
        else:
            logger.error("Max reconnects reached")
            self.running = False

    async def _listener(self):
        while self.running and self.ws:
            async with self.recv_lock:
                try:
                    msg = await self.ws.recv()
                    await self.message_queue.put(msg)
                except Exception as e:
                    logger.error("WS recv error", error=str(e))
                    await self._reconnect()

    async def _process_messages(self):
        while self.running:
            msg = await self.message_queue.get()
            try:
                data = json.loads(msg)
                self.last_message_time = datetime.now(timezone.utc)
                topic = data.get('topic', '')
                if topic.startswith("publicTrade."):
                    sym = topic.split('.', 1)[1]
                    await self._process_trades(data.get('data', []), sym)
            except Exception as e:
                logger.error("Msg process error", error=str(e))
            finally:
                self.message_queue.task_done()

    async def _process_trades(self, trade_data, symbol):
        if not Database._pool:
            return
        if not isinstance(trade_data, list):
            trade_data = [trade_data]
        for trade in trade_data:
            trade_id = trade['i']
            trade_time = datetime.fromtimestamp(int(trade['T'])/1000, tz=timezone.utc)
            price = float(trade['p'])
            volume = float(trade['v'])
            try:
                await Database.execute('''
                    INSERT INTO market_data (symbol, time, trade_id, price, volume)
                    VALUES ($1,$2,$3,$4,$5)
                    ON CONFLICT DO NOTHING
                ''', symbol, trade_time, trade_id, price, volume)
            except Exception as e:
                logger.error("Insert trade failed", error=str(e), symbol=symbol)

    async def _connection_monitor(self):
        while self.running:
            if self.last_message_time:
                delta = (datetime.now(timezone.utc) - self.last_message_time).total_seconds()
                if delta > 60:
                    logger.warning("No messages for 60s, reconnecting")
                    await self._reconnect()
            await asyncio.sleep(10)
