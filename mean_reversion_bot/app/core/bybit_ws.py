# File: app/core/bybit_ws.py

import asyncio
import json
import websockets
from structlog import get_logger

from app.core.config import Config

logger = get_logger(__name__)

class BybitWebSocket:
    """
    Async WebSocket client with topic-based subscription management.
    Emits parsed dict messages to all subscribers.
    """
    def __init__(self):
        base = "stream-testnet" if Config.BYBIT_CONFIG["testnet"] else "stream"
        cat  = Config.BYBIT_CONFIG["category"]
        self.url = f"wss://{base}.bybit.com/v5/public/{cat}"
        self.ws = None
        self._subscribed = set()
        self._topics = set()
        self._msg_queue = asyncio.Queue()
        self._listeners = []
        self._listen_task = None
        self._recv_task = None
        self._lock = asyncio.Lock()
        self.running = False

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.close()

    async def connect(self):
        self.ws = await websockets.connect(
            self.url,
            ping_interval=30,
            ping_timeout=10,
            extra_headers={"User-Agent": "MeanRevBot/1.0"}
        )
        self.running = True
        self._recv_task = asyncio.create_task(self._receiver())
        self._listen_task = asyncio.create_task(self._emitter())
        logger.info("BybitWebSocket connected", url=self.url)

    async def close(self):
        self.running = False
        if self.ws:
            await self.ws.close()
            self.ws = None
        for t in [self._recv_task, self._listen_task]:
            if t:
                t.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await t
        logger.info("BybitWebSocket closed")

    async def subscribe(self, topics):
        async with self._lock:
            new_topics = set(topics) - self._topics
            if not new_topics:
                return
            sub = {"op": "subscribe", "args": list(new_topics)}
            await self.ws.send(json.dumps(sub))
            self._topics.update(new_topics)
            logger.info("Subscribed to topics", topics=list(new_topics))

    def add_listener(self, coro):
        """Register a coroutine to receive every parsed message."""
        self._listeners.append(coro)

    async def _receiver(self):
        while self.running and self.ws:
            try:
                msg = await self.ws.recv()
                self._msg_queue.put_nowait(msg)
            except websockets.ConnectionClosed:
                logger.warning("BybitWebSocket connection closed, exiting recv loop")
                self.running = False
            except Exception as e:
                logger.error("BybitWebSocket recv error", error=str(e))

    async def _emitter(self):
        while self.running:
            msg = await self._msg_queue.get()
            try:
                data = json.loads(msg)
                for cb in self._listeners:
                    # fire-and-forget (don't block on listeners)
                    asyncio.create_task(cb(data))
            except Exception as e:
                logger.error("BybitWebSocket parse error", error=str(e))
            finally:
                self._msg_queue.task_done()
