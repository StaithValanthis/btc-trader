import asyncio, json, websockets
from structlog import get_logger
from app.core.config import Config

logger = get_logger(__name__)

def ws_url():
    base = "stream-testnet" if Config["BYBIT"]["testnet"] else "stream"
    return f"wss://{base}.bybit.com/v5/public/{Config['BYBIT']['category']}"

async def subscribe(ws, new_topics, subscribed):
    to_add = set(new_topics) - subscribed
    if to_add:
        await ws.send(json.dumps({"op":"subscribe","args":list(to_add)}))
        subscribed |= to_add
        logger.info("Subscribed topics", topics=list(to_add))
    return subscribed

async def receive_loop(ws, queue):
    async for raw in ws:
        await queue.put(json.loads(raw))

async def emitter(queue, handlers):
    while True:
        msg = await queue.get()
        for h in handlers:
            asyncio.create_task(h(msg))
        queue.task_done()
