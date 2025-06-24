import asyncio, json, websockets
from structlog import get_logger
from app.core.config import Config

logger = get_logger(__name__)

def get_ws_url():
    base = "stream-testnet" if Config["BYBIT"]["testnet"] else "stream"
    return f"wss://{base}.bybit.com/v5/public/{Config['BYBIT']['category']}"

async def bybit_ws_subscribe(ws, topics):
    await ws.send(json.dumps({"op":"subscribe","args":topics}))
    resp = await ws.recv()
    logger.debug("Subscribed", response=json.loads(resp))

async def start_bybit_ws(topics, message_handler):
    url = get_ws_url()
    async with websockets.connect(url, ping_interval=30, ping_timeout=10) as ws:
        logger.info("WS connected", url=url)
        await bybit_ws_subscribe(ws, topics)
        async for raw in ws:
            msg = json.loads(raw)
            await message_handler(msg)
