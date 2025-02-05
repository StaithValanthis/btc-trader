from pybit.unified_trading import WebSocket
import asyncio
from structlog import get_logger
from app.core.config import Config
from app.core.database import Database
from datetime import datetime, timezone

logger = get_logger(__name__)

class BybitMarketData:
    def __init__(self, symbol: str = "BTCUSDT"):
        self.symbol = symbol
        self.ws = None
        self.running = False

    async def _connect_websocket(self):
        try:
            self.ws = WebSocket(
                testnet=False,
                channel_type="linear",
                api_key=Config.BYBIT_CONFIG['api_key'],
                api_secret=Config.BYBIT_CONFIG['api_secret']
            )
            
            def handle_message(message):
                asyncio.create_task(self._process_message(message))
                
            self.ws.trade_stream(
                symbol=self.symbol,
                callback=handle_message
            )
            logger.info("WebSocket connected successfully")
        except Exception as e:
            logger.error("WebSocket connection failed", error=str(e))
            raise

    async def _process_message(self, msg):
        try:
            if 'data' in msg:
                for trade in msg['data']:
                    price = float(trade['p'])
                    features = {
                        'size': trade['v'],
                        'side': trade['S'],
                        'trade_time': trade['T']
                    }
                    await Database.execute('''
                        INSERT INTO market_data (time, price, features)
                        VALUES ($1, $2, $3)
                    ''', datetime.now(timezone.utc), price, features)
        except Exception as e:
            logger.error("Message processing failed", error=str(e))

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