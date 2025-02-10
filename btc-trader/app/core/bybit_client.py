import json
import asyncio
import websockets
from structlog import get_logger
from datetime import datetime, timezone
from app.core.database import Database
from app.core.config import Config

logger = get_logger(__name__)

class BybitMarketData:
    def __init__(self, strategy=None, symbol: str = Config.TRADING_CONFIG['symbol']):
        self.strategy = strategy
        self.symbol = symbol
        self.ws = None
        self.running = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        self.last_message_time = None

        # If you're using testnet, make sure the URL matches Bybit's testnet public endpoint
        # e.g. for linear instruments:
        # self.websocket_url = "wss://stream-testnet.bybit.com/v5/public/linear"
        #
        # If you're using mainnet for linear instruments:
        self.websocket_url = "wss://stream.bybit.com/v5/public/linear" if not Config.BYBIT_CONFIG['testnet'] \
            else "wss://stream-testnet.bybit.com/v5/public/linear"

    async def _connect_websocket(self):
        """Establish a websocket connection and subscribe."""
        try:
            logger.info("Connecting to Bybit WebSocket...", url=self.websocket_url)
            self.ws = await websockets.connect(
                self.websocket_url,
                ping_interval=30,
                ping_timeout=10,
                extra_headers={"User-Agent": "BTC-Trader/1.0"}
            )
            
            # Subscribe to public trades for self.symbol
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

    async def _reconnect(self):
        """Attempt to reconnect with exponential backoff up to max_reconnect_attempts."""
        if self.reconnect_attempts < self.max_reconnect_attempts:
            self.reconnect_attempts += 1
            wait_time = min(30, 2 ** self.reconnect_attempts)
            logger.warning(
                "Reconnecting attempt",
                attempt=self.reconnect_attempts,
                wait_time=wait_time
            )
            await asyncio.sleep(wait_time)
            await self._connect_websocket()
        else:
            logger.error("Max reconnect attempts reached. Stopping MarketData.")
            self.running = False

    async def _listen(self):
        """Listen for incoming messages from Bybit once the connection is up."""
        while self.running:
            if not self.ws:
                logger.warning("WebSocket is None, trying to reconnect...")
                await self._reconnect()
                if not self.ws:
                    logger.error("No active WebSocket after reconnect attempts; stopping _listen loop.")
                    self.running = False
                    break

            try:
                message = await self.ws.recv()
                self.last_message_time = datetime.now(timezone.utc)

                data = json.loads(message)
                if 'topic' in data and data['topic'] == f"publicTrade.{self.symbol}":
                    # data['data'] should be a list of trade objects
                    await self._process_message(data['data'])
                else:
                    logger.debug("Received non-trade message", message=data)

            except websockets.ConnectionClosed:
                logger.warning("WebSocket connection closed.")
                await self._reconnect()

            except Exception as e:
                logger.error("Message processing error", error=str(e))
                await asyncio.sleep(1)

    async def _process_message(self, trade_data):
        """Insert trade messages into the DB, referencing trade_id as unique key."""
        try:
            if not isinstance(trade_data, list):
                trade_data = [trade_data]

            logger.debug("Processing trades", count=len(trade_data))

            inserted_count = 0
            for trade in trade_data:
                # 'i' is typically the trade ID in Bybit v5 public trades
                trade_id = trade['i']  # unique ID
                trade_time = datetime.fromtimestamp(int(trade['T']) / 1000, tz=timezone.utc)

                try:
                    # Insert with ON CONFLICT to avoid duplicates if the same (time, trade_id) is seen again
                    result = await Database.execute('''
                        INSERT INTO market_data (time, trade_id, price, volume)
                        VALUES ($1, $2, $3, $4)
                        ON CONFLICT (time, trade_id) DO NOTHING
                    ''', trade_time, trade_id, float(trade['p']), float(trade['v']))

                    if result == "INSERT 0 1":
                        inserted_count += 1
                        logger.debug("Inserted new trade", trade_id=trade_id, time=trade_time.isoformat())
                except Exception as e:
                    logger.error("Failed to insert trade", error=str(e), trade_id=trade_id)

            # If using a progress bar in your strategy
            if inserted_count > 0 and self.strategy and hasattr(self.strategy, 'progress_bar'):
                count_result = await Database.fetch("SELECT COUNT(*) FROM market_data")
                current_count = count_result[0]['count'] if count_result else 0
                self.strategy.progress_bar.update(current_count)

        except Exception as e:
            logger.error("Trade processing failed", error=str(e))

    async def _connection_monitor(self):
        """Monitors the connection for staleness, attempts reconnect if no messages for 60s."""
        while self.running:
            if self.last_message_time:
                delta = (datetime.now(timezone.utc) - self.last_message_time).total_seconds()
                if delta > 60:
                    logger.warning("No messages in 60s, reconnecting...")
                    await self._reconnect()
            await asyncio.sleep(10)

    async def run(self):
        """Public method to start the Bybit data feed."""
        self.running = True
        await self._connect_websocket()

        # Start listening task & connection monitor
        if self.running:
            asyncio.create_task(self._listen())
            asyncio.create_task(self._connection_monitor())

    async def stop(self):
        """Stop the data feed."""
        self.running = False
        if self.ws:
            await self.ws.close()
            self.ws = None
        logger.info("Market data service stopped")
