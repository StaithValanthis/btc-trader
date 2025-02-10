# app/services/trade_service.py
import asyncio
from pybit.unified_trading import HTTP
from structlog import get_logger
from datetime import datetime, timezone
from app.core import Database, Config

logger = get_logger(__name__)

class TradeService:
    def __init__(self):
        self.session = None
        self.position_size = Config.TRADING_CONFIG['position_size']
        self.min_qty = None
        self.running = False

    async def initialize(self):
        """Initialize exchange connection"""
        try:
            self.session = HTTP(
                testnet=Config.BYBIT_CONFIG['testnet'],
                api_key=Config.BYBIT_CONFIG['api_key'],
                api_secret=Config.BYBIT_CONFIG['api_secret']
            )
            await self._get_min_order_qty()
            logger.info("Trade service initialized")
        except Exception as e:
            logger.error("Trade service initialization failed", error=str(e))
            raise

    async def _get_min_order_qty(self):
        """Fetch minimum order quantity"""
        if not self.min_qty:
            info = await asyncio.to_thread(
                self.session.get_instruments_info,
                category="linear",
                symbol=Config.TRADING_CONFIG['symbol']
            )
            self.min_qty = float(info['result']['list'][0]['lotSizeFilter']['minOrderQty'])

    async def execute_trade(self, price: float, side: str):
        """Execute a market order"""
        try:
            if self.position_size < self.min_qty:
                logger.error("Position size below minimum", required=self.min_qty)
                return

            response = await asyncio.to_thread(
                self.session.place_order,
                category="linear",
                symbol=Config.TRADING_CONFIG['symbol'],
                side=side,
                orderType="Market",
                qty=str(self.position_size)
            )

            if response['retCode'] == 0:
                await self._log_trade(side, price)
                logger.info("Trade executed", side=side, price=price)
            else:
                logger.error("Trade failed", error=response['retMsg'])

        except Exception as e:
            logger.error("Trade execution error", error=str(e))

    async def get_current_price(self):
        """Get latest market price from database"""
        try:
            record = await Database.fetchrow(
                "SELECT price FROM market_data ORDER BY time DESC LIMIT 1"
            )
            return float(record['price']) if record else None
        except Exception as e:
            logger.error("Price fetch failed", error=str(e))
            return None

    async def stop(self):
        """Clean up resources"""
        if hasattr(self.session, 'close'):
            await asyncio.to_thread(self.session.close)
        logger.info("Trade service stopped")