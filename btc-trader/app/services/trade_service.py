import asyncio
from pybit.unified_trading import HTTP
from structlog import get_logger
from datetime import datetime, timezone
from app.core import Config, Database

logger = get_logger(__name__)

class TradeService:
    def __init__(self):
        self.long_position = 0.0
        self.short_position = 0.0
        self.session = HTTP(
            testnet=Config.BYBIT_CONFIG['testnet'],
            api_key=Config.BYBIT_CONFIG['api_key'],
            api_secret=Config.BYBIT_CONFIG['api_secret']
        )
        self.position_size = Config.TRADING_CONFIG['position_size']
        self.min_qty = None

    async def _get_min_order_qty(self):
        """Fetch minimum order quantity from Bybit"""
        if not self.min_qty:
            info = await asyncio.to_thread(
                self.session.get_instruments_info,
                category="linear",
                symbol=Config.TRADING_CONFIG['symbol']
            )
            self.min_qty = float(info['result']['list'][0]['lotSizeFilter']['minOrderQty'])
        return self.min_qty

    async def execute_trade(self, price: float, side: str):
        """Execute a trade on Bybit with slippage protection"""
        try:
            min_qty = await self._get_min_order_qty()
            if self.position_size < min_qty:
                logger.error("Position size below minimum", required=min_qty)
                return

            current_price = await self.get_current_price()
            if abs(price/current_price - 1) > 0.005:
                logger.warning("Price slippage too high, aborting trade")
                return

            order_params = {
                "category": "linear",
                "symbol": Config.TRADING_CONFIG['symbol'],
                "side": side,
                "orderType": "Market",
                "qty": str(self.position_size),
            }

            response = await asyncio.to_thread(
                self.session.place_order,
                **order_params
            )

            if response['retCode'] == 0:
                await self._log_trade(side, price)
                self._update_position(side)
                logger.info("Trade executed successfully", side=side, price=price)
            else:
                logger.error("Trade failed", error=response['retMsg'])

        except Exception as e:
            logger.error("Trade execution error", error=str(e))

    async def _log_trade(self, side: str, price: float):
        """Log trade in database"""
        await Database.execute(
            "INSERT INTO trades (time, side, price, qty) VALUES ($1, $2, $3, $4)",
            datetime.now(timezone.utc),
            side,
            price,
            self.position_size
        )

    def _update_position(self, side: str):
        """Update local position tracking"""
        if side == 'Buy':
            self.long_position = self.position_size
        else:
            self.short_position = self.position_size

    async def get_current_price(self):
        """Fetch latest market price from database"""
        records = await Database.fetch(
            "SELECT price FROM market_data ORDER BY time DESC LIMIT 1"
        )
        return float(records[0]['price']) if records else None

    async def get_open_orders(self):
        """Fetch current open orders"""
        try:
            response = await asyncio.to_thread(
                self.session.get_open_orders,
                category="linear",
                symbol=Config.TRADING_CONFIG['symbol']
            )
            return response.get('result', {}).get('list', [])
        except Exception as e:
            logger.error("Failed to fetch open orders", error=str(e))
            return []

    async def get_position_info(self):
        """Get current position information"""
        try:
            response = await asyncio.to_thread(
                self.session.get_positions,
                category="linear",
                symbol=Config.TRADING_CONFIG['symbol']
            )
            return response.get('result', {}).get('list', [])
        except Exception as e:
            logger.error("Failed to fetch position info", error=str(e))
            return []

    async def close_all_positions(self):
        """Close all open positions safely"""
        try:
            positions = await self.get_position_info()
            for position in positions:
                if float(position.get('size', 0)) == 0:
                    continue
                
                side = 'Sell' if position.get('side') == 'Buy' else 'Buy'
                await self.execute_trade(
                    float(position.get('avgPrice', 0)),
                    side
                )
            logger.info("All positions closed")
        except Exception as e:
            logger.error("Failed to close positions", error=str(e))
            raise

    async def stop(self):
        """Clean up resources"""
        try:
            await self.close_all_positions()
            if hasattr(self.session, 'close'):
                await asyncio.to_thread(self.session.close)
            logger.info("Trade service stopped successfully")
        except Exception as e:
            logger.error("Error stopping trade service", error=str(e))