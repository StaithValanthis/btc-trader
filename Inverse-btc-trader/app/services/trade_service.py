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
        self.current_position = None

    async def initialize(self):
        """Initialize exchange connection for Bybit inverse trading."""
        try:
            self.session = HTTP(
                testnet=Config.BYBIT_CONFIG['testnet'],
                api_key=Config.BYBIT_CONFIG['api_key'],
                api_secret=Config.BYBIT_CONFIG['api_secret'],
                recv_window=5000
            )
            await self._get_min_order_qty()
            logger.info("Trade service initialized successfully")
        except Exception as e:
            logger.critical("Fatal error initializing trade service", error=str(e))
            raise

    async def _get_min_order_qty(self):
        """Fetch minimum order quantity for inverse BTCUSD contract."""
        if not self.min_qty:
            info = await asyncio.to_thread(
                self.session.get_instruments_info,
                category="inverse",
                symbol=Config.TRADING_CONFIG['symbol']
            )
            self.min_qty = float(info['result']['list'][0]['lotSizeFilter']['minOrderQty'])
            logger.info("Loaded min order qty", min_qty=self.min_qty)

    async def execute_trade(self, side: str, price: float, stop_loss: float=None, take_profit: float=None):
        """
        Execute a market order on Bybit inverse BTCUSD.
        
        :param side: "Buy" or "Sell"
        :param price: Current market price (used only for logging)
        :param stop_loss: optional stop loss price
        :param take_profit: optional take profit price
        """
        position_size = self.position_size
        if position_size < self.min_qty:
            logger.error("Position size below minimum",
                         required=self.min_qty,
                         actual=position_size)
            return

        logger.info("Placing trade",
                    side=side,
                    price=price,
                    size=position_size,
                    stop_loss=stop_loss,
                    take_profit=take_profit)

        order_params = {
            "category": "inverse",
            "symbol": Config.TRADING_CONFIG['symbol'],
            "side": side,
            "orderType": "Market",
            "qty": str(position_size),
            "stopLoss": str(stop_loss) if stop_loss else None,
            "takeProfit": str(take_profit) if take_profit else None
        }
        order_params = {k: v for k, v in order_params.items() if v is not None}

        try:
            response = await asyncio.to_thread(
                self.session.place_order,
                **order_params
            )

            if response['retCode'] == 0:
                await self._log_trade(side, price, position_size)
                logger.info("Trade executed successfully",
                            order_id=response['result']['orderId'])
                self.current_position = side.lower()
            else:
                error_msg = response.get('retMsg', 'Unknown error')
                logger.error("Trade failed", error=error_msg)
        except Exception as e:
            logger.error("Trade execution error", error=str(e))
            self.current_position = None

    async def _log_trade(self, side: str, price: float, qty: float):
        """Store trade in the 'trades' table."""
        try:
            await Database.execute('''
                INSERT INTO trades (time, side, price, quantity)
                VALUES ($1, $2, $3, $4)
            ''', datetime.now(timezone.utc), side, price, qty)
        except Exception as e:
            logger.error("Failed to log trade", error=str(e))

    async def stop(self):
        """Stop trade service (optionally close position)."""
        if hasattr(self.session, 'close'):
            await asyncio.to_thread(self.session.close)
        logger.info("Trade service stopped")
