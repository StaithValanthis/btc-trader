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
        self.current_position = None  # Track open positions

    async def initialize(self):
        """Initialize with better error handling"""
        try:
            self.session = HTTP(
                testnet=Config.BYBIT_CONFIG['testnet'],
                api_key=Config.BYBIT_CONFIG['api_key'],
                api_secret=Config.BYBIT_CONFIG['api_secret']
            )
            
            # Validate position mode first
            await self._validate_position_mode()
            
            # Then check minimum quantity
            await self._get_min_order_qty()
            
            logger.info("Trade service initialized")
        except Exception as e:
            logger.error("Trade service initialization failed", error=str(e))
            await self.stop()
            raise

    # app/services/trade_service.py
    async def _validate_position_mode(self):
        """Safely enforce one-way position mode with comprehensive checks"""
        try:
            # 1. Check current position mode
            mode_info = await asyncio.to_thread(
                self.session.get_position_mode,
                category="linear"
            )
            current_mode = mode_info['result']['mode']
            
            if current_mode == 0:
                logger.info("Already in one-way position mode")
                return

            # 2. Close all positions
            await self._close_all_positions()

            # 3. Cancel all orders
            await self._cancel_all_orders()

            # 4. Switch to one-way mode
            logger.info("Switching to one-way position mode")
            response = await asyncio.to_thread(
                self.session.switch_position_mode,
                category="linear",
                symbol=Config.TRADING_CONFIG['symbol'],
                mode=0
            )

            if response['retCode'] != 0:
                logger.warning("Position mode change response", response=response)

        except Exception as e:
            if '110025' in str(e):
                logger.info("Position mode already set to one-way")
            else:
                logger.error("Position mode validation failed", error=str(e))
                raise

    async def _close_all_positions(self):
        """Close all positions with market orders"""
        try:
            positions = await asyncio.to_thread(
                self.session.get_positions,
                category="linear",
                symbol=Config.TRADING_CONFIG['symbol']
            )
            
            for pos in positions['result']['list']:
                if float(pos['size']) > 0:
                    side = "Sell" if pos['side'] == "Buy" else "Buy"
                    await asyncio.to_thread(
                        self.session.place_order,
                        category="linear",
                        symbol=Config.TRADING_CONFIG['symbol'],
                        side=side,
                        orderType="Market",
                        qty=pos['size'],
                        reduceOnly=True
                    )
                    logger.info("Closed position", position=pos)

        except Exception as e:
            logger.error("Position closure failed", error=str(e))
            raise

    async def _cancel_all_orders(self):
        """Cancel all open orders"""
        try:
            orders = await asyncio.to_thread(
                self.session.get_open_orders,
                category="linear",
                symbol=Config.TRADING_CONFIG['symbol']
            )
            
            for order in orders['result']['list']:
                await asyncio.to_thread(
                    self.session.cancel_order,
                    category="linear",
                    symbol=Config.TRADING_CONFIG['symbol'],
                    orderId=order['orderId']
                )
                logger.info("Cancelled order", order=order)
                
        except Exception as e:
            logger.error("Order cancellation failed", error=str(e))
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

    async def execute_trade(self, price: float, side: str, 
                          stop_loss: float = None, 
                          take_profit: float = None):
        """
        Execute a market order with risk management parameters
        """
        try:
            if self.position_size < self.min_qty:
                logger.error("Position size below minimum", 
                            required=self.min_qty,
                            actual=self.position_size)
                return

            logger.info("Placing trade",
                        side=side,
                        price=price,
                        size=self.position_size,
                        stop_loss=stop_loss,
                        take_profit=take_profit)

            order_params = {
                "category": "linear",
                "symbol": Config.TRADING_CONFIG['symbol'],
                "side": side,
                "orderType": "Market",
                "qty": str(self.position_size),
                "position_idx": 0,  # Explicit one-way mode
                "stopLoss": str(stop_loss) if stop_loss else None,
                "takeProfit": str(take_profit) if take_profit else None
            }

            # Clean null values
            order_params = {k: v for k, v in order_params.items() if v is not None}

            response = await asyncio.to_thread(
                self.session.place_order,
                **order_params
            )

            if response['retCode'] == 0:
                await self._log_trade(side, price, self.position_size)
                logger.info("Trade executed successfully",
                            order_id=response['result']['orderId'])
                self.current_position = side.lower()
            else:
                error_msg = response.get('retMsg', 'Unknown error')
                logger.error("Trade failed", 
                            error=error_msg,
                            response=response)

        except Exception as e:
            logger.error("Trade execution error", error=str(e))
            self.current_position = None
            raise

    async def _log_trade(self, side: str, price: float, qty: float):
        """Store trade in database"""
        try:
            await Database.execute('''
                INSERT INTO trades 
                (time, side, price, quantity)
                VALUES ($1, $2, $3, $4)
            ''', datetime.now(timezone.utc), side, price, qty)
            logger.info("Trade logged to database")
        except Exception as e:
            logger.error("Failed to log trade", error=str(e))

    async def get_current_price(self):
        """Get latest market price from database"""
        try:
            record = await Database.fetchrow(
                "SELECT price FROM market_data "
                "ORDER BY time DESC LIMIT 1"
            )
            return float(record['price']) if record else None
        except Exception as e:
            logger.error("Price fetch failed", error=str(e))
            return None

    async def stop(self):
        """Clean up resources and close position"""
        if self.current_position:
            logger.info("Closing open position", position=self.current_position)
            try:
                await self.execute_trade(
                    price=await self.get_current_price(),
                    side="Buy" if self.current_position == "sell" else "Sell",
                    stop_loss=None,
                    take_profit=None
                )
            except Exception as e:
                logger.error("Position close failed", error=str(e))
        
        if hasattr(self.session, 'close'):
            await asyncio.to_thread(self.session.close)
            
        logger.info("Trade service stopped")

    # app/services/trade_service.py
    async def _validate_position_mode(self):
        """Validate and enforce one-way position mode"""
        try:
            # Switch directly to one-way mode
            logger.info("Enforcing one-way position mode")
            await asyncio.to_thread(
                self.session.switch_position_mode,
                category="linear",
                symbol=Config.TRADING_CONFIG['symbol'],
                mode=0  # 0 = One-Way Mode
            )
        except Exception as e:
            logger.error("Position mode enforcement failed", error=str(e))
            raise