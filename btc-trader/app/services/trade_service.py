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
        self.current_position = None

    async def initialize(self):
        """Initialize exchange connection with proper error handling"""
        try:
            self.session = HTTP(
                testnet=Config.BYBIT_CONFIG['testnet'],
                api_key=Config.BYBIT_CONFIG['api_key'],
                api_secret=Config.BYBIT_CONFIG['api_secret'],
                recv_window=5000                
            )
            if Config.TRADING_CONFIG['auto_position_mode']:
                await self._validate_position_mode()
            await self._get_min_order_qty()
            logger.info("Trade service initialized successfully")
        except Exception as e:
            logger.critical("Fatal error initializing trade service", error=str(e))
            raise

    async def _validate_position_mode(self):
        """Modern position mode validation for Bybit v5 API"""
        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                # Get current position mode
                current_mode = await asyncio.to_thread(
                    self.session.get_positions,
                    category="linear",
                    symbol=Config.TRADING_CONFIG['symbol']
                )
                
                # Parse response for position mode
                if current_mode.get('retCode') != 0:
                    logger.error("Failed to get positions", response=current_mode)
                    continue

                position_mode = current_mode['result']['list'][0].get('positionMode', 'MergedSingle')
                logger.debug("Current position mode", mode=position_mode)

                if position_mode == "MergedSingle":
                    logger.info("Already in one-way position mode")
                    return

                # Switch to one-way mode
                logger.info("Switching position mode")
                switch_response = await asyncio.to_thread(
                    self.session.set_position_mode,
                    category="linear",
                    symbol=Config.TRADING_CONFIG['symbol'],
                    mode=3  # 3 = One-Way Mode in Bybit v5
                )

                if switch_response.get('retCode') == 0:
                    logger.info("Successfully switched to one-way mode")
                    return
                    
                if switch_response.get('retCode') == 110025:
                    logger.info("Position mode already set")
                    return

            except Exception as e:
                if "110025" in str(e):
                    logger.info("Position mode already set (verified via exception)")
                    return
                logger.error(f"Position validation error (attempt {attempt+1}/{max_retries})", 
                            error=str(e))
                await asyncio.sleep(1)
        
        logger.error("Failed to validate position mode after multiple attempts")
        raise ConnectionError("Could not validate position mode")

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
        """Execute a market order with risk management"""
        try:
            if self.position_size < self.min_qty:
                logger.error("Position size below minimum", 
                            required=self.min_qty,
                            actual=self.position_size)
                return

            order_params = {
                "category": "linear",
                "symbol": Config.TRADING_CONFIG['symbol'],
                "side": side,
                "orderType": "Market",
                "qty": str(self.position_size),
                "positionIdx": 0,  # Mandatory for one-way mode
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
                logger.error("Trade failed", error=error_msg)

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
        """Clean up resources and close positions"""
        if self.current_position:
            logger.info("Closing open position", position=self.current_position)
            try:
                await self.execute_trade(
                    price=await self.get_current_price(),
                    side="Buy" if self.current_position == "sell" else "Sell"
                )
            except Exception as e:
                logger.error("Position close failed", error=str(e))
        
        if hasattr(self.session, 'close'):
            await asyncio.to_thread(self.session.close)
            
        logger.info("Trade service stopped")