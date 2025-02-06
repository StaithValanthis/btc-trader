import asyncio
import pandas as pd
from structlog import get_logger
from datetime import datetime, timezone
from pybit.unified_trading import HTTP
from app.core.database import Database
from app.core.config import Config

logger = get_logger(__name__)

class TradeService:
    def __init__(self):
        self.position = None
        self.entry_price = None
        self.running = False
        self.session = HTTP(
            testnet=Config.BYBIT_CONFIG['testnet'],
            api_key=Config.BYBIT_CONFIG['api_key'],
            api_secret=Config.BYBIT_CONFIG['api_secret']
        )

    async def execute_trade(self, price: float, signal: str):
        """Execute a trade and update position tracking"""
        try:
            # Calculate position size (example: 0.001 BTC)
            qty = "0.001"
            
            # Place order
            order_params = {
                'category': 'linear',
                'symbol': 'BTCUSDT',
                'side': signal,
                'orderType': 'Market',
                'qty': qty,
            }
            
            response = await asyncio.to_thread(
                self.session.place_order,
                **order_params
            )
            
            # Calculate P&L if closing position
            pnl = None
            if signal == 'SELL' and self.position:
                pnl = (price - self.entry_price) * float(qty)
            
            # Record trade in database
            await Database.execute('''
                INSERT INTO trades (time, price, signal, profit_loss)
                VALUES ($1, $2, $3, $4)
            ''', datetime.now(timezone.utc), price, signal, pnl)
            
            # Update position tracking
            self.position = float(qty) if signal == 'BUY' else None
            self.entry_price = price if signal == 'BUY' else None
            
            logger.info("Trade executed", 
                       signal=signal, 
                       price=price, 
                       pnl=pnl,
                       response=response)
            return response
        except Exception as e:
            logger.error("Trade execution failed", error=str(e))
            raise

    async def get_market_data(self, limit: int = 100) -> pd.DataFrame:
        """Fetch market data from database"""
        try:
            records = await Database.fetch('''
                SELECT time, price 
                FROM market_data 
                ORDER BY time DESC 
                LIMIT $1
            ''', limit)
            
            return pd.DataFrame([dict(record) for record in records])
        except Exception as e:
            logger.error("Failed to fetch market data", error=str(e))
            raise

    async def get_open_orders(self):
        """Fetch current open orders"""
        try:
            response = await asyncio.to_thread(
                self.session.get_open_orders,
                category="linear",
                symbol="BTCUSDT"
            )
            return response.get('result', {}).get('list', [])
        except Exception as e:
            logger.error("Failed to fetch open orders", error=str(e))
            raise

    async def get_position_info(self):
        """Get current position information"""
        try:
            response = await asyncio.to_thread(
                self.session.get_positions,
                category="linear",
                symbol="BTCUSDT"
            )
            return response.get('result', {}).get('list', [])
        except Exception as e:
            logger.error("Failed to fetch position info", error=str(e))
            raise

    async def monitor_positions(self):
        """Continuous position monitoring"""
        self.running = True
        while self.running:
            try:
                # Check open orders
                orders = await self.get_open_orders()
                
                # Check current positions
                positions = await self.get_position_info()
                
                # Log status
                logger.info("Position monitoring",
                          open_orders=len(orders),
                          positions=len(positions))
                
                # Example: Close position if certain conditions are met
                if self.position and self.entry_price:
                    current_price = await self.get_current_price()
                    if current_price:
                        price_diff = current_price - self.entry_price
                        if abs(price_diff) > 100:  # Example threshold
                            await self.execute_trade(current_price, 'SELL')
                
                await asyncio.sleep(10)
            except Exception as e:
                logger.error("Position monitoring failed", error=str(e))
                await asyncio.sleep(5)

    async def get_current_price(self):
        """Get the latest market price"""
        try:
            records = await Database.fetch('''
                SELECT price 
                FROM market_data 
                ORDER BY time DESC 
                LIMIT 1
            ''')
            if records:
                return float(records[0]['price'])
            return None
        except Exception as e:
            logger.error("Failed to get current price", error=str(e))
            raise

    async def close_all_positions(self):
        """Close all open positions"""
        try:
            positions = await self.get_position_info()
            for position in positions:
                if float(position['size']) > 0:
                    await self.execute_trade(
                        float(position['markPrice']),
                        'SELL'
                    )
            logger.info("All positions closed")
        except Exception as e:
            logger.error("Failed to close positions", error=str(e))
            raise

    async def stop(self):
        """Stop the trade service"""
        self.running = False
        await self.close_all_positions()
        logger.info("Trade service stopped")