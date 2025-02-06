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

    async def get_current_price(self):
        records = await Database.fetch(
            "SELECT price FROM market_data ORDER BY time DESC LIMIT 1"
        )
        return float(records[0]['price']) if records else None

    async def execute_trade(self, price: float, side: str):
        try:
            position_idx = 1 if side == 'Buy' else 2
            await self.session.place_order(
                category="linear",
                symbol=Config.TRADING_CONFIG['symbol'],
                side=side,
                orderType="Market",
                qty=str(self.position_size),
                positionIdx=position_idx
            )
            
            # Update position tracking
            if side == 'Buy':
                self.long_position = self.position_size
            else:
                self.short_position = self.position_size
                
            # Log trade
            await Database.execute(
                "INSERT INTO trades (time, side, price, qty, position_idx) VALUES ($1, $2, $3, $4, $5)",
                datetime.now(timezone.utc), side, price, self.position_size, position_idx
            )
            
            logger.info("Trade executed",
                       side=side,
                       qty=self.position_size,
                       price=price,
                       position=position_idx)
        except Exception as e:
            logger.error("Trade failed", error=str(e))