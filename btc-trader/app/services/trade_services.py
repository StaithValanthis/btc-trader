from pybit.unified_trading import HTTP
from app.core.config import Config

class TradeService:
    def __init__(self):
        self.position = None
        self.entry_price = None
        self.session = HTTP(
            testnet=Config.BYBIT_CONFIG['testnet'],
            api_key=Config.BYBIT_CONFIG['api_key'],
            api_secret=Config.BYBIT_CONFIG['api_secret']
        )

    async def execute_trade(self, price: float, signal: str):
        try:
            # Place order logic
            order_params = {
                'category': 'linear',
                'symbol': 'BTCUSDT',
                'side': signal,
                'orderType': 'Market',
                'qty': '0.001',
            }
            
            if signal == 'BUY':
                response = self.session.place_order(**order_params)
            elif signal == 'SELL':
                response = self.session.place_order(**order_params)
            
            # Calculate P&L if closing position
            pnl = None
            if signal == 'SELL' and self.position:
                pnl = (price - self.entry_price) * self.position
            
            await Database.execute('''
                INSERT INTO trades (time, price, signal, profit_loss)
                VALUES ($1, $2, $3, $4)
            ''', datetime.now(timezone.utc), price, signal, pnl)
            
            self.position = 1.0 if signal == 'BUY' else None
            self.entry_price = price if signal == 'BUY' else None
            
            logger.info("Trade executed", 
                       signal=signal, 
                       price=price, 
                       pnl=pnl,
                       response=response)
        except Exception as e:
            logger.error("Trade execution failed", error=str(e))
            raise