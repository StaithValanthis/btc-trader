import pandas as pd
from structlog import get_logger
from app.services.trade_service import TradeService

logger = get_logger(__name__)

class SMACrossover:
    def __init__(self, short_window=20, long_window=50):
        self.short_window = short_window
        self.long_window = long_window
        self.trade_service = TradeService()

    async def analyze(self, df: pd.DataFrame):
        try:
            if len(df) < self.long_window:
                return
                
            df['sma_short'] = df['price'].rolling(self.short_window).mean()
            df['sma_long'] = df['price'].rolling(self.long_window).mean()
            
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            
            if (latest['sma_short'] > latest['sma_long'] and 
                prev['sma_short'] <= prev['sma_long']):
                await self.trade_service.execute_trade(latest['price'], 'BUY')
                
            elif (latest['sma_short'] < latest['sma_long'] and 
                  prev['sma_short'] >= prev['sma_long']):
                await self.trade_service.execute_trade(latest['price'], 'SELL')
                
        except Exception as e:
            logger.error("Strategy analysis failed", error=str(e))
            raise