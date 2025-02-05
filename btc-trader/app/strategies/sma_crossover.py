import asyncio
import pandas as pd
from structlog import get_logger
from datetime import timedelta
from app.services.trade_service import TradeService
from app.core.config import Config
from app.core.database import Database

logger = get_logger(__name__)

class SMACrossover:
    def __init__(
        self,
        trade_service: TradeService,
        short_window: int = 20,
        long_window: int = 50,
        threshold: float = 0.001,
        cooldown: int = 300
    ):
        self.trade_service = trade_service
        self.short_window = short_window
        self.long_window = long_window
        self.threshold = threshold  # Minimum price movement percentage to act
        self.cooldown = cooldown    # Seconds between strategy checks
        self.last_trade_time = None
        self.running = False

    async def calculate_sma(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate SMA values with validation"""
        try:
            if len(df) < self.long_window:
                logger.warning("Insufficient data for SMA calculation", 
                              required=self.long_window, 
                              available=len(df))
                return pd.DataFrame()

            df['sma_short'] = df['price'].rolling(self.short_window).mean()
            df['sma_long'] = df['price'].rolling(self.long_window).mean()
            return df.dropna()
        except Exception as e:
            logger.error("SMA calculation failed", error=str(e))
            raise

    async def analyze(self, df: pd.DataFrame):
        """Analyze market data and execute trades"""
        try:
            df = await self.calculate_sma(df)
            if df.empty:
                return

            latest = df.iloc[-1]
            prev = df.iloc[-2]

            # Check cooldown period
            current_time = pd.Timestamp.now(tz='UTC')
            if self.last_trade_time and \
               (current_time - self.last_trade_time).total_seconds() < self.cooldown:
                return

            # Calculate price change percentage
            price_change = abs((latest['price'] - prev['price']) / prev['price'])

            # Generate signals
            if (latest['sma_short'] > latest['sma_long'] and 
                prev['sma_short'] <= prev['sma_long'] and
                price_change >= self.threshold):
                
                await self.trade_service.execute_trade(latest['price'], 'BUY')
                self.last_trade_time = current_time
                logger.info("BUY signal triggered", 
                           price=latest['price'], 
                           sma_short=latest['sma_short'],
                           sma_long=latest['sma_long'])

            elif (latest['sma_short'] < latest['sma_long'] and 
                  prev['sma_short'] >= prev['sma_long'] and
                  price_change >= self.threshold):
                
                await self.trade_service.execute_trade(latest['price'], 'SELL')
                self.last_trade_time = current_time
                logger.info("SELL signal triggered", 
                           price=latest['price'], 
                           sma_short=latest['sma_short'],
                           sma_long=latest['sma_long'])

        except Exception as e:
            logger.error("Strategy analysis failed", error=str(e))
            raise

    async def run(self):
        """Main strategy execution loop"""
        self.running = True
        logger.info("SMA Crossover strategy started", 
                   short_window=self.short_window,
                   long_window=self.long_window,
                   threshold=self.threshold)

        while self.running:
            try:
                # Get market data
                df = await self.trade_service.get_market_data(self.long_window * 2)
                
                if not df.empty:
                    # Convert time to datetime index
                    df = df.set_index(pd.to_datetime(df['time']))
                    df = df.sort_index(ascending=True)
                    
                    # Run analysis
                    await self.analyze(df)
                
                await asyncio.sleep(self.cooldown)

            except Exception as e:
                logger.error("Strategy loop error", error=str(e))
                await asyncio.sleep(10)

    async def stop(self):
        """Stop the strategy"""
        self.running = False
        logger.info("SMA Crossover strategy stopped")