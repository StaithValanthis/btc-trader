import asyncio
from database import Database
from bybit_client import BybitMarketData
import pandas as pd
from typing import Optional

class TradingBot:
    def __init__(self):
        self.bybit = BybitMarketData()
        self.position: Optional[float] = None
        self.entry_price: Optional[float] = None

    async def analyze_market_data(self):
        while True:
            try:
                # Example strategy: Simple Moving Average crossover
                async with Database.get_pool().acquire() as conn:
                    data = await conn.fetch('''
                        SELECT time, price 
                        FROM market_data 
                        ORDER BY time DESC 
                        LIMIT 100
                    ''')
                
                df = pd.DataFrame([dict(record) for record in data])
                df['sma20'] = df['price'].rolling(20).mean()
                df['sma50'] = df['price'].rolling(50).mean()

                if len(df) >= 50:
                    latest = df.iloc[-1]
                    if latest['sma20'] > latest['sma50'] and not self.position:
                        # Buy signal
                        await self.execute_trade(latest['price'], 'BUY')
                    elif latest['sma20'] < latest['sma50'] and self.position:
                        # Sell signal
                        await self.execute_trade(latest['price'], 'SELL')

                await asyncio.sleep(60)  # Analyze every minute
            except Exception as e:
                print(f"Analysis error: {e}")
                await asyncio.sleep(10)

    async def execute_trade(self, price: float, signal: str):
        try:
            # Calculate P&L if closing position
            pnl = None
            if signal == 'SELL' and self.position:
                pnl = (price - self.entry_price) * self.position
            
            # Record trade
            await Database.insert_trade(price, signal, pnl)
            
            # Update position
            self.position = 1.0 if signal == 'BUY' else None
            self.entry_price = price if signal == 'BUY' else None
            
            print(f"Executed {signal} at {price} | P&L: {pnl}")
        except Exception as e:
            print(f"Trade execution error: {e}")

async def main():
    await Database.initialize()
    bot = TradingBot()
    await asyncio.gather(
        bot.bybit.run(),
        bot.analyze_market_data()
    )

if __name__ == "__main__":
    import nest_asyncio
    nest_asyncio.apply()
    asyncio.run(main())