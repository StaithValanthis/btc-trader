from app.utils.logger import logger
from app.core import Database, BybitMarketData
from app.services import TradeService
from app.strategies import LSTMStrategy
from app.debug.startup_check import StartupChecker
import asyncio
import nest_asyncio
import time

nest_asyncio.apply()

class TradingBot:
    def __init__(self):
        self.trade_service = TradeService()
        self.strategy = LSTMStrategy(self.trade_service)
        self.bybit = BybitMarketData(strategy=self.strategy)  # Pass strategy reference
        self.running = False

    async def run(self):
        try:
            # Run comprehensive startup checks
            await StartupChecker.run_checks()
            
            logger.info("Initializing database...")
            await Database.initialize()
            
            self.running = True
            logger.info("Starting main components...")
            
            await asyncio.gather(
                self.bybit.run(),
                self.strategy.run(),
                self._monitor_system()
            )
        except Exception as e:
            logger.error("Bot failed to start", error=str(e))
            raise
        finally:
            await self.stop()

    async def _monitor_system(self):
        """Continuous system health monitoring"""
        while self.running:
            try:
                # Convert all datetime objects to ISO strings
                last_msg = self.bybit.last_message_time.isoformat() if self.bybit.last_message_time else None
                warmup_start = datetime.fromtimestamp(
                    self.strategy.warmup_start_time
                ).isoformat() if self.strategy.warmup_start_time else None
                
                logger.info("System status", 
                    ws_status={
                        "connected": self.bybit.ws is not None,
                        "last_message": last_msg,
                        "reconnect_attempts": self.bybit.reconnect_attempts
                    },
                    db_status="OK" if await self._check_db() else "Error",
                    strategy_status={
                        "data_ready": self.strategy.data_ready,
                        "model_loaded": self.strategy.model_loaded,
                        "warmup_start": warmup_start,
                        "samples_collected": await self._get_data_count()
                    }
                )
                await asyncio.sleep(60)

    async def _check_db(self):
        """Simple database health check"""
        try:
            await Database.fetch("SELECT 1")
            return True
        except:
            return False

    async def _get_data_count(self):
        """Get current market data count"""
        result = await Database.fetch("SELECT COUNT(*) FROM market_data")
        return result[0]['count'] if result else 0

    async def stop(self):
        self.running = False
        await self.bybit.stop()
        await self.trade_service.stop()
        await Database.close()
        logger.info("Trading bot fully stopped")

async def main():
    bot = TradingBot()
    await bot.run()

if __name__ == "__main__":
    asyncio.run(main())