from app.utils.logger import logger
from app.core import Database, BybitMarketData
from app.services.trade_service import TradeService
from app.strategies.lstm_strategy import LSTMStrategy
import asyncio
import nest_asyncio
import time

nest_asyncio.apply()

class TradingBot:
    """Alternate location for the TradingBot class (if needed)."""
    def __init__(self):
        self.trade_service = TradeService()
        self.strategy = LSTMStrategy(self.trade_service)
        self.bybit = BybitMarketData(strategy=self.strategy)
        self.running = False

    async def run(self):
        try:
            from app.debug.startup_check import StartupChecker
            await StartupChecker.run_checks()
            await Database.initialize()
            self.running = True
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
        while self.running:
            try:
                last_msg = self.bybit.last_message_time.isoformat() if self.bybit.last_message_time else None
                warmup_start = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(self.strategy.warmup_start_time)) if self.strategy.warmup_start_time else None
                db_ok = "OK" if await self._check_db() else "Error"
                data_count = await self._get_data_count()
                logger.info("System status", 
                            ws_status={
                                "connected": self.bybit.ws is not None,
                                "last_message": last_msg,
                                "reconnect_attempts": self.bybit.reconnect_attempts
                            },
                            db_status=db_ok,
                            strategy_status={
                                "data_ready": self.strategy.data_ready,
                                "model_loaded": self.strategy.model_loaded,
                                "warmup_start": warmup_start,
                                "samples_collected": data_count
                            })
                await asyncio.sleep(60)
            except Exception as e:
                logger.error("Monitoring error", error=str(e))
                await asyncio.sleep(10)

    async def _check_db(self) -> bool:
        try:
            await Database.fetch("SELECT 1")
            return True
        except Exception:
            return False

    async def _get_data_count(self) -> int:
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
