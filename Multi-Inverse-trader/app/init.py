import asyncio
from app.utils.logger import logger
from app.core import Database, BybitMarketData, Config
from app.services import TradeService, CandleService
from app.debug.startup_check import StartupChecker

class TradingBot:
    def __init__(self, symbol):
        self.symbol = symbol
        # Create a market data client for this symbol only
        self.bybit = BybitMarketData(symbols=[symbol])
        self.trade_service = TradeService(symbol)
        self.candle_service = CandleService(symbol, interval_seconds=60)  # Aggregates 1-minute candles
        self.running = False

    async def run(self):
        try:
            await StartupChecker.run_checks(symbol=self.symbol)
            logger.info(f"Initializing database for {self.symbol}...")
            await Database.initialize()
            await self.trade_service.initialize()
            await self.candle_service.start()
            self.running = True
            logger.info(f"Starting main tasks for {self.symbol}...")
            await asyncio.gather(
                self.bybit.run(),
                self._monitor_system(),
                self._trading_loop()
            )
        except Exception as e:
            logger.error("Bot failed to start", error=str(e), symbol=self.symbol)
            raise
        finally:
            await self.stop()

    async def _monitor_system(self):
        while self.running:
            try:
                ws_status = {
                    "connected": self.bybit.ws is not None,
                    "last_message": self.bybit.last_message_time,
                    "reconnect_attempts": self.bybit.reconnect_attempts
                }
                db_status = await Database.fetch("SELECT 1")
                logger.info("System status", ws_status=ws_status, db_status=db_status, symbol=self.symbol)
            except Exception as e:
                logger.error("Monitoring error", error=str(e), symbol=self.symbol)
            await asyncio.sleep(30)

    async def _trading_loop(self):
        while self.running:
            await self.trade_service.run_trading_logic()
            await asyncio.sleep(3600)

    async def stop(self):
        self.running = False
        await self.bybit.stop()
        await self.candle_service.stop()
        await self.trade_service.stop()
        await Database.close()
        logger.info("Trading bot fully stopped", symbol=self.symbol)

if __name__ == "__main__":
    from app import TradingBot
    import sys
    symbol = sys.argv[1] if len(sys.argv) > 1 else Config.TRADING_CONFIG['symbols'][0]
    asyncio.run(TradingBot(symbol).run())
