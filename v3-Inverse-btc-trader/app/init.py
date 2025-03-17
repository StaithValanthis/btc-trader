# File: v2-Inverse-btc-trader/app/init.py

import asyncio
from app.utils.logger import logger
from app.core import Database, BybitMarketData, Config
from app.services import TradeService, CandleService
from app.debug.startup_check import StartupChecker

class TradingBot:
    """
    Main orchestrator of the trading bot.
    Initializes and runs database, data feed, candle service, trade service, etc.
    """

    def __init__(self) -> None:
        self.bybit = BybitMarketData()
        self.trade_service = TradeService()
        self.candle_service = CandleService(interval_seconds=3600)  # Aggregates 1-hour candles
        self.running = False

    async def run(self) -> None:
        """
        Run the trading bot by initializing services and concurrently running
        the Bybit data feed, monitoring, and the trading loop.
        """
        try:
            await StartupChecker.run_checks()
            logger.info("Initializing database...")
            await Database.initialize()
            await self.trade_service.initialize()
            await self.candle_service.start()
            self.running = True
            logger.info("Starting main tasks...")
            await asyncio.gather(
                self.bybit.run(),
                self._monitor_system(),
                self._trading_loop()
            )
        except Exception as e:
            logger.error("Bot failed to start", error=str(e))
            raise
        finally:
            await self.stop()

    async def _monitor_system(self) -> None:
        """
        Periodically logs system status (WS connection, DB status, etc.) every 30s.
        """
        while self.running:
            try:
                ws_status = {
                    "connected": self.bybit.ws is not None,
                    "last_message": self.bybit.last_message_time,
                    "reconnect_attempts": self.bybit.reconnect_attempts
                }
                db_status = await Database.fetch("SELECT 1")
                logger.info("System status", ws_status=ws_status, db_status=db_status)
            except Exception as e:
                logger.error("Monitoring error", error=str(e))
            await asyncio.sleep(30)

    async def _trading_loop(self) -> None:
        """
        Invokes the trade logic from TradeService every 1 hour.
        """
        while self.running:
            await self.trade_service.run_trading_logic()
            await asyncio.sleep(3600)

    async def stop(self) -> None:
        """
        Gracefully stop all services (WebSocket, Candle aggregation, Trade service, DB).
        """
        self.running = False
        await self.bybit.stop()
        await self.candle_service.stop()
        await self.trade_service.stop()
        await Database.close()
        logger.info("Trading bot fully stopped")
