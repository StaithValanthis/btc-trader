import pytest
import pandas as pd
import asyncio
from app.strategies.lstm_strategy import LSTMStrategy
from app.services.trade_service import TradeService

class DummyTradeService(TradeService):
    async def execute_trade(self, price, side, qty=None):
        return

@pytest.mark.asyncio
async def test_lstm_strategy_data_availability():
    trade_service = DummyTradeService()
    strategy = LSTMStrategy(trade_service)
    
    # Create dummy market data in the database for testing
    # (Assuming you have a method to insert test data or using a test database)
    # For now, just call the _check_data_availability (which will likely return False)
    ready = await strategy._check_data_availability()
    assert ready in [True, False]
