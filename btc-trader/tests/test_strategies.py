# tests/test_strategies.py
import pytest
import pandas as pd
from app.strategies.lstm_strategy import LSTMStrategy

@pytest.mark.asyncio
async def test_lstm_strategy():
    trade_service = TradeService()
    strategy = LSTMStrategy(trade_service)
    data = pd.DataFrame({
        'price': [100, 101, 102, 103, 104, 105]
    })
    await strategy.analyze(data)