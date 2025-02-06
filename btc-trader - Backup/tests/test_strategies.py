import pytest
import pandas as pd
from app.strategies.sma_crossover import SMACrossover

@pytest.mark.asyncio
async def test_sma_crossover():
    strategy = SMACrossover(short_window=2, long_window=4)
    data = pd.DataFrame({
        'price': [100, 101, 102, 103, 104, 105]
    })
    await strategy.analyze(data)