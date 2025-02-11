import pytest
import asyncio
from app.core.database import Database

@pytest.mark.asyncio
async def test_database_connection():
    await Database.initialize()
    try:
        result = await Database.fetch("SELECT 1 as value")
        # Check that a result is returned and contains the expected key
        assert result[0]['value'] == 1
    finally:
        await Database.close()
