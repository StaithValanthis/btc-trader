import pytest
import asyncio
from app.core.database import Database
from datetime import datetime, timezone

@pytest.mark.asyncio
async def test_database_connection():
    await Database.initialize()
    try:
        result = await Database.execute("SELECT 1")
        assert result == "SELECT 1"
    finally:
        await Database.close()