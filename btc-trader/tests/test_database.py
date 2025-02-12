# tests/test_database.py
import pytest
import asyncio
from app.core.database import Database
from datetime import datetime, timezone

@pytest.mark.asyncio
async def test_database_connection():
    await Database.initialize()
    try:
        result = await Database.fetchval("SELECT 1")
        assert result == 1
    finally:
        await Database.close()