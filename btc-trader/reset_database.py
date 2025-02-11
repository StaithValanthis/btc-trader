import asyncio
import asyncpg
from structlog import get_logger
from app.core.config import Config

logger = get_logger(__name__)

async def reset_database():
    """Reset the database by dropping tables and removing policies."""
    conn = None
    try:
        # Connect to the database
        conn = await asyncpg.connect(**Config.DB_CONFIG)
        logger.info("Connected to the database")

        # Remove policies and drop market_data table
        await conn.execute('''
            SELECT remove_retention_policy('market_data');
            SELECT remove_compression_policy('market_data');
        ''')
        logger.info("Removed policies for market_data")

        await conn.execute('DROP TABLE IF EXISTS market_data;')
        logger.info("Dropped market_data table")

        # Remove policy and drop trades table
        await conn.execute('''
            SELECT remove_retention_policy('trades');
        ''')
        logger.info("Removed retention policy for trades")
        await conn.execute('DROP TABLE IF EXISTS trades;')
        logger.info("Dropped trades table")

        logger.info("Database reset complete")
    except Exception as e:
        logger.error("Failed to reset database", error=str(e))
    finally:
        if conn:
            await conn.close()
            logger.info("Database connection closed")

if __name__ == "__main__":
    asyncio.run(reset_database())
