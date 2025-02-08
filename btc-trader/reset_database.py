import asyncio
import asyncpg
from structlog import get_logger
from app.core.config import Config

logger = get_logger(__name__)

async def reset_database():
    """Reset the database by dropping tables and removing policies."""
    try:
        # Connect to the database
        conn = await asyncpg.connect(**Config.DB_CONFIG)
        logger.info("Connected to the database")

        # Remove retention and compression policies for market_data
        await conn.execute('''
            SELECT remove_retention_policy('market_data');
            SELECT remove_compression_policy('market_data');
        ''')
        logger.info("Removed retention and compression policies for market_data")

        # Drop the market_data table
        await conn.execute('DROP TABLE IF EXISTS market_data;')
        logger.info("Dropped market_data table")

        # Remove retention policy for trades
        await conn.execute('''
            SELECT remove_retention_policy('trades');
        ''')
        logger.info("Removed retention policy for trades")

        # Drop the trades table
        await conn.execute('DROP TABLE IF EXISTS trades;')
        logger.info("Dropped trades table")

        logger.info("Database reset complete")
    except Exception as e:
        logger.error("Failed to reset database", error=str(e))
    finally:
        # Close the connection
        await conn.close()
        logger.info("Database connection closed")

if __name__ == "__main__":
    asyncio.run(reset_database())