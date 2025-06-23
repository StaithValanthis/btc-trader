# reset_database.py
import asyncio
import asyncpg
from structlog import get_logger
from app.core.config import Config

logger = get_logger(__name__)

async def reset_database():
    """Reset the database by dropping tables and removing policies."""
    try:
        conn = await asyncpg.connect(**Config.DB_CONFIG)
        logger.info("Connected to the database")

        # Remove retention/compression policies if they exist
        await conn.execute('''
            SELECT remove_retention_policy('market_data') 
            WHERE EXISTS(
                SELECT 1 
                FROM _timescaledb_catalog.policy 
                WHERE hypertable_name = 'market_data'
            );
            SELECT remove_compression_policy('market_data') 
            WHERE EXISTS(
                SELECT 1 
                FROM _timescaledb_catalog.policy 
                WHERE hypertable_name = 'market_data'
            );
        ''')
        logger.info("Removed retention and compression policies")

        # Drop tables
        await conn.execute('DROP TABLE IF EXISTS market_data;')
        await conn.execute('DROP TABLE IF EXISTS trades;')
        logger.info("Dropped tables")

        logger.info("Database reset complete")
    except Exception as e:
        logger.error("Failed to reset database", error=str(e))
    finally:
        await conn.close()
        logger.info("Database connection closed")

if __name__ == "__main__":
    asyncio.run(reset_database())