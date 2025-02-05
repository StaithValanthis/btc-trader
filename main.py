# main.py
import os
import logging
import time
import signal
from dotenv import load_dotenv
from pybit.unified_trading import WebSocket
from psycopg2.pool import SimpleConnectionPool

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/trading.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# Database connection pool
DB_POOL = None

def init_db_pool():
    global DB_POOL
    try:
        DB_POOL = SimpleConnectionPool(
            minconn=1,
            maxconn=10,
            dbname=os.getenv("TS_DB_NAME"),
            user=os.getenv("TS_DB_USER"),
            password=os.getenv("TS_DB_PASSWORD"),
            host=os.getenv("TS_DB_HOST"),
            port=os.getenv("TS_DB_PORT"),
        )
        logger.info("Database connection pool initialized")
    except Exception as e:
        logger.error(f"Failed to initialize DB pool: {e}")
        raise

def handle_orderbook(message: dict):
    """Process order book WebSocket messages."""
    try:
        # Skip non-data messages (heartbeats, etc.)
        if "data" not in message:
            return

        data = message["data"]
        symbol = data["s"]  # e.g., BTCUSDT
        bid_price = float(data["b"][0][0])  # Best bid price
        ask_price = float(data["a"][0][0])  # Best ask price

        # Get DB connection from pool
        conn = DB_POOL.getconn()
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO order_book (symbol, bid_price, ask_price, ts)
            VALUES (%s, %s, %s, NOW())
            """,
            (symbol, bid_price, ask_price),
        )
        conn.commit()
        logger.debug(f"Inserted {symbol} bid={bid_price}, ask={ask_price}")

    except KeyError as e:
        logger.error(f"Missing key in message: {e} - {message}")
    except Exception as e:
        logger.error(f"Error processing message: {e}", exc_info=True)
    finally:
        # Return connection to pool
        if "conn" in locals():
            cursor.close()
            DB_POOL.putconn(conn)

def manage_websocket():
    """WebSocket connection manager with auto-reconnect."""
    while True:
        ws = None
        try:
            ws = WebSocket(
                testnet=False,
                channel_type="linear",
                api_key=os.getenv("BYBIT_API_KEY"),
                api_secret=os.getenv("BYBIT_API_SECRET"),
                log_level="ERROR",  # Reduce pybit internal logging
            )

            # Subscribe to BTCUSDT order book
            ws.orderbook_stream(depth=25, symbol="BTCUSDT", callback=handle_orderbook)
            logger.info("WebSocket connected and subscribed")

            # Keep connection alive
            while True:
                time.sleep(1)

        except KeyboardInterrupt:
            logger.info("Shutting down gracefully...")
            if ws:
                ws.exit()
            break
        except Exception as e:
            logger.error(f"WebSocket error: {e}. Reconnecting in 5s...")
            time.sleep(5)
        finally:
            if ws:
                ws.exit()

def cleanup(signum, frame):
    """Handle shutdown signals."""
    logger.info("Cleaning up resources...")
    if DB_POOL:
        DB_POOL.closeall()
    exit(0)

if __name__ == "__main__":
    # Create logs directory
    os.makedirs("logs", exist_ok=True)

    # Initialize DB pool
    init_db_pool()

    # Register signal handlers
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    # Start WebSocket manager
    manage_websocket()