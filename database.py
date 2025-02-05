import psycopg2
from psycopg2.extras import execute_batch
from datetime import datetime
import json
import os

class TimescaleDB:
    def __init__(self):
        """
        Initialize a connection to the TimescaleDB database using environment variables.
        """
        try:
            self.conn = psycopg2.connect(
                host=os.getenv("DB_HOST", "localhost"),
                database=os.getenv("DB_NAME", "trading_bot"),
                user=os.getenv("DB_USER", "postgres"),
                password=os.getenv("DB_PASSWORD", "your_password"),
                port=int(os.getenv("DB_PORT", "5432"))  # Convert to integer
            )
            self.cursor = self.conn.cursor()
            print("Successfully connected to TimescaleDB!")
        except Exception as e:
            print(f"Failed to connect to TimescaleDB: {e}")
            raise

    def log_market_data(self, time: datetime, price: float, features: dict):
        """
        Log market data to the `market_data` table.
        """
        try:
            query = """
            INSERT INTO market_data (time, price, features)
            VALUES (%s, %s, %s)
            """
            self.cursor.execute(query, (time, price, json.dumps(features)))
            self.conn.commit()
        except Exception as e:
            print(f"Failed to log market data: {e}")
            self.conn.rollback()

    def log_trade(self, time: datetime, price: float, signal: str, profit_loss: float = None):
        """
        Log a trade to the `trades` table.
        """
        try:
            query = """
            INSERT INTO trades (time, price, signal, profit_loss)
            VALUES (%s, %s, %s, %s)
            """
            self.cursor.execute(query, (time, price, signal, profit_loss))
            self.conn.commit()
        except Exception as e:
            print(f"Failed to log trade: {e}")
            self.conn.rollback()

    def get_historical_data(self, lookback_days: int = 30):
        """
        Retrieve historical market data.
        """
        try:
            query = """
            SELECT time, price, features
            FROM market_data
            WHERE time > NOW() - INTERVAL '%s days'
            ORDER BY time DESC
            """
            self.cursor.execute(query, (lookback_days,))
            return self.cursor.fetchall()
        except Exception as e:
            print(f"Failed to fetch historical data: {e}")
            return []

    def close(self):
        """
        Close the database connection.
        """
        try:
            self.cursor.close()
            self.conn.close()
            print("Database connection closed.")
        except Exception as e:
            print(f"Failed to close database connection: {e}")

# Example usage
if __name__ == "__main__":
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()

    # Initialize database connection
    db = TimescaleDB()

    # Log sample market data
    db.log_market_data(
        time=datetime.now(),
        price=30000.5,
        features={"moving_average": 30000.0, "volatility": 100.0}
    )

    # Log sample trade
    db.log_trade(
        time=datetime.now(),
        price=30000.5,
        signal="BUY",
        profit_loss=50.0
    )

    # Fetch historical data
    historical_data = db.get_historical_data(lookback_days=7)
    print("Historical Data:")
    for row in historical_data:
        print(row)

    # Close connection
    db.close()