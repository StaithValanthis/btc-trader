import psycopg2
from psycopg2.extras import execute_batch
from datetime import datetime
import json

class TimescaleDB:
    def __init__(self):
        self.conn = psycopg2.connect(
            host="localhost",
            database="trading_bot",
            user="postgres",
            password="your_password"
        )
        self.cursor = self.conn.cursor()

    def log_market_data(self, time: datetime, price: float, features: dict):
        query = """
        INSERT INTO market_data (time, price, features)
        VALUES (%s, %s, %s)
        """
        self.cursor.execute(query, (time, price, json.dumps(features)))
        self.conn.commit()

    def log_trade(self, time: datetime, price: float, signal: str, profit_loss: float = None):
        query = """
        INSERT INTO trades (time, price, signal, profit_loss)
        VALUES (%s, %s, %s, %s)
        """
        self.cursor.execute(query, (time, price, signal, profit_loss))
        self.conn.commit()

    def get_historical_data(self, lookback_days: int = 30):
        query = """
        SELECT time, price, features
        FROM market_data
        WHERE time > NOW() - INTERVAL '%s days'
        ORDER BY time DESC
        """
        self.cursor.execute(query, (lookback_days,))
        return self.cursor.fetchall()

    def close(self):
        self.cursor.close()
        self.conn.close()