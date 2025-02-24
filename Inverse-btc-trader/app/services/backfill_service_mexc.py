# app/services/backfill_service.py

import requests
import time
import logging
from datetime import datetime
import sqlite3  # Replace with your actual database connection

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MEXC API endpoint
API_URL = "https://contract.mexc.com/api/v1/contract/kline"

def fetch_candlestick_data(symbol, period, start_time, size=200):
    """
    Fetch historical candlestick data from MEXC.

    :param symbol: Trading pair symbol, e.g., 'BTC_USD'
    :param period: K-line interval, e.g., '1m'
    :param start_time: Start time in Unix timestamp (seconds)
    :param size: Number of data points to retrieve (max 2000)
    :return: List of candlestick data
    """
    params = {
        'symbol': symbol,
        'interval': period,
        'start': start_time,
        'limit': size
    }
    try:
        response = requests.get(API_URL, params=params)
        response.raise_for_status()  # Raise an exception for HTTP errors
        data = response.json()
        if data.get('success'):
            return data.get('data', [])
        else:
            logger.error(f"API Error: {data.get('msg')}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Request Exception: {e}")
    return []

def save_to_database(candles):
    """
    Save candlestick data to the database.

    :param candles: List of candlestick data
    """
    # Establish a database connection (replace with your actual database connection)
    conn = sqlite3.connect('candles.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS candles (
            time INTEGER PRIMARY KEY,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL
        )
    ''')
    conn.commit()

    for candle in candles:
        timestamp = candle[0]
        open_price = candle[1]
        high_price = candle[2]
        low_price = candle[3]
        close_price = candle[4]
        volume = candle[5]
        cursor.execute('''
            INSERT OR IGNORE INTO candles (time, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (timestamp, open_price, high_price, low_price, close_price, volume))
    conn.commit()
    conn.close()

def backfill_mexc_kline(symbol='BTC_USD', period='Min1', days_to_fetch=180):
    """
    Backfill historical candlestick data from MEXC.

    :param symbol: Trading pair symbol, e.g., 'BTC_USD'
    :param period: K-line interval, e.g., 'Min1' for 1 minute
    :param days_to_fetch: Number of days to backfill
    """
    current_time = int(time.time())
    start_time = current_time - days_to_fetch * 24 * 60 * 60  # days_to_fetch days ago

    logger.info(f"Fetching data for {symbol} starting from {datetime.utcfromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')} UTC")

    while start_time < current_time:
        candles = fetch_candlestick_data(symbol, period, start_time)
        if not candles:
            logger.info("No more data returned from MEXC.")
            break
        save_to_database(candles)
        start_time = candles[-1][0] + 60  # Increment start_time by 60 seconds (next minute)
        time.sleep(1)  # To respect API rate limits

    logger.info("Data fetching complete.")
