import os
import pandas as pd
from pybit.unified_trading import HTTP


def fetch_bybit_data():
    """Fetch Bitcoin historical data from Bybit."""
    client = HTTP(testnet=True)  # Use testnet for testing
    response = client.get_kline(
        category="linear",
        symbol="BTCUSDT",
        interval="D",  # Daily data
        limit=1095  # Fetch 1095 days of data
    )
    data = pd.DataFrame(response['result']['list'], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
    return data


def save_data(data):
    """Save data to a CSV file."""
    os.makedirs('data', exist_ok=True)
    data.to_csv('data/btc_historical.csv', index=False)


if __name__ == "__main__":
    print("Fetching Bitcoin historical data from Bybit...")
    data = fetch_bybit_data()
    save_data(data)
    print("Data saved to 'data/btc_historical.csv'")
