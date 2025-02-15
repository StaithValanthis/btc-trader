import asyncio
import pandas as pd
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from joblib import dump
import logging
import asyncpg
from config import Config

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

DATABASE_URL = f"postgres://{Config.POSTGRES_USER}:{Config.POSTGRES_PASSWORD}@{Config.POSTGRES_HOST}:{Config.POSTGRES_PORT}/{Config.POSTGRES_DB}"

async def get_historical_data():
    try:
        conn = await asyncpg.connect(DATABASE_URL)
        rows = await conn.fetch("SELECT price FROM market_data ORDER BY timestamp ASC;")
        await conn.close()
        if not rows:
            logging.warning("No market data found.")
            return None
        df = pd.DataFrame([dict(r) for r in rows])
        return df
    except Exception as e:
        logging.error("Error fetching historical data: %s", e)
        return None

def generate_synthetic_data(num_points=200, start_price=1800):
    np.random.seed(42)
    prices = start_price + np.cumsum(np.random.randn(num_points))
    return pd.DataFrame({"price": prices})

def create_features(df, window_size=Config.PRICE_WINDOW):
    X, y = [], []
    prices = df["price"].values
    for i in range(len(prices) - window_size):
        X.append(prices[i:i+window_size])
        y.append(prices[i+window_size])
    X = np.array(X)
    y = np.array(y)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.reshape(-1, 1)).reshape(X.shape)
    return X_scaled, y, scaler

async def main():
    df = await get_historical_data()
    if df is None or df.empty:
        logging.info("Using synthetic data for training.")
        df = generate_synthetic_data()
    X, y, scaler = create_features(df)
    logging.info("Feature shape: %s, Target shape: %s", X.shape, y.shape)
    sgd = SGDRegressor(max_iter=1000, tol=1e-3)
    sgd.partial_fit(X, y)
    rf = RandomForestRegressor(n_estimators=100)
    rf.fit(X, y)
    dump(sgd, "sgd_model.joblib")
    dump(rf, "rf_model.joblib")
    dump(scaler, SCALER_FILENAME)
    logging.info("Ensemble models trained and saved.")

if __name__ == "__main__":
    asyncio.run(main())
