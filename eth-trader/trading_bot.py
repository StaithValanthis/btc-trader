import asyncio
import aiohttp
import numpy as np
import logging
from collections import deque
from tenacity import retry, wait_exponential, stop_after_attempt
from config import Config
from database import init_db, batch_insert_market_data, log_trade
from indicators import compute_rsi, compute_atr, compute_bb, EMAUpdater

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

# Rolling windows for price data
ml_price_history = deque(maxlen=Config.PRICE_WINDOW)
full_price_history = deque(maxlen=1000)

# Buffer for batch DB insertion
market_data_buffer = []
BATCH_SIZE = 10

# EMA updater instance
ema_updater = EMAUpdater(period=Config.EMA_PERIOD)

@retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(5))
async def fetch_latest_price(session):
    async with session.get(Config.TICKER_ENDPOINT) as response:
        data = await response.json()
        price = float(data["result"][0]["last_price"])
        return price

async def get_prediction(features):
    url = "http://model_service:5000/predict"
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json={"features": features.tolist()}) as resp:
            result = await resp.json()
            return result.get("prediction", [None])[0]

async def trading_loop():
    await init_db()
    iteration = 0
    async with aiohttp.ClientSession() as session:
        while True:
            try:
                price = await fetch_latest_price(session)
            except Exception:
                await asyncio.sleep(10)
                continue
            if price is None:
                await asyncio.sleep(10)
                continue
            logging.info("Fetched price: %.2f", price)

            ml_price_history.append(price)
            full_price_history.append(price)
            market_data_buffer.append(("ETHUSD", price))
            if len(market_data_buffer) >= BATCH_SIZE:
                await batch_insert_market_data(market_data_buffer)
                market_data_buffer.clear()

            rsi_value = compute_rsi(ml_price_history)
            atr_value = compute_atr(full_price_history)
            bb_ma, bb_upper, bb_lower = compute_bb(full_price_history)
            ema_value = ema_updater.update(price)
            if rsi_value is not None:
                logging.info("RSI: %.2f", rsi_value)
            if atr_value is not None:
                logging.info("ATR: %.4f", atr_value)
            if bb_ma is not None:
                logging.info("BB: MA=%.2f, Upper=%.2f, Lower=%.2f", bb_ma, bb_upper, bb_lower)
            logging.info("EMA: %.2f", ema_value)

            iteration += 1
            ml_signal = None
            if len(ml_price_history) == Config.PRICE_WINDOW:
                features = np.array(list(ml_price_history)).reshape(1, Config.PRICE_WINDOW)
                try:
                    predicted_price = await get_prediction(features)
                    logging.info("Predicted price (ensemble): %.2f", predicted_price)
                    if predicted_price is not None:
                        if predicted_price - price > Config.PREDICTION_MARGIN:
                            ml_signal = "Buy"
                        elif price - predicted_price > Config.PREDICTION_MARGIN:
                            ml_signal = "Sell"
                except Exception as e:
                    logging.error("Error obtaining prediction: %s", e)

            indicator_signals = []
            if rsi_value is not None:
                if rsi_value < Config.RSI_OVERSOLD:
                    indicator_signals.append("Buy")
                elif rsi_value > Config.RSI_OVERBOUGHT:
                    indicator_signals.append("Sell")
            if bb_lower is not None and bb_upper is not None:
                if price < bb_lower:
                    indicator_signals.append("Buy")
                elif price > bb_upper:
                    indicator_signals.append("Sell")
            if ema_value is not None:
                if price > ema_value:
                    indicator_signals.append("Buy")
                elif price < ema_value:
                    indicator_signals.append("Sell")
            indicator_signal = None
            if indicator_signals:
                if indicator_signals.count("Buy") > indicator_signals.count("Sell"):
                    indicator_signal = "Buy"
                elif indicator_signals.count("Sell") > indicator_signals.count("Buy"):
                    indicator_signal = "Sell"

            fallback_signal = None
            if not ml_signal and not indicator_signal:
                if price < 1800:
                    fallback_signal = "Buy"
                elif price > 1900:
                    fallback_signal = "Sell"

            final_signal = ml_signal or indicator_signal or fallback_signal
            logging.info("Final signal: %s", final_signal if final_signal else "None")

            # Simulate automated order management (including stop-loss, take-profit stubs)
            if final_signal:
                await log_trade("ETHUSD", final_signal, price, Config.DEFAULT_QUANTITY)
                logging.info("Executed %s trade at %.2f (Simulated advanced order management)", final_signal, price)
            else:
                logging.info("No trade executed.")

            await asyncio.sleep(10)

if __name__ == "__main__":
    try:
        asyncio.run(trading_loop())
    except KeyboardInterrupt:
        logging.info("Trading bot stopped by user.")
