FROM python:3.9-slim

WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . .

EXPOSE 8000

ENV PYTHONUNBUFFERED=1

# By default, start the trading bot. (The model service is started separately via docker-compose.)
CMD ["python", "trading_bot.py"]
