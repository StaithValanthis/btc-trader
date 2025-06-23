FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-dev \
    python3-pip \
    libpq-dev \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip3 install --upgrade pip
RUN pip3 install --no-cache-dir -r requirements.txt
COPY . .

ENV PYTHONPATH=/app

RUN useradd -m trader && chown -R trader:trader /app
USER trader

CMD ["python", "main.py"]
