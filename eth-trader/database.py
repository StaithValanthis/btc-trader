import logging
from tortoise import Tortoise, fields
from tortoise.models import Model
from config import Config

class Trade(Model):
    id = fields.IntField(pk=True)
    symbol = fields.CharField(max_length=10)
    trade_time = fields.DatetimeField(auto_now_add=True)
    side = fields.CharField(max_length=4)
    price = fields.FloatField()
    quantity = fields.FloatField()

    class Meta:
        table = "trades"

class MarketData(Model):
    id = fields.IntField(pk=True)
    symbol = fields.CharField(max_length=10)
    timestamp = fields.DatetimeField(auto_now_add=True)
    price = fields.FloatField()

    class Meta:
        table = "market_data"

async def init_db():
    await Tortoise.init(config=Config.TORTOISE_ORM)
    await Tortoise.generate_schemas()
    logging.info("Database initialized and schemas generated.")

async def log_trade(symbol, side, price, quantity):
    await Trade.create(symbol=symbol, side=side, price=price, quantity=quantity)
    logging.info("Logged trade: %s %s @ %.2f, quantity %.4f", side, symbol, price, quantity)

async def batch_insert_market_data(data):
    objs = [MarketData(symbol=s, price=p) for s, p in data]
    await MarketData.bulk_create(objs)
    logging.info("Batch inserted %d market data records.", len(data))
