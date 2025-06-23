# app/services/trade_service.py

import asyncio
import pandas as pd
import ta
from pybit.unified_trading import HTTP

from app.core.database import Database
from app.core.config import Config
from app.core.logger import get_logger

logger = get_logger(__name__)

class TradeService:
    """
    Mean-reversion per‐symbol using BB + RSI + dynamic position sizing based on % risk.
    """

    def __init__(self, symbol: str, db_pool):
        self.symbol       = symbol
        self.db           = db_pool
        self.cfg          = Config.TRADING_CONFIG
        self.bb_window    = self.cfg['BB_WINDOW']
        self.bb_dev       = self.cfg['BB_DEV']
        self.rsi_long     = self.cfg['RSI_LONG']
        self.rsi_short    = self.cfg['RSI_SHORT']
        self.sl_pct       = self.cfg['SL_PCT']
        self.risk_pct     = self.cfg['risk_pct']
        self.leverage     = self.cfg['leverage']

        self.session      = None
        self.tick_size    = None
        self.lot_size     = None

        self.current_pos  = None   # 'long' or 'short'
        self.entry_price  = None

        # in-memory buffer of the last N candles
        self.candles      = pd.DataFrame()

    async def initialize(self):
        # 1) REST session
        self.session = HTTP(
            testnet   = Config.BYBIT_CONFIG['testnet'],
            api_key   = Config.BYBIT_CONFIG['api_key'],
            api_secret= Config.BYBIT_CONFIG['api_secret'],
            recv_window=5000,
        )

        # 2) Instrument info → tick & lot size
        try:
            info = await asyncio.to_thread(
                self.session.get_instruments_info,
                category=Config.BYBIT_CONFIG['category'],
                symbol=self.symbol
            )
            inst = info['result']['list'][0]
            pf   = inst['priceFilter']
            lf   = inst['lotSizeFilter']
            self.tick_size = float(pf['tickSize'])
            self.lot_size  = float(lf['qtyStep'])

        except Exception as e:
            if "symbol invalid" in str(e).lower():
                logger.warning("Skipping invalid symbol %s: %s", self.symbol, e,
                               extra={"event":"Skipping invalid symbol"})
                return
            raise

        # 3) Set leverage
        await asyncio.to_thread(
            self.session.set_leverage,
            category   = Config.BYBIT_CONFIG['category'],
            symbol     = self.symbol,
            buy_leverage = self.leverage,
            sell_leverage= self.leverage,
        )

        # 4) Load any existing position from DB
        row = await Database.fetchrow(
            "SELECT side, size FROM positions WHERE symbol=$1",
            self.symbol
        )
        if row:
            self.current_pos = row['side']
            # you could also load an entry_price if you store that

        logger.info(
            "TradeService initialized for %s (tick=%s lot=%s lev=%s)",
            self.symbol, self.tick_size, self.lot_size, self.leverage,
            extra={"event":"initialized"}
        )

    async def on_candle(self, candle: dict):
        # append the new candle
        df = pd.DataFrame([{
            'time':   pd.to_datetime(candle['startTime'], unit='ms'),
            'open':   float(candle['open']),
            'high':   float(candle['high']),
            'low':    float(candle['low']),
            'close':  float(candle['close']),
            'volume': float(candle['volume']),
        }]).set_index('time')

        self.candles = pd.concat([self.candles, df]).iloc[-self.bb_window*3:]

        if len(self.candles) < self.bb_window:
            return

        # BB & RSI
        bb   = ta.volatility.BollingerBands(self.candles['close'],
                                            window=self.bb_window,
                                            window_dev=self.bb_dev)
        self.candles['mid']   = bb.bollinger_mavg()
        self.candles['upper'] = bb.bollinger_hband()
        self.candles['lower'] = bb.bollinger_lband()

        rsi  = ta.momentum.RSIIndicator(self.candles['close'],
                                        window=self.rsi_long)
        self.candles['rsi'] = rsi.rsi()

        cur   = self.candles.iloc[-1]
        price = float(cur['close'])
        mid   = float(cur['mid'])
        upper = float(cur['upper'])
        lower = float(cur['lower'])
        r     = float(cur['rsi'])

        # signal
        signal = None
        if price < lower and r < self.rsi_long:
            signal = 'long'
        elif price > upper and r > self.rsi_short:
            signal = 'short'

        # exit logic
        if self.current_pos == 'long':
            stop_price = self.entry_price * (1 - self.sl_pct)
            if price >= mid or price <= stop_price:
                await self._exit(price)

        elif self.current_pos == 'short':
            stop_price = self.entry_price * (1 + self.sl_pct)
            if price <= mid or price >= stop_price:
                await self._exit(price)

        # entry logic
        elif not self.current_pos and signal:
            await self._enter(signal, price)

    async def _enter(self, side: str, price: float):
        # 1) compute risk‐based qty
        bal = await asyncio.to_thread(
            self.session.get_wallet_balance, accountType="UNIFIED"
        )
        acct = bal["result"]["list"][0]["coin"]
        equity = next(
            (float(c["usdValue"]) for c in acct if c["coin"]=="USDT"),
            0.0
        )
        risk_amount = equity * self.risk_pct

        if side=='long':
            stop_price = price * (1 - self.sl_pct)
        else:
            stop_price = price * (1 + self.sl_pct)

        unit_risk = abs(price - stop_price) or self.lot_size
        qty = max(risk_amount / unit_risk, self.lot_size)
        # round down to lot_size multiples
        qty = (int(qty // self.lot_size)) * self.lot_size

        resp = await asyncio.to_thread(
            self.session.place_active_order,
            category = Config.BYBIT_CONFIG['category'],
            symbol   = self.symbol,
            side     = 'Buy' if side=='long' else 'Sell',
            orderType= 'Market',
            qty      = str(qty),
            leverage = str(self.leverage),
        )

        if resp.get('retCode') == 0:
            self.current_pos = side
            self.entry_price = price
            logger.info("Entered %s @ %s (qty=%s)", side, price, qty,
                        extra={"event":"enter","symbol":self.symbol})
        else:
            logger.error("Entry failed %s", resp,
                         extra={"event":"enter_error","symbol":self.symbol})

    async def _exit(self, price: float):
        # opposite side
        side = 'Sell' if self.current_pos=='long' else 'Buy'

        resp = await asyncio.to_thread(
            self.session.place_active_order,
            category = Config.BYBIT_CONFIG['category'],
            symbol   = self.symbol,
            side     = side,
            orderType= 'Market',
            # omit qty to close full
            leverage = str(self.leverage),
        )

        if resp.get('retCode') == 0:
            logger.info("Exited %s @ %s", self.current_pos, price,
                        extra={"event":"exit","symbol":self.symbol})
            self.current_pos = None
            self.entry_price = None
        else:
            logger.error("Exit failed %s", resp,
                         extra={"event":"exit_error","symbol":self.symbol})

    async def on_trade(self, trade: dict):
        # record fills & update your positions table as needed
        # e.g.
        # await Database.execute(
        #    "INSERT INTO fills (...) VALUES (...)",
        #    ...
        # )
        pass

    async def run(self, ws):
        # subscribe & dispatch
        await ws.subscribe([
            f"candle.1.{self.symbol}",
            f"trade.100ms.{self.symbol}",
        ])
        async for msg in ws:
            topic = msg.get("topic","")
            data  = msg.get("data",{})
            if topic.startswith("candle."):
                await self.on_candle(data)
            elif topic.startswith("trade."):
                await self.on_trade(data)
            else:
                logger.debug("Ignored message", topic=topic)

    def stop(self):
        logger.info("Stopping", symbol=self.symbol)
