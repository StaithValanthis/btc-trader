import asyncio
import pandas as pd
import ta
from structlog import get_logger

from app.core.database import Database
from app.core.config import Config

logger = get_logger(__name__)

class TradeService:
    """
    Mean-reversion per-symbol using BB + RSI + dynamic position sizing based on % risk.
    """
    def __init__(self, symbol: str):
        self.symbol     = symbol
        cfg              = Config.TRADING_CONFIG
        self.bb_window  = cfg['BB_WINDOW']
        self.bb_dev     = cfg['BB_DEV']
        self.rsi_long   = cfg['RSI_LONG']
        self.rsi_short  = cfg['RSI_SHORT']
        self.sl_pct     = cfg['SL_PCT']
        self.risk_pct   = cfg['risk_pct']
        self.leverage   = cfg['leverage']
        self.session    = None
        self.min_qty    = None
        self.current_pos= None
        self.entry_price= None
        self.running    = False

    async def initialize(self):
        from pybit.unified_trading import HTTP
        # init REST session
        self.session = HTTP(
            testnet=Config.BYBIT_CONFIG['testnet'],
            api_key=Config.BYBIT_CONFIG['api_key'],
            api_secret=Config.BYBIT_CONFIG['api_secret'],
            recv_window=5000
        )
        # fetch min order qty
        info = await asyncio.to_thread(
            self.session.get_instruments_info,
            category=Config.BYBIT_CONFIG['category'],
            symbol=self.symbol
        )
        self.min_qty = float(info['result']['list'][0]['lotSizeFilter']['minOrderQty'])
        self.running = True
        asyncio.create_task(self.run_trading_logic())
        logger.info(f"TradeService initialized for {self.symbol}")

    async def run_trading_logic(self):
        while self.running:
            try:
                # fetch recent candles
                rows = await Database.fetch('''
                    SELECT time, open, high, low, close, volume
                    FROM candles
                    WHERE symbol=$1
                    ORDER BY time DESC
                    LIMIT 100
                ''', self.symbol)
                if len(rows) < self.bb_window:
                    await asyncio.sleep(60)
                    continue

                df = pd.DataFrame(rows, columns=["time","open","high","low","close","volume"])
                df['time'] = pd.to_datetime(df['time'])
                df.set_index('time', inplace=True)
                df.sort_index(inplace=True)

                # compute indicators
                bb = ta.volatility.BollingerBands(
                    close=df['close'], window=self.bb_window, window_dev=self.bb_dev
                )
                df['mid']   = bb.bollinger_mavg()
                df['upper'] = bb.bollinger_hband()
                df['lower'] = bb.bollinger_lband()

                rsi = ta.momentum.RSIIndicator(close=df['close'], window=self.rsi_long)
                df['rsi'] = rsi.rsi()

                cur = df.iloc[-1]
                price = float(cur['close'])
                mid   = float(cur['mid'])
                upper = float(cur['upper'])
                lower = float(cur['lower'])
                r     = float(cur['rsi'])

                # determine entry signal
                signal = None
                if price < lower and r < self.rsi_long:
                    signal = 'long'
                elif price > upper and r > self.rsi_short:
                    signal = 'short'

                # exit logic: mid-band or SL
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

            except Exception as e:
                logger.error("Trading logic error", error=str(e), symbol=self.symbol)

            # run every 10 minutes
            await asyncio.sleep(600)

    async def _enter(self, side: str, price: float):
        # compute dynamic position size by risk-per-trade
        # get equity in USDT
        bal = await asyncio.to_thread(self.session.get_wallet_balance, accountType="UNIFIED")
        acct = bal["result"]["list"][0]["coin"]
        # find USDT balance entry
        equity = 0.0
        for c in acct:
            if c["coin"] == "USDT":
                equity = float(c["usdValue"])
                break
        risk_amount = equity * self.risk_pct
        # calculate stop price
        if side == 'long':
            stop_price = price * (1 - self.sl_pct)
        else:
            stop_price = price * (1 + self.sl_pct)
        unit_risk = abs(price - stop_price)
        qty = risk_amount / unit_risk if unit_risk > 0 else self.min_qty
        qty = max(qty, self.min_qty)

        order = {
            'symbol':    self.symbol,
            'side':      'Buy' if side=='long' else 'Sell',
            'orderType': 'Market',
            'qty':       str(qty),
            'leverage':  str(self.leverage),
            'category':  Config.BYBIT_CONFIG['category']
        }
        resp = await asyncio.to_thread(self.session.place_order, **order)
        if resp.get('retCode') == 0:
            self.current_pos = side
            self.entry_price = price
            logger.info(f"Entered {side}", symbol=self.symbol, price=price, qty=qty)
        else:
            logger.error("Entry failed", response=resp, symbol=self.symbol)

    async def _exit(self, price: float):
        side = 'short' if self.current_pos=='long' else 'long'
        # use same qty as entry
        order = {
            'symbol':    self.symbol,
            'side':      'Buy' if side=='long' else 'Sell',
            'orderType': 'Market',
            'qty':       None,  # using default contract size
            'leverage':  str(self.leverage),
            'category':  Config.BYBIT_CONFIG['category']
        }
        # If you want to exit full position, you could omit qty (exchange closes full POS)
        resp = await asyncio.to_thread(self.session.place_order, **{k:v for k,v in order.items() if v})
        if resp.get('retCode') == 0:
            logger.info(f"Exited {self.current_pos}", symbol=self.symbol, price=price)
            self.current_pos = None
            self.entry_price = None
        else:
            logger.error("Exit failed", response=resp, symbol=self.symbol)

    def stop(self):
        self.running = False
        logger.info("TradeService stopped", symbol=self.symbol)
