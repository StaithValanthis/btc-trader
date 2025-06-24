# File: app/services/trade_service.py

import asyncio
import pandas as pd
import ta
from pybit.unified_trading import HTTP

from app.core.config import Config
from app.utils.logger import logger

class TradeService:
    """
    Mean-reversion per-symbol using BB + RSI + dynamic position sizing based on % risk.
    Operates in event-driven mode: reacts to each new 1-min candle immediately.
    """

    def __init__(self, symbol: str):
        self.symbol       = symbol
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
        self.position_idx = 0

        self.current_pos  = None   # 'long' or 'short'
        self.entry_price  = None

        # in-memory buffer of recent candles
        self.candles      = pd.DataFrame()

    async def initialize(self) -> bool:
        # 1) REST session
        try:
            self.session = HTTP(
                testnet     = Config.BYBIT_CONFIG['testnet'],
                api_key     = Config.BYBIT_CONFIG['api_key'],
                api_secret  = Config.BYBIT_CONFIG['api_secret'],
                recv_window = 5000,
            )
        except Exception as e:
            logger.error("Failed to initialize Bybit HTTP session", symbol=self.symbol, error=str(e))
            return False

        # 2) Instrument info → tick & lot size
        try:
            info = await asyncio.to_thread(
                self.session.get_instruments_info,
                category=Config.BYBIT_CONFIG['category'],
                symbol=self.symbol
            )
            inst_list = info.get('result', {}).get('list', [])
            if not inst_list:
                logger.warning("Skipping: symbol not found in Bybit instruments", symbol=self.symbol)
                return False
            inst = inst_list[0]
            if inst.get('status') != 'Trading':
                logger.warning("Skipping: symbol not trading", symbol=self.symbol)
                return False
            pf = inst['priceFilter']
            lf = inst['lotSizeFilter']
            self.tick_size = float(pf['tickSize'])
            self.lot_size  = float(lf['qtyStep'])
        except Exception as e:
            logger.error("Instrument info fetch failed", symbol=self.symbol, error=str(e))
            return False

        # 3) Determine position mode if possible
        self.position_idx = 0  # Default OneWay
        try:
            acct_info = await asyncio.to_thread(
                self.session.get_account_info,
                accountType="UNIFIED"
            )
            acct_list = acct_info.get('result', {}).get('list')
            if acct_list and isinstance(acct_list, list) and len(acct_list) > 0:
                mode = acct_list[0].get("positionMode", "OneWay")
                self.position_idx = 0 if mode == "OneWay" else 1
                logger.info("Account position mode", symbol=self.symbol, position_mode=mode, position_idx=self.position_idx)
            else:
                logger.warning("No position mode info, using default OneWay", symbol=self.symbol)
        except Exception as e:
            logger.warning("Could not determine position mode, defaulting to OneWay", symbol=self.symbol, error=str(e))

        # 4) Try to set leverage; only send position_idx for non-linear categories
        try:
            params = dict(
                category      = Config.BYBIT_CONFIG['category'],
                symbol        = self.symbol,
                buy_leverage  = self.leverage,
                sell_leverage = self.leverage
            )
            # Do NOT send position_idx for linear contracts on mainnet
            if Config.BYBIT_CONFIG['category'] != 'linear':
                params['position_idx'] = self.position_idx
            await asyncio.to_thread(
                self.session.set_leverage,
                **params
            )
        except Exception as e:
            logger.warning(
                "Skipping symbol; leverage not supported",
                symbol=self.symbol, error=str(e)
            )
            return False

        logger.info(
            "TradeService initialized",
            symbol=self.symbol,
            tick_size=self.tick_size,
            lot_size=self.lot_size,
            leverage=self.leverage,
            position_idx=self.position_idx
        )
        return True

    async def on_candle(self, candle: dict):
        try:
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

            # compute BB & RSI
            bb = ta.volatility.BollingerBands(
                close=self.candles['close'],
                window=self.bb_window,
                window_dev=self.bb_dev
            )
            self.candles['mid']   = bb.bollinger_mavg()
            self.candles['upper'] = bb.bollinger_hband()
            self.candles['lower'] = bb.bollinger_lband()

            rsi = ta.momentum.RSIIndicator(
                close=self.candles['close'],
                window=self.rsi_long
            )
            self.candles['rsi'] = rsi.rsi()

            cur   = self.candles.iloc[-1]
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

            # exit on mid‐band or stop‐loss
            if self.current_pos == 'long':
                stop_price = self.entry_price * (1 - self.sl_pct)
                if price >= mid or price <= stop_price:
                    await self._exit(price)

            elif self.current_pos == 'short':
                stop_price = self.entry_price * (1 + self.sl_pct)
                if price <= mid or price >= stop_price:
                    await self._exit(price)

            # entry
            elif not self.current_pos and signal:
                await self._enter(signal, price)
        except Exception as e:
            logger.exception("TradeService on_candle error", symbol=self.symbol, error=str(e))

    async def _enter(self, side: str, price: float):
        try:
            bal = await asyncio.to_thread(
                self.session.get_wallet_balance,
                accountType="UNIFIED"
            )
            acct = bal.get("result", {}).get("list")
            equity = 0.0
            if acct and isinstance(acct, list) and len(acct) > 0:
                coins = acct[0].get("coin", [])
                equity = next(
                    (float(c["usdValue"]) for c in coins if c["coin"] == "USDT"),
                    0.0
                )
            if equity <= 0:
                logger.warning("No USDT equity available", symbol=self.symbol)
                return
            risk_amount = equity * self.risk_pct

            stop_price = price * (1 - self.sl_pct) if side == 'long' else price * (1 + self.sl_pct)
            unit_risk  = abs(price - stop_price) or self.lot_size
            qty = max(risk_amount / unit_risk, self.lot_size)
            qty = (int(qty // self.lot_size)) * self.lot_size  # round down

            resp = await asyncio.to_thread(
                self.session.place_active_order,
                category  = Config.BYBIT_CONFIG['category'],
                symbol    = self.symbol,
                side      = 'Buy' if side == 'long' else 'Sell',
                orderType = 'Market',
                qty       = str(qty),
                leverage  = str(self.leverage),
            )

            if resp.get('retCode') == 0:
                self.current_pos = side
                self.entry_price = price
                logger.info(
                    "Entered",
                    symbol=self.symbol, side=side, price=price, qty=qty, event="enter"
                )
            else:
                logger.error(
                    "Entry failed",
                    symbol=self.symbol, response=resp, event="enter_error"
                )
        except Exception as e:
            logger.exception("TradeService _enter error", symbol=self.symbol, error=str(e))

    async def _exit(self, price: float):
        try:
            side = 'Sell' if self.current_pos == 'long' else 'Buy'
            resp = await asyncio.to_thread(
                self.session.place_active_order,
                category  = Config.BYBIT_CONFIG['category'],
                symbol    = self.symbol,
                side      = side,
                orderType = 'Market',
                leverage  = str(self.leverage),
            )

            if resp.get('retCode') == 0:
                logger.info(
                    "Exited",
                    symbol=self.symbol, side=self.current_pos, price=price, event="exit"
                )
                self.current_pos = None
                self.entry_price = None
            else:
                logger.error(
                    "Exit failed",
                    symbol=self.symbol, response=resp, event="exit_error"
                )
        except Exception as e:
            logger.exception("TradeService _exit error", symbol=self.symbol, error=str(e))

    async def on_trade(self, trade: dict):
        # Optional: record fills into your DB and update positions table
        pass

    async def run(self, ws):
        # Subscribe to candle/trade topics for this symbol
        await ws.subscribe([
            f"candle.1.{self.symbol}",
            f"trade.100ms.{self.symbol}",
        ])
        ws.add_listener(self._on_ws_msg)

    async def _on_ws_msg(self, msg):
        topic = msg.get("topic", "")
        data  = msg.get("data", {})
        if not topic.endswith(self.symbol):
            return
        try:
            if topic.startswith("candle."):
                await self.on_candle(data)
            elif topic.startswith("trade."):
                await self.on_trade(data)
            else:
                logger.debug("Ignored message", topic=topic)
        except Exception as e:
            logger.exception("TradeService _on_ws_msg error", symbol=self.symbol, error=str(e))

    def stop(self):
        logger.info("Stopping TradeService", symbol=self.symbol)
