import time
import hmac
import hashlib
import requests

class MEXCClient:
    def __init__(self, api_key, api_secret, base_url="https://contract.mexc.com"):
        self.api_key = api_key
        self.api_secret = api_secret.encode('utf-8')
        self.base_url = base_url.rstrip("/")

    def _sign(self, params):
        sorted_params = "&".join(f"{k}={params[k]}" for k in sorted(params))
        signature = hmac.new(self.api_secret, sorted_params.encode('utf-8'), hashlib.sha256).hexdigest()
        return signature

    def get_instruments_info(self, symbol):
        endpoint = "/api/v1/contract/detail"
        params = {"symbol": symbol}  # Do not include contractType here
        url = self.base_url + endpoint
        response = requests.get(url, params=params)
        return response.json()

    def get_positions(self, symbol):
        endpoint = "/api/v1/private/position/list"
        timestamp = int(time.time() * 1000)
        params = {"symbol": symbol, "timestamp": timestamp}
        params["signature"] = self._sign(params)
        headers = {"X-MEXC-APIKEY": self.api_key}
        url = self.base_url + endpoint
        response = requests.get(url, params=params, headers=headers)
        return response.json()

    def place_order(self, **order_params):
        endpoint = "/api/v1/private/order/create"
        timestamp = int(time.time() * 1000)
        order_params["timestamp"] = timestamp
        order_params["signature"] = self._sign(order_params)
        headers = {"X-MEXC-APIKEY": self.api_key}
        url = self.base_url + endpoint
        response = requests.post(url, data=order_params, headers=headers)
        return response.json()

    def get_wallet_balance(self, accountType="UNIFIED"):
        endpoint = "/api/v1/private/account/assets"
        timestamp = int(time.time() * 1000)
        params = {"accountType": accountType, "timestamp": timestamp}
        params["signature"] = self._sign(params)
        headers = {"X-MEXC-APIKEY": self.api_key}
        url = self.base_url + endpoint
        response = requests.get(url, params=params, headers=headers)
        return response.json()

    def set_trading_stop(self, **params):
        # Dummy implementation; update if trailing stops are supported.
        return {"code": 0, "message": "Trailing stop set", "result": {}}

    def set_leverage(self, symbol, leverage):
        # Dummy implementation; update if leverage setting is supported.
        return {"code": 0, "message": "Leverage set", "result": {}}

    def get_historical_candles(self, symbol, interval, start_time_ms, size):
        """
        Fetch historical kline data from MEXC for coin-m futures.
        
        Endpoint: /api/v1/contract/kline
        Parameters:
          - symbol: e.g., "BTCUSD"
          - period: candle interval as a string (e.g., "1m")
          - size: maximum number of candles to return (max 200)
          - from: starting timestamp in Unix seconds
        """
        endpoint = "/api/v1/contract/kline"
        period_str = f"{interval}m"
        # Convert start_time from milliseconds to seconds.
        start_time_sec = int(start_time_ms / 1000)
        params = {
            "symbol": symbol,
            "period": period_str,
            "size": size,
            "from": start_time_sec
        }
        url = self.base_url + endpoint
        headers = {
            "Connection": "close",
            "User-Agent": "MEXCClient/1.0"
        }
        try:
            response = requests.get(url, params=params, headers=headers, timeout=30)
            return response.json()
        except Exception as e:
            return {"code": -1, "message": str(e), "data": []}
