import time
import hmac
import hashlib
import requests
import json

class MexcClient:
    """
    A basic client for interacting with MEXC's API.
    Adjust endpoints, parameter names, and signing as per MEXC’s current API documentation.
    """
    def __init__(self, api_key, api_secret, testnet=False):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = "https://contract.mexc.com" if not testnet else "https://testnet.mexc.com"
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "X-MEXC-APIKEY": self.api_key
        })
    
    def _sign(self, params):
        # Sign parameters sorted alphabetically (example; adjust as needed)
        sorted_params = "&".join(f"{k}={params[k]}" for k in sorted(params))
        signature = hmac.new(self.api_secret.encode(), sorted_params.encode(), hashlib.sha256).hexdigest()
        return signature

    def _get(self, endpoint, params=None):
        if params is None:
            params = {}
        params["timestamp"] = int(time.time() * 1000)
        params["signature"] = self._sign(params)
        url = f"{self.base_url}{endpoint}"
        response = self.session.get(url, params=params)
        return response.json()

    def _post(self, endpoint, data=None):
        if data is None:
            data = {}
        data["timestamp"] = int(time.time() * 1000)
        data["signature"] = self._sign(data)
        url = f"{self.base_url}{endpoint}"
        response = self.session.post(url, json=data)
        return response.json()

    def get_instruments_info(self, symbol):
        endpoint = "/api/v1/contract/instruments"
        params = {"symbol": symbol}
        return self._get(endpoint, params)
    
    def get_positions(self, symbol):
        endpoint = "/api/v1/contract/position"
        params = {"symbol": symbol}
        return self._get(endpoint, params)
    
    def place_order(self, symbol, side, qty, orderType="Market", stopLoss=None, takeProfit=None, leverage=None):
        endpoint = "/api/v1/contract/order"
        data = {"symbol": symbol, "side": side, "orderType": orderType, "qty": qty}
        if stopLoss is not None:
            data["stopLoss"] = stopLoss
        if takeProfit is not None:
            data["takeProfit"] = takeProfit
        if leverage is not None:
            data["leverage"] = leverage
        return self._post(endpoint, data)
    
    def set_trading_stop(self, symbol, stopLoss, takeProfit, positionIdx=0):
        endpoint = "/api/v1/contract/order/stop"
        data = {"symbol": symbol, "stopLoss": stopLoss, "takeProfit": takeProfit, "positionIdx": positionIdx}
        return self._post(endpoint, data)
    
    def get_wallet_balance(self, accountType="UNIFIED"):
        endpoint = "/api/v1/contract/wallet"
        params = {"accountType": accountType}
        return self._get(endpoint, params)
