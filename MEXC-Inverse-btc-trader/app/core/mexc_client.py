# File: app/core/mexc_client.py

import time
import hmac
import hashlib
import requests
import json
import logging
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from requests.exceptions import RequestException, Timeout, ConnectionError

logger = logging.getLogger(__name__)

class MexcClient:
    """
    A basic client for interacting with MEXC's API.
    Adjust endpoints, parameter names, and signing as per MEXC’s current API documentation.
    """

    def __init__(self, api_key, api_secret, testnet=False):
        self.api_key = api_key
        self.api_secret = api_secret

        # Use the appropriate base URL for live or testnet trading.
        self.base_url = "https://contract.mexc.com" if not testnet else "https://testnet.mexc.com"

        # Create a requests session with default headers.
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "X-MEXC-APIKEY": self.api_key,
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json"
        })

        # Configure a retry strategy to handle transient errors (5xx, 429, etc.).
        retry_strategy = Retry(
            total=3,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS", "POST"],
            backoff_factor=1  # 1s, 2s, 4s delays, etc.
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)

    def _sign(self, params):
        """
        Sort parameters alphabetically, create a query string, then generate a
        signature using the HMAC-SHA256 method.
        """
        sorted_params = "&".join(f"{k}={params[k]}" for k in sorted(params))
        signature = hmac.new(
            self.api_secret.encode(), sorted_params.encode(), hashlib.sha256
        ).hexdigest()
        return signature

    def public_get(self, endpoint, params=None, timeout=30):
        """
        Makes a GET request to a public (unsigned) endpoint.
        Includes basic error handling for timeouts and connection issues.
        """
        if params is None:
            params = {}

        url = f"{self.base_url}{endpoint}"
        logger.debug("Public GET request", extra={"url": url, "params": params})

        try:
            response = self.session.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except Timeout:
            logger.error(f"Timeout while requesting {url}.")
            raise
        except ConnectionError:
            logger.error(f"Connection error while requesting {url}.")
            raise
        except RequestException as e:
            logger.error(f"HTTP request failed: {str(e)}")
            raise

    def _get(self, endpoint, params=None):
        """
        Makes a GET request to a private (signed) endpoint, adding timestamp and signature.
        """
        if params is None:
            params = {}

        # For authenticated endpoints, add timestamp and signature.
        params["timestamp"] = int(time.time() * 1000)
        params["signature"] = self._sign(params)
        url = f"{self.base_url}{endpoint}"
        logger.debug("GET request", extra={"url": url, "params": params})

        try:
            response = self.session.get(url, params=params, timeout=90)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error("Error in GET request", extra={"url": url, "error": str(e)})
            raise

    def _post(self, endpoint, data=None):
        """
        Makes a POST request to a private (signed) endpoint, adding timestamp and signature.
        """
        if data is None:
            data = {}

        data["timestamp"] = int(time.time() * 1000)
        data["signature"] = self._sign(data)
        url = f"{self.base_url}{endpoint}"
        logger.debug("POST request", extra={"url": url, "data": data})

        try:
            response = self.session.post(url, json=data, timeout=90)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error("Error in POST request", extra={"url": url, "error": str(e)})
            raise

    def get_instruments_info(self, symbol):
        """
        Fetch instrument details with exponential backoff for MEXC. 
        Many devs report the symbol must have underscores removed (e.g. BTC_USD -> BTCUSD).
        """
        clean_symbol = symbol.replace("_", "")
        endpoint = "/api/v1/contract/instruments"
        max_retries = 5
        retry_delay = 5  # start with 5 seconds delay
        params = {"symbol": clean_symbol}

        for attempt in range(max_retries):
            try:
                # Use our public GET method with a shorter initial timeout of 30s
                response = self.public_get(endpoint, params=params, timeout=30)
                return response
            except Timeout:
                logger.error(
                    f"[Attempt {attempt+1}/{max_retries}] Timeout fetching instrument info for {clean_symbol}. Retrying in {retry_delay}s..."
                )
            except ConnectionError:
                logger.error(
                    f"[Attempt {attempt+1}/{max_retries}] Connection error for {clean_symbol}. Retrying in {retry_delay}s..."
                )
            except RequestException as e:
                # Non-timeout, non-connection errors (like 4xx client errors)
                logger.error(f"[Attempt {attempt+1}/{max_retries}] Request failed: {str(e)}")
                break  # Don't retry on 4xx, just break

            time.sleep(retry_delay)
            retry_delay *= 2  # Exponential backoff

        logger.critical(
            f"Failed to retrieve instrument info for {clean_symbol} after {max_retries} attempts."
        )
        return None

    def get_positions(self, symbol):
        endpoint = "/api/v1/contract/position"
        params = {"symbol": symbol}
        return self._get(endpoint, params)

    def place_order(
        self, symbol, side, qty, orderType="Market", stopLoss=None, takeProfit=None, leverage=None
    ):
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
        data = {
            "symbol": symbol,
            "stopLoss": stopLoss,
            "takeProfit": takeProfit,
            "positionIdx": positionIdx,
        }
        return self._post(endpoint, data)

    def get_wallet_balance(self, accountType="UNIFIED"):
        endpoint = "/api/v1/contract/wallet"
        params = {"accountType": accountType}
        return self._get(endpoint, params)
