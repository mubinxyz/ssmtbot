# utils/get_data.py

import requests
# import itertools
# from datetime import datetime, timedelta
import pandas as pd
import time
from utils.scrape_last_data import get_last_data
from utils.normalize_data import normalize_symbol, normalize_timeframe, normalize_ohlc  
import json


def get_ohlc(symbol: str, timeframe: int = 15, from_date: int = None, to_date: int = None) -> pd.DataFrame:
    """
    Get OHLC candles for a symbol.
    """
    norm_symbol = normalize_symbol(symbol)
    norm_timeframe = normalize_timeframe(timeframe)
    to_date = to_date or int(time.time())

    try:
        lite_finance_url = (
            f"https://lfdata.pmobint.workers.dev/"
            f"?symbol={norm_symbol}&tf={norm_timeframe}&from={from_date}&to={to_date}"
        )

        resp = requests.get(lite_finance_url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        ohlc_data = data.get("data", {})

        if ohlc_data:
            df = normalize_ohlc(ohlc_data)
            return df
    except Exception as e:
        print(f"[LiteFinance] OHLC error for {norm_symbol}: {e}")

    return pd.DataFrame()



def get_price(symbol: str) -> dict | None:
    """
    Get the latest price of a symbol via Cloudflare Worker.
    """
    norm_symbol = normalize_symbol(symbol)
    try:
        # use Worker endpoint
        worker_url = f"https://lfdata.pmobint.workers.dev/?symbol={norm_symbol}&tf=1&from=0&to={int(time.time())}"
        resp = requests.get(worker_url, timeout=15)
        resp.raise_for_status()
        data = resp.json().get("data", {})
        if not data:
            return None

        # Use last candle close as latest price
        price = data["c"][-1] if "c" in data and data["c"] else None
        if price is not None:
            return {
                "source": "litefinance worker",
                "symbol": norm_symbol,
                "price": float(price),
                "bid": float(price),
                "ask": float(price)
            }
    except Exception as e:
        print(f"[LiteFinance Worker] Error fetching price: {e}")
    return None
