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
    Get the latest price of a symbol.
    1. Try scraping LiteFinance.
    2. If scraping fails, use TwelveData API.
    """
    norm_symbol = normalize_symbol(symbol)

    # --- 1. Try LiteFinance scrape ---
    try:
        result_raw = get_last_data(symbol)  # this is a string
        # parse it into a dict
        if isinstance(result_raw, str):
            result = json.loads(result_raw)
        else:
            result = result_raw

        price = result.get("price")
        bid = result.get("bid")
        ask = result.get("ask")
        if price is not None:
            return {
                "source": "litefinance scraped last data",
                "symbol": norm_symbol,
                "price": float(price),
                "bid": bid,
                "ask": ask
            }
        else:
            print(f"[LiteFinance] Script returned no price.")
    except Exception as e:
        print(f"[LiteFinance] Error: {e}")