# keyboards/futures_menu.py

from telegram import InlineKeyboardButton, InlineKeyboardMarkup

def get_futures_menu() -> InlineKeyboardMarkup:
    """
    Futures (stock indices) groups. Use common tickers ES / NQ / DOW etc.
    callback_data:
    - futures_es_nq_dow
    - futures_nq_es_dow
    - futures_es_dow_nq
    """
    buttons = [
        [InlineKeyboardButton("ES / NQ / DOW", callback_data="futures_es_nq_dow")],
        # [InlineKeyboardButton("NQ / ES / DOW", callback_data="futures_nq_es_dow")],
        # [InlineKeyboardButton("ES / DOW / NQ", callback_data="futures_es_dow_nq")],
        [InlineKeyboardButton("🔙 Back", callback_data="back_to_main")]
    ]
    return InlineKeyboardMarkup(buttons)