# keyboards/forex_menu.py

from telegram import InlineKeyboardButton, InlineKeyboardMarkup

def get_forex_menu() -> InlineKeyboardMarkup:
    """
    Forex group selection (3 groups). callback_data matches handler patterns like:
    - forex_dxy_eu_gu
    - forex_dxy_chf_jpy
    - forex_dxy_aud_nzd
    """
    buttons = [
        [InlineKeyboardButton("DXY / EURUSD / GBPUSD", callback_data="forex_dxy_eu_gu")],
        [InlineKeyboardButton("🔙 Back", callback_data="back_to_main")]
    ]
    return InlineKeyboardMarkup(buttons)
