# keyboards/main_menu.py

from telegram import InlineKeyboardButton, InlineKeyboardMarkup

def get_main_menu() -> InlineKeyboardMarkup:
    """
    Returns the main menu inline keyboard with market categories and an Alerts entry.
    """
    buttons = [
        [InlineKeyboardButton("📊 FOREX CURRENCIES", callback_data="menu_forex")],
        [InlineKeyboardButton("📈 FUTURES (STOCKS)", callback_data="menu_futures")],
        [InlineKeyboardButton("💰 CRYPTO CURRENCY (USD)", callback_data="menu_crypto_usd")],
        [InlineKeyboardButton("💵 CRYPTO CURRENCY (USDT)", callback_data="menu_crypto_usdt")],
        [InlineKeyboardButton("🪙 METALS", callback_data="menu_metals")],
        # Alerts row appended at the end:
        [InlineKeyboardButton("🔔 Alerts", callback_data="menu_alerts")],
    ]
    return InlineKeyboardMarkup(buttons)
