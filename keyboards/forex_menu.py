from telegram import InlineKeyboardButton, InlineKeyboardMarkup

def get_forex_menu() -> InlineKeyboardMarkup:
    """
    Returns the inline keyboard for Forex currency pair groups.
    """
    buttons = [
        [InlineKeyboardButton("Dxy, EU, GU", callback_data="forex_dxy_eu_gu")],
        [InlineKeyboardButton("DXY, CHF, JPY", callback_data="forex_dxy_chf_jpy")],
        # Add more groups as needed
        [InlineKeyboardButton("🔙 Back", callback_data="back_to_main")]
    ]
    return InlineKeyboardMarkup(buttons)
