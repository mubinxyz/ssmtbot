from telegram import InlineKeyboardButton, InlineKeyboardMarkup

def get_timeframe_menu(group_id: str) -> InlineKeyboardMarkup:
    """
    Returns an inline keyboard with timeframes for a given asset group.
    The group_id helps track which asset group the user is exploring.
    """
    buttons = [
        [InlineKeyboardButton("15 Minute SSMT", callback_data=f"{group_id}_15min_ssmt")],
        [InlineKeyboardButton("1 Hour SSMT", callback_data=f"{group_id}_1hr_ssmt")],
        # Add more timeframes as needed
        [InlineKeyboardButton("🔙 Back", callback_data=f"{group_id}_back")]
    ]
    return InlineKeyboardMarkup(buttons)
