from telegram import InlineKeyboardButton, InlineKeyboardMarkup

def get_ssmt_action_menu(group_id: str, timeframe: str) -> InlineKeyboardMarkup:
    """
    Returns an inline keyboard with actions for a given asset group and timeframe.
    """
    prefix = f"{group_id}_{timeframe}"

    buttons = [
        [InlineKeyboardButton("📊 View Charts", callback_data=f"{prefix}_view_charts")],
        [InlineKeyboardButton("🔔 Activate Alert", callback_data=f"{prefix}_activate_alert")],
        [InlineKeyboardButton("🔕 Deactivate Alert", callback_data=f"{prefix}_deactivate_alert")],
        # Add more actions as needed
        [InlineKeyboardButton("🔙 Back", callback_data=f"{group_id}_back_to_timeframes")]
    ]
    return InlineKeyboardMarkup(buttons)
