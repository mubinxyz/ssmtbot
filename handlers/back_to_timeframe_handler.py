# handlers/back_to_timeframe_handler.py


from telegram import Update
from telegram.ext import ContextTypes
from keyboards.timeframe_menu import get_timeframe_menu

async def back_to_timeframe_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handles 'Back to Timeframes' button for a specific group.
    Extracts group_id from callback_data.
    """
    query = update.callback_query
    await query.answer()

    # Example: "forex_dxy_eu_gu_back_to_timeframes"
    callback_data = query.data
    group_id = callback_data.replace("_back_to_timeframes", "")

    await query.edit_message_text(
        text=f"🔙 Back to timeframes for {group_id.replace('_', ', ').upper()}:",
        reply_markup=get_timeframe_menu(group_id)
    )
