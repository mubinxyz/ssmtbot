# handlers/back_to_forex_handler.py

from telegram import Update
from telegram.ext import ContextTypes
from keyboards.forex_menu import get_forex_menu

async def back_to_forex_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handles 'Back to Forex Menu' button.
    """
    query = update.callback_query
    await query.answer()

    await query.edit_message_text(
        text="🔙 Back to Forex Currencies:\nChoose a pair group:",
        reply_markup=get_forex_menu()
    )
