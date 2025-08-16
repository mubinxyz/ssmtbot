# handlers/back_to_main_handler.py

from telegram import Update
from telegram.ext import ContextTypes
from keyboards.main_menu import get_main_menu

async def back_to_main_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handles 'Back to Main Menu' button.
    """
    query = update.callback_query
    await query.answer()

    await query.edit_message_text(
        text="🔙 Back to main menu. Choose a category:",
        reply_markup=get_main_menu()
    )
