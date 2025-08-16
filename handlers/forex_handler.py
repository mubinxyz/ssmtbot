from telegram import Update
from telegram.ext import ContextTypes
from keyboards.forex_menu import get_forex_menu

async def forex_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handles the 'FOREX CURRENCIES' menu selection and shows Forex pairs.
    """
    query = update.callback_query
    await query.answer()

    await query.edit_message_text(
        text="📊 Forex Currencies:\nChoose a pair group to explore:",
        reply_markup=get_forex_menu()
    )
