from telegram import Update
from telegram.ext import ContextTypes
from keyboards.timeframe_menu import get_timeframe_menu

async def forex_dxy_eu_gu_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handles selection of 'Dxy, EU, GU' and shows available timeframes.
    """
    query = update.callback_query
    await query.answer()

    await query.edit_message_text(
        text="🕒 Choose a timeframe for Dxy, EU, GU:",
        reply_markup=get_timeframe_menu("forex_dxy_eu_gu")
    )
