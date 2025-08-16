from telegram import Update
from telegram.ext import ContextTypes
from keyboards.ssmt_action_menu import get_ssmt_action_menu

async def forex_dxy_eu_gu_15min_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handles selection of '15 Minute SSMT' for Dxy, EU, GU and shows available actions.
    """
    query = update.callback_query
    await query.answer()

    await query.edit_message_text(
        text="📉 15 Minute SSMT for Dxy, EU, GU:\nChoose an action:",
        reply_markup=get_ssmt_action_menu("forex_dxy_eu_gu", "15min")
    )
