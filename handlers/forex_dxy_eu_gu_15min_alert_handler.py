# handlers/forex_dxy_eu_gu_15min_alert_handler.py

from telegram import Update
from telegram.ext import ContextTypes
from services.alert_service import set_ssmt_alert

async def forex_dxy_eu_gu_15min_alert_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handles activation of the 15min SSMT alert for Dxy, EU, GU.
    """
    query = update.callback_query
    await query.answer()

    user_id = query.from_user.id
    group_id = "forex_dxy_eu_gu"
    timeframe = "15min"

    # Set the alert using your service logic
    set_ssmt_alert(user_id=user_id, group=group_id, timeframe=timeframe)

    await query.edit_message_text(
        text="✅ Alert activated for 15min SSMT Dxy, EU, GU.\nYou will be notified if the rule is violated."
    )
