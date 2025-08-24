# handlers/action_dispatcher.py
from telegram import Update
from telegram.ext import ContextTypes
import logging

# Import group sets and handlers
from handlers.forex_handler import GROUP_ID_SET as FOREX_GROUPS, charts_handler as forex_charts, activate_handler as forex_activate, deactivate_handler as forex_deactivate
from handlers.futures_handler import GROUP_ID_SET as FUTURES_GROUPS, charts_handler as futures_charts, activate_handler as futures_activate, deactivate_handler as futures_deactivate
from handlers.crypto_usd_handler import GROUP_ID_SET as CRYPTO_GROUPS, charts_handler as crypto_charts, activate_handler as crypto_activate, deactivate_handler as crypto_deactivate
from handlers.metals_handler import GROUP_ID_SET as METALS_GROUPS, charts_handler as metals_charts, activate_handler as metals_activate, deactivate_handler as metals_deactivate
# CORRECTED: Import actual function names from energy_handler
from handlers.energy_handler import GROUP_ID_SET as ENERGY_GROUPS, charts_handler as energy_charts, activate_handler as energy_activate, deactivate_handler as energy_deactivate


logger = logging.getLogger(__name__)

async def action_dispatcher(update: Update, context: ContextTypes.DEFAULT_TYPE):
    cq = update.callback_query
    await cq.answer()
    data = cq.data or ""
    parts = data.split("::")
    # IMPROVED: Check for minimum required parts (action::gid)
    if len(parts) < 2:
        # fall back or show error
        await cq.edit_message_text("Invalid action.")
        return

    action, gid = parts[0], parts[1]
    # pick module by gid
    target = None
    if gid in FOREX_GROUPS:
        target = {
            "charts": forex_charts,
            "activate": forex_activate,
            "deactivate": forex_deactivate
        }.get(action)
    elif gid in FUTURES_GROUPS:
        target = {
            "charts": futures_charts,
            "activate": futures_activate,
            "deactivate": futures_deactivate
        }.get(action)
    elif gid in CRYPTO_GROUPS:
        target = {
            "charts": crypto_charts,
            "activate": crypto_activate,
            "deactivate": crypto_deactivate
        }.get(action)
    elif gid in METALS_GROUPS:
        target = {
            "charts": metals_charts,
            "activate": metals_activate,
            "deactivate": metals_deactivate
        }.get(action)
    # ADDED: Handler for ENERGY_GROUPS
    elif gid in ENERGY_GROUPS: # <--- This was missing/wrong
        target = {
            "charts": energy_charts,   # <--- Use correct alias
            "activate": energy_activate, # <--- Use correct alias
            "deactivate": energy_deactivate # <--- Use correct alias
        }.get(action)

    if target is None:
        await cq.edit_message_text("Unknown group or action.")
        return

    # call the actual handler from the target module (it expects the same args)
    await target(update, context)
