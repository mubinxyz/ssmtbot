# bot.py
import logging
from telegram.ext import Application, CommandHandler, CallbackQueryHandler
from config import BOT_TOKEN, LOG_LEVEL

# db
from services.db_service import init_db

# command
from commands.start_command import start_command
from commands.start_command import back_to_main_handler

# forex
from handlers.forex_handler import (
    forex_menu_handler,
    group_select_handler as forex_group_select,
    timeframe_select_handler as forex_timeframe_select,
    charts_handler as forex_charts,
    activate_handler as forex_activate,
    deactivate_handler as forex_deactivate,
    GROUP_ID_SET as FOREX_GROUPS,
)

# futures
from handlers.futures_handler import (
    futures_menu_handler,
    group_select_handler as futures_group_select,
    timeframe_select_handler as futures_timeframe_select,
    charts_handler as futures_charts,
    activate_handler as futures_activate,
    deactivate_handler as futures_deactivate,
    GROUP_ID_SET as FUTURES_GROUPS,
)

# crypto USD
from handlers.crypto_usd_handler import (
    crypto_usd_menu_handler,
    group_select_handler as crypto_group_select,
    timeframe_select_handler as crypto_timeframe_select,
    charts_handler as crypto_charts,
    activate_handler as crypto_activate,
    deactivate_handler as crypto_deactivate,
    GROUP_ID_SET as CRYPTO_GROUPS,
)

# metals
from handlers.metals_handler import (
    metals_menu_handler,
    group_select_handler as metals_group_select,
    timeframe_select_handler as metals_timeframe_select,
    charts_handler as metals_charts,
    activate_handler as metals_activate,
    deactivate_handler as metals_deactivate,
    GROUP_ID_SET as METALS_GROUPS,
)

# alerts
from handlers.alerts_handler import (
    alerts_menu_handler,
    delete_alert_handler,
    view_alert_handler,
    ssmt_alerts_handler,
    single_symbol_alerts_handler,
)

# logging
logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=LOG_LEVEL)
logger = logging.getLogger(__name__)


async def action_dispatcher(update, context):
    cq = update.callback_query
    if cq is None:
        return
    await cq.answer()
    data = cq.data or ""
    parts = data.split("::")
    if len(parts) < 2:
        try:
            await cq.edit_message_text("Invalid action.")
        except Exception:
            pass
        return

    action = parts[0]
    gid = parts[1]
    target = None

    if gid in FOREX_GROUPS:
        if action == "charts":
            target = forex_charts
        elif action == "activate":
            target = forex_activate
        elif action == "deactivate":
            target = forex_deactivate

    elif gid in FUTURES_GROUPS:
        if action == "charts":
            target = futures_charts
        elif action == "activate":
            target = futures_activate
        elif action == "deactivate":
            target = futures_deactivate

    elif gid in CRYPTO_GROUPS:
        if action == "charts":
            target = crypto_charts
        elif action == "activate":
            target = crypto_activate
        elif action == "deactivate":
            target = crypto_deactivate

    elif gid in METALS_GROUPS:
        if action == "charts":
            target = metals_charts
        elif action == "activate":
            target = metals_activate
        elif action == "deactivate":
            target = metals_deactivate

    if target is None:
        try:
            await cq.edit_message_text("Unknown group or action.")
        except Exception:
            logger.debug("Unknown action/gid and cannot edit message")
        return

    await target(update, context)


def main():
    init_db()  # Create tables if not exist
    
    app = Application.builder().token(BOT_TOKEN).build()

    # commands
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CallbackQueryHandler(back_to_main_handler, pattern="^back_to_main$"))

    # main menus
    app.add_handler(CallbackQueryHandler(forex_menu_handler, pattern="^menu_forex$"))
    app.add_handler(CallbackQueryHandler(futures_menu_handler, pattern="^menu_futures$"))
    app.add_handler(CallbackQueryHandler(crypto_usd_menu_handler, pattern="^menu_crypto_usd$"))
    app.add_handler(CallbackQueryHandler(metals_menu_handler, pattern="^menu_metals$"))
    app.add_handler(CallbackQueryHandler(alerts_menu_handler, pattern="^menu_alerts$"))

    # alerts handlers
    app.add_handler(CallbackQueryHandler(delete_alert_handler, pattern="^delete_alert::"))
    app.add_handler(CallbackQueryHandler(view_alert_handler, pattern="^view_alert::"))
    app.add_handler(CallbackQueryHandler(ssmt_alerts_handler, pattern="^ssmt_alerts$"))
    app.add_handler(CallbackQueryHandler(single_symbol_alerts_handler, pattern="^single_symbol_alerts$"))

    # group selection
    app.add_handler(CallbackQueryHandler(forex_group_select, pattern="^(dxy_eu_gu|dxy_chf_jpy|dxy_aud_nzd)$"))
    app.add_handler(CallbackQueryHandler(futures_group_select, pattern="^(spx_nq_ym|es_nq_dow|spx_dow_nq)$"))
    app.add_handler(CallbackQueryHandler(crypto_group_select, pattern="^(btc_eth_xrp|btc_eth_total|btc_xrp_doge)$"))
    app.add_handler(CallbackQueryHandler(metals_group_select, pattern="^(dxy_xau_xag_aud|xau_xag_aud|dxy_xau_aud)$"))

    # timeframe selection handlers (timeframe-first UX)
    app.add_handler(CallbackQueryHandler(forex_timeframe_select, pattern=r"^timeframe::(dxy_eu_gu|dxy_chf_jpy|dxy_aud_nzd)::"))
    app.add_handler(CallbackQueryHandler(futures_timeframe_select, pattern=r"^timeframe::(spx_nq_ym|es_nq_dow|spx_dow_nq)::"))
    app.add_handler(CallbackQueryHandler(crypto_timeframe_select, pattern=r"^timeframe::(btc_eth_xrp|btc_eth_total|btc_xrp_doge)::"))
    app.add_handler(CallbackQueryHandler(metals_timeframe_select, pattern=r"^timeframe::(dxy_xau_xag_aud|xau_xag_aud|dxy_xau_aud)::"))

    # single action dispatcher (charts/activate/deactivate)
    app.add_handler(CallbackQueryHandler(action_dispatcher, pattern=r"^(charts|activate|deactivate)::"))

    logger.info("🤖 Bot is running...")
    app.run_polling()


if __name__ == "__main__":
    main()