# bot.py
import logging
from telegram.ext import Application, CommandHandler, CallbackQueryHandler
from config import BOT_TOKEN, LOG_LEVEL

# /start command
from commands.start_command import start_command

# Forex handlers
from handlers.forex_handler import (
    forex_menu_handler,
    group_select_handler as forex_group_select,
    timeframe_select_handler as forex_timeframe_select,
    charts_handler as forex_charts,
    activate_handler as forex_activate,
    deactivate_handler as forex_deactivate,
    GROUP_ID_SET as FOREX_GROUPS,
)

# Futures handlers
from handlers.futures_handler import (
    futures_menu_handler,
    group_select_handler as futures_group_select,
    timeframe_select_handler as futures_timeframe_select,
    charts_handler as futures_charts,
    activate_handler as futures_activate,
    deactivate_handler as futures_deactivate,
    GROUP_ID_SET as FUTURES_GROUPS,
)

# Crypto (USD) handlers
from handlers.crypto_usd_handler import (
    crypto_usd_menu_handler,
    group_select_handler as crypto_group_select,
    timeframe_select_handler as crypto_timeframe_select,
    charts_handler as crypto_charts,
    activate_handler as crypto_activate,
    deactivate_handler as crypto_deactivate,
    GROUP_ID_SET as CRYPTO_GROUPS,
)

# Metals handlers
from handlers.metals_handler import (
    metals_menu_handler,
    group_select_handler as metals_group_select,
    timeframe_select_handler as metals_timeframe_select,
    charts_handler as metals_charts,
    activate_handler as metals_activate,
    deactivate_handler as metals_deactivate,
    GROUP_ID_SET as METALS_GROUPS,
)

# Alerts & navigation
from handlers.alerts_handler import (
    menu_alerts_handler,
    alerts_one_symbol_handler,
    alerts_trio_handler,
    trio_group_list_all_handler,
    manage_active_trio_handler,
    manage_active_trio_view_handler,
    manage_active_trio_deactivate_handler,
)

from handlers.back_to_main_handler import back_to_main_handler
from handlers.back_to_forex_handler import back_to_forex_handler
from handlers.back_to_timeframe_handler import back_to_timeframe_handler

# logging
logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=LOG_LEVEL)
logger = logging.getLogger(__name__)


async def action_dispatcher(update, context):
    """
    Central dispatcher for callbacks of the form:
      charts::{gid}::{tf}
      activate::{gid}::{tf}
      deactivate::{gid}::{tf}

    Determines which category owns gid, then forwards the call to the
    corresponding handler (which are designed to accept (update, context)).
    """
    cq = update.callback_query
    if cq is None:
        return
    await cq.answer()
    data = cq.data or ""
    parts = data.split("::")
    if len(parts) < 2:
        await cq.edit_message_text("Invalid action.")
        return

    action = parts[0]
    gid = parts[1]

    # map gid -> module handlers
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
        # unknown gid or unsupported action
        try:
            await cq.edit_message_text("Unknown group or action.")
        except Exception:
            logger.debug("Could not edit message for unknown action/gid")
        return

    # forward to the module handler (they expect update, context)
    await target(update, context)


def main():
    app = Application.builder().token(BOT_TOKEN).build()

    # /start
    app.add_handler(CommandHandler("start", start_command))

    # Main menus
    app.add_handler(CallbackQueryHandler(forex_menu_handler, pattern="^menu_forex$"))
    app.add_handler(CallbackQueryHandler(futures_menu_handler, pattern="^menu_futures$"))
    app.add_handler(CallbackQueryHandler(crypto_usd_menu_handler, pattern="^menu_crypto_usd$"))
    app.add_handler(CallbackQueryHandler(metals_menu_handler, pattern="^menu_metals$"))
    app.add_handler(CallbackQueryHandler(menu_alerts_handler, pattern="^menu_alerts$"))

    # Back navigation
    app.add_handler(CallbackQueryHandler(back_to_main_handler, pattern="^back_to_main$"))
    app.add_handler(CallbackQueryHandler(back_to_forex_handler, pattern="^back_to_forex$"))
    app.add_handler(CallbackQueryHandler(back_to_timeframe_handler, pattern="^back_to_timeframes$"))

    # Group selection handlers (user clicks a group button)
    app.add_handler(CallbackQueryHandler(forex_group_select, pattern="^(dxy_eu_gu|dxy_chf_jpy|dxy_aud_nzd)$"))
    app.add_handler(CallbackQueryHandler(futures_group_select, pattern="^(spx_nq_ym|es_nq_dow|spx_dow_nq)$"))
    app.add_handler(CallbackQueryHandler(crypto_group_select, pattern="^(btc_eth_xrp|btc_eth_total|btc_xrp_doge)$"))
    app.add_handler(CallbackQueryHandler(metals_group_select, pattern="^(dxy_xau_xag_aud|xau_xag_aud|dxy_xau_aud)$"))

    # Timeframe selection handlers (timeframe-first UX)
    app.add_handler(CallbackQueryHandler(forex_timeframe_select, pattern=r"^timeframe::(dxy_eu_gu|dxy_chf_jpy|dxy_aud_nzd)::"))
    app.add_handler(CallbackQueryHandler(futures_timeframe_select, pattern=r"^timeframe::(spx_nq_ym|es_nq_dow|spx_dow_nq)::"))
    app.add_handler(CallbackQueryHandler(crypto_timeframe_select, pattern=r"^timeframe::(btc_eth_xrp|btc_eth_total|btc_xrp_doge)::"))
    app.add_handler(CallbackQueryHandler(metals_timeframe_select, pattern=r"^timeframe::(dxy_xau_xag_aud|xau_xag_aud|dxy_xau_aud)::"))

    # Single global dispatcher for charts / activate / deactivate
    app.add_handler(CallbackQueryHandler(action_dispatcher, pattern=r"^(charts|activate|deactivate)::"))

    # Alerts-specific handlers
    app.add_handler(CallbackQueryHandler(alerts_one_symbol_handler, pattern="^alerts_one_symbol$"))
    app.add_handler(CallbackQueryHandler(alerts_trio_handler, pattern="^alerts_trio$"))
    app.add_handler(CallbackQueryHandler(trio_group_list_all_handler, pattern="^trio_group_list_all$"))
    app.add_handler(CallbackQueryHandler(manage_active_trio_handler, pattern=r"^manage_active_trio::"))
    app.add_handler(CallbackQueryHandler(manage_active_trio_view_handler, pattern=r"^manage_active_trio_view::"))
    app.add_handler(CallbackQueryHandler(manage_active_trio_deactivate_handler, pattern=r"^manage_active_trio_deactivate::"))

    # start
    logger.info("🤖 Bot is running...")
    app.run_polling()


if __name__ == "__main__":
    main()
