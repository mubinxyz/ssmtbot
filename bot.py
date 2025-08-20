import asyncio
import logging
import concurrent.futures
import time
import inspect
from typing import Callable, Awaitable

from telegram import Update
from telegram.error import BadRequest
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
from config import BOT_TOKEN, LOG_LEVEL

# db
from services.db_service import init_db
# command handlers
from commands.start_command import start_command, back_to_main_handler
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
# alerts handlers
from handlers.alerts_handler import (
    alerts_menu_handler,
    delete_alert_handler,
    single_symbol_alerts_handler,
    show_alerts_handler,
)

# Use the batched coordinator from the alert service
import services.alert_service as alert_service

# logging
logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=LOG_LEVEL)
logger = logging.getLogger(__name__)

# Global executor for sync operations (DB, file I/O)
executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

# These were used by the old per-alert runner; keep if you need elsewhere
MAX_CONCURRENT_ALERT_CHECKS = 6
PER_ALERT_TIMEOUT = 30.0
GLOBAL_BATCH_TIMEOUT = 120.0


# -----------------------------
# Utilities for safe handlers
# -----------------------------
def safe_answer_callback(query):
    if query is None:
        return
    try:
        return query.answer()
    except BadRequest as e:
        logger.debug("Stale CallbackQuery.answer ignored: %s", e)
    except Exception as e:
        logger.exception("Unexpected error answering CallbackQuery: %s", e)


def safe_edit_message(edit_fn: Callable[..., Awaitable], *args, **kwargs):
    async def _inner():
        try:
            return await edit_fn(*args, **kwargs)
        except BadRequest as e:
            logger.debug("Stale edit_message ignored: %s", e)
        except Exception:
            logger.exception("Unexpected error while editing message")
    return _inner()


def safe_handler(fn: Callable[[Update, ContextTypes.DEFAULT_TYPE], Awaitable[None]]):
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            cq = getattr(update, "callback_query", None)
            if cq is not None:
                try:
                    await cq.answer()
                except BadRequest as e:
                    logger.debug("safe_handler: callback answer stale/BadRequest: %s", e)
                except Exception:
                    logger.exception("safe_handler: error answering callback query")
            await fn(update, context)
        except BadRequest as e:
            logger.debug("BadRequest in handler %s: %s", getattr(fn, "__name__", str(fn)), e)
        except Exception:
            logger.exception("Unhandled exception in handler %s", getattr(fn, "__name__", str(fn)))
    return wrapper


# -----------------------------
# Periodic coordinator job
# -----------------------------
async def check_all_alerts_periodically(context: ContextTypes.DEFAULT_TYPE):
    """
    JobQueue callback — schedule the batched coordinator as a background task
    so the JobQueue remains responsive. The real work is done inside
    alert_service.check_all_alerts_periodically_coordinator.
    """
    try:
        # schedule coordinator on the same loop (do not await here)
        asyncio.create_task(alert_service.check_all_alerts_periodically_coordinator(bot=context.bot))
    except Exception as e:
        logger.exception("_periodic_job_callback: failed to schedule coordinator: %s", e)


# -----------------------------
# action_dispatcher (safe inside)
# -----------------------------
@safe_handler
async def action_dispatcher(update: Update, context: ContextTypes.DEFAULT_TYPE):
    cq = update.callback_query
    if cq is None:
        return
    try:
        await cq.answer()
    except BadRequest:
        logger.debug("action_dispatcher: callback too old (ignored).")
    except Exception:
        logger.exception("action_dispatcher: failed to answer callback")

    data = cq.data or ""
    parts = data.split("::")
    if len(parts) < 2:
        try:
            await cq.edit_message_text("Invalid action.")
        except BadRequest:
            logger.debug("action_dispatcher: edit_message_text failed (stale).")
        except Exception:
            logger.exception("action_dispatcher: error editing message")
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
        except BadRequest:
            logger.debug("action_dispatcher: edit_message_text failed (stale).")
        except Exception:
            logger.exception("action_dispatcher: error editing message")
        return

    try:
        await target(update, context)
    except Exception:
        logger.exception("action_dispatcher: target handler raised")


def cleanup_executor():
    global executor
    if executor:
        executor.shutdown(wait=True)
        logger.info("Executor shutdown completed")


def main():
    init_db()
    app = Application.builder().token(BOT_TOKEN).build()

    # Add job for alerts: use JobQueue to run repeating callback which schedules the coordinator.
    job_queue = app.job_queue
    job_queue.run_repeating(check_all_alerts_periodically, interval=15, first=5)

    # Register handlers using safe_handler decorator
    app.add_handler(CommandHandler("start", safe_handler(start_command)))
    app.add_handler(CallbackQueryHandler(safe_handler(back_to_main_handler), pattern=r"^back_to_main$"))

    # Main menus (CallbackQuery handlers)
    app.add_handler(CallbackQueryHandler(safe_handler(forex_menu_handler), pattern=r"^menu_forex$"))
    app.add_handler(CallbackQueryHandler(safe_handler(futures_menu_handler), pattern=r"^menu_futures$"))
    app.add_handler(CallbackQueryHandler(safe_handler(crypto_usd_menu_handler), pattern=r"^menu_crypto_usd$"))
    app.add_handler(CallbackQueryHandler(safe_handler(metals_menu_handler), pattern=r"^menu_metals$"))
    app.add_handler(CallbackQueryHandler(safe_handler(alerts_menu_handler), pattern=r"^menu_alerts$"))

    # alerts handlers (show list, delete per-alert, single-symbol placeholder)
    app.add_handler(CallbackQueryHandler(safe_handler(show_alerts_handler), pattern=r"^show_alerts$"))
    app.add_handler(CallbackQueryHandler(safe_handler(delete_alert_handler), pattern=r"^delete_alert::"))
    app.add_handler(CallbackQueryHandler(safe_handler(single_symbol_alerts_handler), pattern=r"^single_symbol_alerts$"))

    # group selection
    app.add_handler(CallbackQueryHandler(safe_handler(forex_group_select), pattern=r"^(dxy_eu_gu|dxy_chf_jpy|dxy_aud_nzd)$"))
    app.add_handler(CallbackQueryHandler(safe_handler(futures_group_select), pattern=r"^(spx_nq_ym|es_nq_dow|spx_dow_nq)$"))
    app.add_handler(CallbackQueryHandler(safe_handler(crypto_group_select), pattern=r"^(btc_eth_xrp|btc_eth_total|btc_xrp_doge)$"))
    app.add_handler(CallbackQueryHandler(safe_handler(metals_group_select), pattern=r"^(dxy_xau_xag_aud|xau_xag_aud|dxy_xau_aud)$"))

    # timeframe selection handlers
    app.add_handler(CallbackQueryHandler(safe_handler(forex_timeframe_select), pattern=r"^timeframe::(dxy_eu_gu|dxy_chf_jpy|dxy_aud_nzd)::"))
    app.add_handler(CallbackQueryHandler(safe_handler(futures_timeframe_select), pattern=r"^timeframe::(spx_nq_ym|es_nq_dow|spx_dow_nq)::"))
    app.add_handler(CallbackQueryHandler(safe_handler(crypto_timeframe_select), pattern=r"^timeframe::(btc_eth_xrp|btc_eth_total|btc_xrp_doge)::"))
    app.add_handler(CallbackQueryHandler(safe_handler(metals_timeframe_select), pattern=r"^timeframe::(dxy_xau_xag_aud|xau_xag_aud|dxy_xau_aud)::"))

    # single action dispatcher (charts/activate/deactivate)
    app.add_handler(CallbackQueryHandler(safe_handler(action_dispatcher), pattern=r"^(charts|activate|deactivate)::"))

    # Register cleanup at exit
    import atexit
    atexit.register(cleanup_executor)

    logger.info("🤖 Bot is running...")
    try:
        app.run_polling()
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    finally:
        cleanup_executor()


if __name__ == "__main__":
    main()
