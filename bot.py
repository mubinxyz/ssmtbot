# bot.py (updated - safe handler wrapper + alert job improvements)
import asyncio
import logging
import concurrent.futures
import time
import inspect
from typing import Dict, Any, Tuple, List, Callable, Awaitable

from telegram import Update, Bot
from telegram.error import BadRequest
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
from config import BOT_TOKEN, LOG_LEVEL

# db
from services.db_service import init_db
# command handlers (your modules)
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
    view_alert_handler,
    ssmt_alerts_handler,
    single_symbol_alerts_handler,
    show_trio_alerts_handler,
)

# alerts checking
from services.alert_service import get_active_alerts, check_alert_conditions

# logging
logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=LOG_LEVEL)
logger = logging.getLogger(__name__)

# Global executor for sync operations (DB, file I/O)
executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

# Alert check concurrency/timeouts (you already tuned these)
MAX_CONCURRENT_ALERT_CHECKS = 6
PER_ALERT_TIMEOUT = 30.0
GLOBAL_BATCH_TIMEOUT = 120.0

# -----------------------------
# Utilities for safe handlers
# -----------------------------
def safe_answer_callback(query):
    """
    Answer callback query safely (swallow BadRequest if it's stale).
    Use this helper inside handlers when you need to call query.answer().
    """
    if query is None:
        return
    try:
        # answer without text (fast)
        return query.answer()
    except BadRequest as e:
        # common: "Query is too old and response timeout expired" -> safe to ignore
        logger.debug("Stale CallbackQuery.answer ignored: %s", e)
    except Exception as e:
        # log but don't raise
        logger.exception("Unexpected error answering CallbackQuery: %s", e)

def safe_edit_message(edit_fn: Callable[..., Awaitable], *args, **kwargs):
    """
    Call an edit_message_* function safely (swallow BadRequest if stale or invalid).
    edit_fn should be an awaitable function like message.edit_text or bot.edit_message_text bound method.
    """
    async def _inner():
        try:
            return await edit_fn(*args, **kwargs)
        except BadRequest as e:
            logger.debug("Stale edit_message ignored: %s", e)
        except Exception:
            logger.exception("Unexpected error while editing message")
    return _inner()

def safe_handler(fn: Callable[[Update, ContextTypes.DEFAULT_TYPE], Awaitable[None]]):
    """
    Decorator for handlers to prevent uncaught exceptions from stopping the bot.
    Wraps the handler call and logs exceptions.
    """
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            # if there's a callback_query, attempt to answer early & safely to avoid Telegram timeout
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
            # swallow BadRequest coming from edits / answer attempts inside handler
            logger.debug("BadRequest in handler %s: %s", getattr(fn, "__name__", str(fn)), e)
        except Exception:
            logger.exception("Unhandled exception in handler %s", getattr(fn, "__name__", str(fn)))
    return wrapper

# -----------------------------
# Alert periodic job
# -----------------------------
async def check_all_alerts_periodically(context: ContextTypes.DEFAULT_TYPE):
    start_time = time.monotonic()
    logger.info("Running periodic alert check...")
    try:
        loop = asyncio.get_running_loop()
        # get_active_alerts is sync / IO bound, run it in threadpool
        active_alerts = await loop.run_in_executor(executor, get_active_alerts)
        if not active_alerts:
            logger.debug("No active alerts.")
            return

        alert_items = list(active_alerts.items())

        sem = asyncio.Semaphore(MAX_CONCURRENT_ALERT_CHECKS)

        async def _sem_wrapped_check(alert_key, alert_details):
            await sem.acquire()
            try:
                # pass context.bot into check_alert_conditions if it accepts bot param
                try:
                    sig = inspect.signature(check_alert_conditions)
                    accepts_bot = 'bot' in sig.parameters or 'context' in sig.parameters
                except Exception:
                    accepts_bot = False

                if accepts_bot:
                    # call with bot
                    try:
                        return await asyncio.wait_for(check_alert_conditions(alert_key, alert_details, bot=context.bot), timeout=PER_ALERT_TIMEOUT)
                    except asyncio.TimeoutError:
                        raise
                else:
                    return await asyncio.wait_for(check_alert_conditions(alert_key, alert_details), timeout=PER_ALERT_TIMEOUT)
            finally:
                sem.release()

        tasks = [asyncio.create_task(_sem_wrapped_check(k, v)) for k, v in alert_items]

        try:
            done, pending = await asyncio.wait(tasks, timeout=GLOBAL_BATCH_TIMEOUT, return_when=asyncio.ALL_COMPLETED)
        except Exception as e:
            logger.exception("Error awaiting alert tasks: %s", e)
            for t in tasks:
                if not t.done():
                    t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            return

        if pending:
            logger.warning("Global alert-check timeout reached; cancelling %d tasks", len(pending))
            for t in pending:
                t.cancel()
            await asyncio.gather(*pending, return_exceptions=True)

        # Log results
        for t in done:
            try:
                res = t.result()
                logger.debug("Alert check result: %s", res)
            except asyncio.CancelledError:
                logger.warning("A per-alert task was cancelled.")
            except Exception as e:
                logger.exception("Per-alert task raised: %s", e)

        end_time = time.monotonic()
        logger.info("Finished periodic alert check. Processed %d alerts in %.2f seconds.", len(alert_items), end_time - start_time)

    except Exception as e:
        logger.exception("Unexpected error in periodic alert check: %s", e)


# -----------------------------
# action_dispatcher (safe inside)
# -----------------------------
@safe_handler
async def action_dispatcher(update: Update, context: ContextTypes.DEFAULT_TYPE):
    cq = update.callback_query
    if cq is None:
        return
    # Try answer safely (we already do in the decorator but keep local guard)
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
    # dispatch map (same as your previous code)
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

    # Call the target handler (handlers should themselves be safe and quick)
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

    # Add job for alerts
    job_queue = app.job_queue
    job_queue.run_repeating(callback=check_all_alerts_periodically, interval=59, first=10)

    # Register handlers using safe_handler decorator
    app.add_handler(CommandHandler("start", safe_handler(start_command)))
    # if back_to_main_handler is a callback handler (CallbackQuery) wrap it
    app.add_handler(CallbackQueryHandler(safe_handler(back_to_main_handler), pattern="^back_to_main$"))

    # Main menus (CallbackQuery handlers)
    app.add_handler(CallbackQueryHandler(safe_handler(forex_menu_handler), pattern="^menu_forex$"))
    app.add_handler(CallbackQueryHandler(safe_handler(futures_menu_handler), pattern="^menu_futures$"))
    app.add_handler(CallbackQueryHandler(safe_handler(crypto_usd_menu_handler), pattern="^menu_crypto_usd$"))
    app.add_handler(CallbackQueryHandler(safe_handler(metals_menu_handler), pattern="^menu_metals$"))
    app.add_handler(CallbackQueryHandler(safe_handler(alerts_menu_handler), pattern="^menu_alerts$"))

    # alerts handlers
    app.add_handler(CallbackQueryHandler(safe_handler(delete_alert_handler), pattern="^delete_alert::"))
    app.add_handler(CallbackQueryHandler(safe_handler(view_alert_handler), pattern="^view_alert::"))
    app.add_handler(CallbackQueryHandler(safe_handler(ssmt_alerts_handler), pattern="^ssmt_alerts$"))
    app.add_handler(CallbackQueryHandler(safe_handler(single_symbol_alerts_handler), pattern="^single_symbol_alerts$"))
    app.add_handler(CallbackQueryHandler(safe_handler(show_trio_alerts_handler), pattern="^show_trio_alerts$"))

    # group selection
    app.add_handler(CallbackQueryHandler(safe_handler(forex_group_select), pattern="^(dxy_eu_gu|dxy_chf_jpy|dxy_aud_nzd)$"))
    app.add_handler(CallbackQueryHandler(safe_handler(futures_group_select), pattern="^(spx_nq_ym|es_nq_dow|spx_dow_nq)$"))
    app.add_handler(CallbackQueryHandler(safe_handler(crypto_group_select), pattern="^(btc_eth_xrp|btc_eth_total|btc_xrp_doge)$"))
    app.add_handler(CallbackQueryHandler(safe_handler(metals_group_select), pattern="^(dxy_xau_xag_aud|xau_xag_aud|dxy_xau_aud)$"))

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
