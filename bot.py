# bot.py
import asyncio
import logging
import concurrent.futures
import time
import inspect
from typing import Dict, Any, Tuple, List

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
    show_trio_alerts_handler,
)
# --- Import for alert checking ---
from services.alert_service import get_active_alerts, check_alert_conditions

# logging
logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=LOG_LEVEL)
logger = logging.getLogger(__name__)

# Global executor for sync operations (DB, get_active_alerts, etc.)
executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

# Configuration for alert-check concurrency/timeouts
MAX_CONCURRENT_ALERT_CHECKS = 6        # throttle concurrent alert checks
PER_ALERT_TIMEOUT = 30.0               # seconds per alert (same as before)
GLOBAL_BATCH_TIMEOUT = 120.0           # seconds for the whole batch


async def check_all_alerts_periodically(context):
    """
    Periodic job that checks all active alerts.
    It runs with concurrency limiting and robust timeout/cleanup.
    `context` is the JobContext provided by PTB (and contains .bot).
    """
    start_time = time.monotonic()
    logger.info("Running periodic alert check...")
    try:
        # Get active alerts in a thread (safe if load_alerts is IO-bound)
        loop = asyncio.get_running_loop()
        active_alerts: Dict[str, Dict[str, Any]] = await loop.run_in_executor(executor, get_active_alerts)

        if not active_alerts:
            logger.info("No active alerts to check.")
            return

        alert_items: List[Tuple[str, Dict[str, Any]]] = list(active_alerts.items())

        # Semaphore to limit concurrency
        sem = asyncio.Semaphore(MAX_CONCURRENT_ALERT_CHECKS)

        # Build tasks: each task wraps the per-alert check and is throttled by semaphore
        tasks = []
        for alert_key, alert_details in alert_items:
            # pass context.bot down so alert service can reuse application bot if it supports it
            task = asyncio.create_task(_sem_wrapped_check(alert_key, alert_details, context.bot, sem))
            tasks.append(task)

        # Wait for all tasks with a global timeout
        try:
            done, pending = await asyncio.wait(tasks, timeout=GLOBAL_BATCH_TIMEOUT, return_when=asyncio.ALL_COMPLETED)
        except Exception as e:
            logger.exception("Unexpected error awaiting alert check tasks: %s", e)
            # Cancel all tasks if wait() itself errored
            for t in tasks:
                if not t.done():
                    t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            return

        # If any are still pending after the global timeout, cancel them
        if pending:
            logger.warning("Global timeout (%ss) reached: cancelling %d pending alert checks.", GLOBAL_BATCH_TIMEOUT, len(pending))
            for t in pending:
                t.cancel()
            # Give cancelled tasks a moment to finish cleanup
            await asyncio.gather(*pending, return_exceptions=True)

        # Process results and log exceptions
        results = []
        for t in done:
            try:
                res = t.result()  # will raise if the task raised
                results.append(res)
            except asyncio.CancelledError:
                results.append(asyncio.CancelledError("task cancelled"))
            except Exception as e:
                results.append(e)

        # Pair results with alert keys (order of `done` is not guaranteed; we pair by original order)
        # We'll iterate tasks in the same order as alert_items to keep pairing predictable
        for idx, (alert_key, _) in enumerate(alert_items):
            if idx < len(results):
                result = results[idx]
                if isinstance(result, Exception):
                    if isinstance(result, asyncio.TimeoutError):
                        logger.error("Timeout (individual) checking alert %s: %s", alert_key, result)
                    elif isinstance(result, asyncio.CancelledError):
                        logger.warning("Alert check for %s was cancelled.", alert_key)
                    else:
                        logger.error("Error checking alert %s: %s", alert_key, result)
                else:
                    # result expected to be (bool, message, chart_data) or None
                    logger.debug("Result for %s: %s", alert_key, result)
            else:
                # No result available for this index (shouldn't normally happen)
                logger.debug("No result entry for index %d (alert %s)", idx, alert_key)

        end_time = time.monotonic()
        duration = end_time - start_time
        logger.info("Finished periodic alert check. Processed %d alerts in %.2f seconds.", len(active_alerts), duration)

    except Exception as e:
        logger.exception("Unexpected error at the start of alert checker: %s", e)


async def _sem_wrapped_check(alert_key: str, alert_details: dict, bot, sem: asyncio.Semaphore):
    """
    Acquire semaphore, then run the per-alert check with a per-alert timeout.
    Uses _check_single_alert_safe to maintain the same safety semantics.
    """
    await sem.acquire()
    try:
        # Respect PER_ALERT_TIMEOUT for each check; raise TimeoutError if exceeded
        try:
            return await asyncio.wait_for(_check_single_alert_safe(alert_key, alert_details, bot), timeout=PER_ALERT_TIMEOUT)
        except asyncio.TimeoutError:
            logger.warning("Per-alert timeout (%ss) reached for %s", PER_ALERT_TIMEOUT, alert_key)
            raise
    finally:
        sem.release()


async def _check_single_alert_safe(alert_key, alert_details, bot):
    """
    Safely check a single alert with timeout and optional bot passed.
    We inspect the signature of check_alert_conditions and pass `bot` only if supported.
    """
    check_id = f"{alert_key}_{int(time.time())}"
    logger.debug("[%s] Starting check for alert.", check_id)
    try:
        # Determine whether check_alert_conditions accepts a 'bot' argument
        try:
            sig = inspect.signature(check_alert_conditions)
            accepts_bot = 'bot' in sig.parameters or 'context' in sig.parameters
        except Exception:
            accepts_bot = False

        if accepts_bot:
            # Call with bot/context if supported
            # try 'bot' first, then 'context'
            try:
                if 'bot' in sig.parameters:
                    result = await check_alert_conditions(alert_key, alert_details, bot=bot)
                else:
                    result = await check_alert_conditions(alert_key, alert_details, context=bot)
            except TypeError:
                # Fallback to original signature
                result = await check_alert_conditions(alert_key, alert_details)
        else:
            result = await check_alert_conditions(alert_key, alert_details)

        logger.debug("[%s] Alert check completed successfully.", check_id)
        return result
    except asyncio.TimeoutError:
        logger.warning("[%s] Timeout checking alert %s", check_id, alert_key)
        # Re-raise so caller (_sem_wrapped_check) knows about the timeout
        raise
    except Exception as e:
        logger.exception("[%s] Error checking alert %s: %s", check_id, alert_key, e)
        # Propagate the error to the caller to be logged in the batch
        raise


# --- action dispatcher + other handlers remain unchanged; keep your safe handler patterns there ---
async def action_dispatcher(update, context):
    cq = update.callback_query
    if cq is None:
        return
    # Answer callback quickly (handlers should use safe-answer pattern)
    try:
        await cq.answer()
    except Exception:
        # Swallow BadRequest or other answer errors to avoid crashing
        logger.debug("CallbackQuery.answer may have failed (stale/invalid).")

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
    # Dispatch without blocking (handlers themselves should answer early and background heavy work)
    await target(update, context)


def cleanup_executor():
    """Cleanup function to properly shutdown executor."""
    global executor
    if executor:
        executor.shutdown(wait=True)
        logger.info("Executor shutdown completed")


def main():
    init_db()
    app = Application.builder().token(BOT_TOKEN).build()

    # --- Add the job to check alerts periodically ---
    job_queue = app.job_queue
    # Run the check every 59 seconds. Start checking 10 seconds after the bot starts.
    job_queue.run_repeating(callback=check_all_alerts_periodically, interval=59, first=10)
    # --- End of job addition ---

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
    app.add_handler(CallbackQueryHandler(show_trio_alerts_handler, pattern="^show_trio_alerts$"))

    # group selection
    app.add_handler(CallbackQueryHandler(forex_group_select, pattern="^(dxy_eu_gu|dxy_chf_jpy|dxy_aud_nzd)$"))
    app.add_handler(CallbackQueryHandler(futures_group_select, pattern="^(spx_nq_ym|es_nq_dow|spx_dow_nq)$"))
    app.add_handler(CallbackQueryHandler(crypto_group_select, pattern="^(btc_eth_xrp|btc_eth_total|btc_xrp_doge)$"))
    app.add_handler(CallbackQueryHandler(metals_group_select, pattern="^(dxy_xau_xag_aud|xau_xag_aud|dxy_xau_aud)$"))

    # timeframe selection handlers
    app.add_handler(CallbackQueryHandler(forex_timeframe_select, pattern=r"^timeframe::(dxy_eu_gu|dxy_chf_jpy|dxy_aud_nzd)::"))
    app.add_handler(CallbackQueryHandler(futures_timeframe_select, pattern=r"^timeframe::(spx_nq_ym|es_nq_dow|spx_dow_nq)::"))
    app.add_handler(CallbackQueryHandler(crypto_timeframe_select, pattern=r"^timeframe::(btc_eth_xrp|btc_eth_total|btc_xrp_doge)::"))
    app.add_handler(CallbackQueryHandler(metals_timeframe_select, pattern=r"^timeframe::(dxy_xau_xag_aud|xau_xag_aud|dxy_xau_aud)::"))

    # single action dispatcher (charts/activate/deactivate)
    app.add_handler(CallbackQueryHandler(action_dispatcher, pattern=r"^(charts|activate|deactivate)::"))

    # Register cleanup on exit
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
