# bot.py
import asyncio
import logging
import concurrent.futures
import time # Import for timing
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
    show_trio_alerts_handler,  # Added this import
)
# --- Import for alert checking ---
from services.alert_service import get_active_alerts, check_alert_conditions

# logging (ensure you have a logger set up)
logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=LOG_LEVEL)
logger = logging.getLogger(__name__)

# Global executor for any sync operations
executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

# ... (other handler definitions remain the same) ...

# --- Job Queue Function ---
async def check_all_alerts_periodically(context):
    """Function to be called by the job queue to check all active alerts."""
    start_time = time.monotonic()
    logger.info("Running periodic alert check...")
    try:
        # Get active alerts in a thread if this operation might block
        loop = asyncio.get_event_loop()
        active_alerts = await loop.run_in_executor(executor, get_active_alerts)
        if not active_alerts:
            logger.info("No active alerts to check.")
            return

        # Create tasks for all async alert checks
        # --- FIX: Create Task objects explicitly for asyncio.wait ---
        tasks = [
            asyncio.create_task(_check_single_alert_safe(alert_key, alert_details))
            for alert_key, alert_details in active_alerts.items()
        ]
        # --- End of FIX ---

        # --- Apply a GLOBAL timeout to the entire batch of checks ---
        # This prevents the whole job from hanging indefinitely
        GLOBAL_BATCH_TIMEOUT = 120.0 # e.g., 2 minutes for the whole batch
        try:
            # Run all tasks with timeout to prevent hanging
            # Use asyncio.wait instead of gather for better timeout control on the whole batch
            done, pending = await asyncio.wait(
                tasks,
                timeout=GLOBAL_BATCH_TIMEOUT,
                return_when=asyncio.ALL_COMPLETED # Default, but explicit
            )

            # Cancel any remaining pending tasks (those that timed out)
            if pending:
                logger.warning(f"Global timeout ({GLOBAL_BATCH_TIMEOUT}s) reached. Cancelling {len(pending)} pending alert checks.")
                for task in pending:
                    task.cancel()
                # Wait a short time for cancellations to be processed
                await asyncio.gather(*pending, return_exceptions=True)

            # Process results from completed tasks
            results = []
            for task in done:
                 try:
                     # This will raise the exception if the task failed or was cancelled
                     result = await task
                     results.append(result)
                 except asyncio.CancelledError:
                     results.append(asyncio.CancelledError("Task was cancelled due to global timeout"))
                 except Exception as e:
                     results.append(e) # Append the actual exception

            # Log any exceptions that occurred (including timeouts and cancellations)
            for i, (result, (alert_key, _)) in enumerate(zip(results, active_alerts.items())): # Pair results with alert keys
                if isinstance(result, Exception):
                    if isinstance(result, asyncio.TimeoutError):
                        logger.error(f"Timeout (individual) checking alert {alert_key}: {result}")
                    elif isinstance(result, asyncio.CancelledError):
                         logger.warning(f"Alert check for {alert_key} was cancelled (likely due to global timeout).")
                    else:
                        logger.error(f"Error checking alert {alert_key}: {result}")

            end_time = time.monotonic()
            duration = end_time - start_time
            logger.info(f"Finished periodic alert check. Processed {len(active_alerts)} alerts in {duration:.2f} seconds.")

        except Exception as global_wait_error: # Catch unexpected errors in the wait() itself
             logger.error(f"Unexpected error during asyncio.wait in alert checker: {global_wait_error}")
             # Cancel any tasks that might have been started if wait() failed unexpectedly
             if 'tasks' in locals():
                 for task in tasks:
                     if not task.done(): # Check if task exists and is not done before cancelling
                         task.cancel()
                 await asyncio.gather(*tasks, return_exceptions=True) # Wait for cancellations

    except Exception as e:
        logger.error(f"Unexpected error at the start of alert checker: {e}")


async def _check_single_alert_safe(alert_key, alert_details):
    """Safely check a single alert with timeout."""
    # Add a unique identifier for logging this specific check instance if needed
    check_id = f"{alert_key}_{int(time.time())}"
    logger.debug(f"[{check_id}] Starting check for alert.")
    try:
        # Add timeout to prevent hanging on individual alerts
        # Ensure the timeout is applied directly to the check function call
        result = await asyncio.wait_for(
            check_alert_conditions(alert_key, alert_details),
            timeout=30.0  # 30 second timeout per alert
        )
        logger.debug(f"[{check_id}] Alert check completed successfully.")
        return result # Return the result (True/False, message, chart_data) from check_alert_conditions
    except asyncio.TimeoutError:
        logger.warning(f"[{check_id}] Timeout (30s) checking alert {alert_key}")
        # Re-raise the TimeoutError so the calling code (gather/wait) knows it happened
        raise
    except Exception as e:
        logger.error(f"[{check_id}] Error checking alert {alert_key}: {e}")
        # Re-raise other exceptions so the calling code knows it happened
        raise
    # Note: No 'finally' block that awaits here, as that could also hang.

# ... (action_dispatcher, cleanup_executor, main remain largely the same) ...

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


def cleanup_executor():
    """Cleanup function to properly shutdown executor."""
    global executor
    if executor:
        executor.shutdown(wait=True)
        logger.info("Executor shutdown completed")

def main():
    init_db()  # Create tables if not exist
    app = Application.builder().token(BOT_TOKEN).build()
    
    # --- Add the job to check alerts periodically ---
    # This adds a job that runs every 59 seconds
    job_queue = app.job_queue
    # Run the check every 59 seconds. Start checking 10 seconds after the bot starts.
    job_queue.run_repeating(
        callback=check_all_alerts_periodically,
        interval=59,  # seconds
        first=10      # Start after 10 seconds
    )
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
    app.add_handler(CallbackQueryHandler(show_trio_alerts_handler, pattern="^show_trio_alerts$"))  # Added this handler
    
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
    
    # Register cleanup
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