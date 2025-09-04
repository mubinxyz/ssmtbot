# -------------------------
# wsgi_bot.py
# -------------------------
import logging
import asyncio
import threading
import concurrent.futures
import atexit
from flask import Flask, request
from telegram import Update
from telegram.error import BadRequest
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, ContextTypes

# config: ensure you have WEBHOOK_URL in config.py (full https URL where Telegram should send updates)
try:
    from config import BOT_TOKEN, LOG_LEVEL, WEBHOOK_URL
except Exception:
    # fallbacks: LOG_LEVEL/BOT_TOKEN must exist; WEBHOOK_URL may be missing (we'll log a warning)
    from config import BOT_TOKEN, LOG_LEVEL
    WEBHOOK_URL = None

# db + handlers + services (import everything used by your original bot)
from services.db_service import init_db
from commands.start_command import start_command, back_to_main_handler

from handlers.forex_handler import (
    forex_menu_handler,
    group_select_handler as forex_group_select,
    timeframe_select_handler as forex_timeframe_select,
    charts_handler as forex_charts,
    activate_handler as forex_activate,
    deactivate_handler as forex_deactivate,
    GROUP_ID_SET as FOREX_GROUPS,
)
from handlers.futures_handler import (
    futures_menu_handler,
    group_select_handler as futures_group_select,
    timeframe_select_handler as futures_timeframe_select,
    charts_handler as futures_charts,
    activate_handler as futures_activate,
    deactivate_handler as futures_deactivate,
    GROUP_ID_SET as FUTURES_GROUPS,
)
from handlers.crypto_usd_handler import (
    crypto_usd_menu_handler,
    group_select_handler as crypto_group_select,
    timeframe_select_handler as crypto_timeframe_select,
    charts_handler as crypto_charts,
    activate_handler as crypto_activate,
    deactivate_handler as crypto_deactivate,
    GROUP_ID_SET as CRYPTO_GROUPS,
)
from handlers.metals_handler import (
    metals_menu_handler,
    group_select_handler as metals_group_select,
    timeframe_select_handler as metals_timeframe_select,
    charts_handler as metals_charts,
    activate_handler as metals_activate,
    deactivate_handler as metals_deactivate,
    GROUP_ID_SET as METALS_GROUPS,
)
from handlers.energy_handler import (
    energy_menu_handler,
    group_select_handler as energy_group_select,
    timeframe_select_handler as energy_timeframe_select,
    charts_handler as energy_charts,
    activate_handler as energy_activate,
    deactivate_handler as energy_deactivate,
    GROUP_ID_SET as ENERGY_GROUPS,
)
from handlers.alerts_handler import (
    alerts_menu_handler,
    delete_alert_handler,
    single_symbol_alerts_handler,
    show_alerts_handler,
)

import services.alert_service as alert_service

# ------------------ Logging ------------------
logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=LOG_LEVEL)
logger = logging.getLogger(__name__)

# Global executor for sync operations
executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

# Utilities
def safe_answer_callback(query):
    if query is None:
        return
    try:
        return query.answer()
    except BadRequest as e:
        logger.debug("Stale CallbackQuery.answer ignored: %s", e)
    except Exception as e:
        logger.exception("Unexpected error answering CallbackQuery: %s", e)


def safe_edit_message(edit_fn, *args, **kwargs):
    async def _inner():
        try:
            return await edit_fn(*args, **kwargs)
        except BadRequest as e:
            logger.debug("Stale edit_message ignored: %s", e)
        except Exception:
            logger.exception("Unexpected error while editing message")
    return _inner()


def safe_handler(fn):
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

# Periodic coordinator job (schedules the real coordinator as a background task)
async def check_all_alerts_periodically(context: ContextTypes.DEFAULT_TYPE):
    try:
        asyncio.create_task(alert_service.check_all_alerts_periodically_coordinator(bot=context.bot))
    except Exception as e:
        logger.exception("_periodic_job_callback: failed to schedule coordinator: %s", e)


# Action dispatcher copied/kept from your bot
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
    elif gid in ENERGY_GROUPS:
        if action == "charts":
            target = energy_charts
        elif action == "activate":
            target = energy_activate
        elif action == "deactivate":
            target = energy_deactivate

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


# Cleanup executor on exit
def cleanup_executor():
    global executor
    if executor:
        executor.shutdown(wait=True)
        logger.info("Executor shutdown completed")

atexit.register(cleanup_executor)

# ------------------ Build Telegram Application ------------------
init_db()
application = Application.builder().token(BOT_TOKEN).build()

# Register handlers (same as in your polling bot)
application.add_handler(CommandHandler("start", safe_handler(start_command)))
application.add_handler(CallbackQueryHandler(safe_handler(back_to_main_handler), pattern=r"^back_to_main$"))

# menus
application.add_handler(CallbackQueryHandler(safe_handler(forex_menu_handler), pattern=r"^menu_forex$"))
application.add_handler(CallbackQueryHandler(safe_handler(futures_menu_handler), pattern=r"^menu_futures$"))
application.add_handler(CallbackQueryHandler(safe_handler(crypto_usd_menu_handler), pattern=r"^menu_crypto_usd$"))
application.add_handler(CallbackQueryHandler(safe_handler(metals_menu_handler), pattern=r"^menu_metals$"))
application.add_handler(CallbackQueryHandler(safe_handler(alerts_menu_handler), pattern=r"^menu_alerts$"))
application.add_handler(CallbackQueryHandler(safe_handler(energy_menu_handler), pattern=r"^menu_energy$"))

# alerts handlers
application.add_handler(CallbackQueryHandler(safe_handler(show_alerts_handler), pattern=r"^show_alerts$"))
application.add_handler(CallbackQueryHandler(safe_handler(delete_alert_handler), pattern=r"^delete_alert::"))
application.add_handler(CallbackQueryHandler(safe_handler(single_symbol_alerts_handler), pattern=r"^single_symbol_alerts$"))

# group selection
application.add_handler(CallbackQueryHandler(safe_handler(forex_group_select), pattern=r"^(dxy_eu_gu_chf|dxy_chf_jpy|dxy_aud_nzd)$"))
application.add_handler(CallbackQueryHandler(safe_handler(futures_group_select), pattern=r"^(spx_nq_ym|es_nq_dow|spx_dow_nq)$"))
application.add_handler(CallbackQueryHandler(safe_handler(crypto_group_select), pattern=r"^(btc_eth_xrp|btc_eth_total|btc_xrp_doge)$"))
application.add_handler(CallbackQueryHandler(safe_handler(metals_group_select), pattern=r"^(dxy_xau_xag_aud|xau_xag_aud|dxy_xau_aud)$"))
application.add_handler(CallbackQueryHandler(safe_handler(energy_group_select), pattern=r"^(dxy_usdcad_owest_obrent)$"))

# timeframe selection
application.add_handler(CallbackQueryHandler(safe_handler(forex_timeframe_select), pattern=r"^timeframe::(dxy_eu_gu_chf|dxy_chf_jpy|dxy_aud_nzd)::"))
application.add_handler(CallbackQueryHandler(safe_handler(futures_timeframe_select), pattern=r"^timeframe::(spx_nq_ym|es_nq_dow|spx_dow_nq)::"))
application.add_handler(CallbackQueryHandler(safe_handler(crypto_timeframe_select), pattern=r"^timeframe::(btc_eth_xrp|btc_eth_total|btc_xrp_doge)::"))
application.add_handler(CallbackQueryHandler(safe_handler(metals_timeframe_select), pattern=r"^timeframe::(dxy_xau_xag_aud|xau_xag_aud|dxy_xau_aud)::"))
application.add_handler(CallbackQueryHandler(safe_handler(energy_timeframe_select), pattern=r"^timeframe::(dxy_usdcad_owest_obrent)::"))

# dispatcher for charts/activate/deactivate
application.add_handler(CallbackQueryHandler(safe_handler(action_dispatcher), pattern=r"^(charts|activate|deactivate)::"))

# ------------------ Flask app + webhook endpoint ------------------
flask_app = Flask(__name__)

@flask_app.post(f"/webhook/{BOT_TOKEN}")
def webhook():
    data = request.get_json(force=True)
    update = Update.de_json(data, application.bot)

    try:
        future = asyncio.run_coroutine_threadsafe(application.process_update(update), bot_loop)

        def _done_callback(f):
            try:
                f.result()
            except Exception:
                logger.exception("Exception while processing update in bot loop")
        future.add_done_callback(_done_callback)
    except Exception:
        logger.exception("Failed to schedule update on bot loop")
    return "ok"

# ------------------ Startup coroutine (runs inside bot loop) ------------------
async def _on_startup():
    # initialize/start the Application and job queue
    await application.initialize()
    await application.start()

    # Set webhook if provided
    if WEBHOOK_URL:
        try:
            await application.bot.set_webhook(WEBHOOK_URL)
            logger.info("Webhook set to %s", WEBHOOK_URL)
        except Exception:
            logger.exception("Failed to set webhook to %s", WEBHOOK_URL)
    else:
        logger.warning("WEBHOOK_URL not set in config.py — incoming Telegram updates will not be delivered by Telegram.\nYou can still POST updates on /webhook/<BOT_TOKEN> from your proxy or ngrok for testing.")

    # schedule background job on the bot's job queue
    application.job_queue.run_repeating(check_all_alerts_periodically, interval=15, first=5)
    logger.info("Background jobs scheduled.")

# ------------------ Run the bot's asyncio loop in a background thread ------------------
bot_loop = asyncio.new_event_loop()

def _start_bot_loop(loop: asyncio.AbstractEventLoop):
    asyncio.set_event_loop(loop)
    # run startup then keep loop running
    loop.run_until_complete(_on_startup())
    logger.info("Bot loop started and running forever")
    loop.run_forever()

bot_thread = threading.Thread(target=_start_bot_loop, args=(bot_loop,), daemon=True)
bot_thread.start()

# expose WSGI app for Passenger / Gunicorn
app = flask_app

# helpful: allow local run for testing
if __name__ == "__main__":
    # Local only: do not use this for production behind a real webhook — use a proper HTTPS fronting proxy
    flask_app.run(host="0.0.0.0", port=5000)



