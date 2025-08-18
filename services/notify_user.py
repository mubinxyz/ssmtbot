# services/notify_user.py

import logging
import inspect
from typing import Optional, List, Dict, Any

from telegram import Bot
from telegram.constants import ParseMode
from config import BOT_TOKEN

logger = logging.getLogger(__name__)

TELEGRAM_MAX_MSG_LEN = 4096


def _is_valid_token(token: Optional[str]) -> bool:
    return bool(token and isinstance(token, str) and token.strip())


async def _maybe_await(callable_or_coroutine):
    """
    Helper that will await if the object is awaitable, otherwise just return it.
    This lets the code work with both sync and async client implementations.
    """
    if inspect.isawaitable(callable_or_coroutine):
        return await callable_or_coroutine
    return callable_or_coroutine


def _truncate_message(msg: str) -> str:
    if not isinstance(msg, str):
        msg = str(msg)
    if len(msg) <= TELEGRAM_MAX_MSG_LEN:
        return msg
    # Truncate with an indicator
    return msg[: TELEGRAM_MAX_MSG_LEN - 3] + "..."


async def notify_user(user_id: int, message: str, parse_mode: Optional[str] = None) -> bool:
    """
    Send a plain text notification to a Telegram user.

    Works with both synchronous and asynchronous `Bot.send_message` implementations.
    """
    try:
        if not _is_valid_token(BOT_TOKEN):
            logger.error("BOT_TOKEN is not configured or invalid.")
            return False

        bot = Bot(token=BOT_TOKEN)
        text = _truncate_message(message)

        # Some telegram client versions return a coroutine for send_message, some are sync.
        try:
            result = bot.send_message(chat_id=user_id, text=text, parse_mode=parse_mode)
            await _maybe_await(result)
        except TypeError:
            # Fallback (some clients expect named arguments differently)
            result = bot.send_message(user_id, text)
            await _maybe_await(result)

        logger.info("Successfully notified user %s", user_id)
        return True
    except Exception as e:
        logger.exception("Failed to notify user %s: %s", user_id, e)
        return False


async def notify_user_with_charts(user_id: int, message: str, chart_data_list: Optional[List[Dict[str, Any]]] = None) -> bool:
    """
    Send a message and optional chart information to a Telegram user.

    chart_data_list: list of dicts that may contain:
      - {'symbol': 'BTCUSD', 'timeframe': 15, 'generated': True}
      - {'error': 'error message'}
      - potentially other keys in future (e.g., URLs / buffers).
    """
    try:
        if not _is_valid_token(BOT_TOKEN):
            logger.error("BOT_TOKEN is not configured or invalid.")
            return False

        bot = Bot(token=BOT_TOKEN)
        text = _truncate_message(message)

        # Send main message (use MARKDOWN by default for formatting)
        main_send = bot.send_message(chat_id=user_id, text=text, parse_mode=ParseMode.MARKDOWN)
        await _maybe_await(main_send)

        # Send charts/info messages (simple textual placeholders for now)
        if chart_data_list:
            for chart_info in chart_data_list:
                try:
                    if isinstance(chart_info, dict) and 'error' in chart_info:
                        chart_msg = f"❌ Chart error: {chart_info['error']}"
                        send_res = bot.send_message(chat_id=user_id, text=_truncate_message(chart_msg))
                        await _maybe_await(send_res)
                    elif isinstance(chart_info, dict):
                        sym = chart_info.get("symbol", "Unknown")
                        tf = chart_info.get("timeframe", "N/A")
                        chart_msg = f"📊 Chart for {sym} ({tf}m)"
                        send_res = bot.send_message(chat_id=user_id, text=_truncate_message(chart_msg))
                        await _maybe_await(send_res)
                    else:
                        # Unknown chart_info format; stringify
                        send_res = bot.send_message(chat_id=user_id, text=_truncate_message(f"📊 Chart: {chart_info}"))
                        await _maybe_await(send_res)
                except Exception:
                    logger.exception("Failed to send chart info to user %s for chart %s", user_id, chart_info)
                    # continue sending remaining chart messages

        logger.info("Successfully notified user %s with charts info", user_id)
        return True
    except Exception as e:
        logger.exception("Failed to notify user %s with charts: %s", user_id, e)
        return False


def format_alert_message(message: str) -> str:
    """
    Format alert message for better presentation.
    Currently returns the message unchanged, but centralizes formatting
    so you can enhance it later (add bold headers, escaping, etc).
    """
    if not isinstance(message, str):
        message = str(message)
    # Potentially perform escaping or richer formatting here
    return message


async def send_alert_notification(user_id: int, alert_message: str, chart_data_list: Optional[List[Dict[str, Any]]] = None) -> bool:
    """
    High-level API used by the alert service to notify users.
    If chart_data_list is provided, send a message followed by chart info.
    Otherwise send a single formatted message.
    """
    try:
        if chart_data_list:
            return await notify_user_with_charts(user_id, alert_message, chart_data_list)
        else:
            formatted_message = format_alert_message(alert_message)
            # Use MARKDOWN parse mode for formatted alerts
            return await notify_user(user_id, formatted_message, ParseMode.MARKDOWN)
    except Exception as e:
        logger.exception("Error sending alert notification to user %s: %s", user_id, e)
        return False
