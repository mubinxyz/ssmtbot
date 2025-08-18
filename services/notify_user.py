# services/notify_user.py
import logging
import inspect
import asyncio
from typing import Optional, List, Dict, Any

from telegram import Bot
from telegram.constants import ParseMode
from config import BOT_TOKEN

logger = logging.getLogger(__name__)

TELEGRAM_MAX_MSG_LEN = 4096

# Create a shared Bot instance lazily so we don't recreate HTTP clients all the time.
_SHARED_BOT: Optional[Bot] = None


def _is_valid_token(token: Optional[str]) -> bool:
    return bool(token and isinstance(token, str) and token.strip())


def _truncate_message(msg: str) -> str:
    if not isinstance(msg, str):
        msg = str(msg)
    if len(msg) <= TELEGRAM_MAX_MSG_LEN:
        return msg
    return msg[: TELEGRAM_MAX_MSG_LEN - 3] + "..."


def _get_bot(provided_bot: Optional[Bot] = None) -> Bot:
    """
    Return a Bot to use: prefer provided_bot, fallback to a shared Bot created from BOT_TOKEN.
    """
    global _SHARED_BOT
    if provided_bot is not None:
        return provided_bot
    if not _SHARED_BOT:
        if not _is_valid_token(BOT_TOKEN):
            raise RuntimeError("BOT_TOKEN not configured")
        _SHARED_BOT = Bot(token=BOT_TOKEN)
    return _SHARED_BOT


async def _maybe_send(bot: Bot, /, **send_kwargs) -> Any:
    """
    Send using bot.send_message but do so safely:
      - If send_message is coroutine function (async client), await it.
      - If send_message is synchronous, run it in a thread to avoid blocking the event loop.
    Returns whatever the underlying send returns (or the coroutine's result).
    """
    send_fn = getattr(bot, "send_message")
    # If the method itself is implemented as a coroutine function, call/await it.
    if inspect.iscoroutinefunction(send_fn):
        return await send_fn(**send_kwargs)
    else:
        # run blocking call in a worker thread
        return await asyncio.to_thread(lambda: send_fn(**send_kwargs))


async def notify_user(user_id: int, message: str, parse_mode: Optional[str] = None, bot: Optional[Bot] = None) -> bool:
    """
    Send a plain text notification to a Telegram user.

    Args:
        user_id (int): Telegram chat id
        message (str): message text
        parse_mode (str|None): optional parse mode constant
        bot (Bot|None): optional Bot instance to use (recommended: pass context.bot)

    Returns:
        bool: True if successful
    """
    try:
        if not _is_valid_token(BOT_TOKEN) and bot is None:
            logger.error("BOT_TOKEN is not configured or invalid and no bot provided.")
            return False

        bot_to_use = _get_bot(bot)
        text = _truncate_message(message)

        # Use the helper that will await or push into thread as needed
        await _maybe_send(bot_to_use, chat_id=user_id, text=text, parse_mode=parse_mode)
        logger.info("Successfully notified user %s", user_id)
        return True
    except Exception as e:
        logger.exception("Failed to notify user %s: %s", user_id, e)
        return False


async def notify_user_with_charts(
    user_id: int,
    message: str,
    chart_data_list: Optional[List[Dict[str, Any]]] = None,
    bot: Optional[Bot] = None
) -> bool:
    """
    Send a message and optional chart information to a Telegram user.
    Each chart info is sent as a small follow-up message (placeholder text).
    """
    try:
        if not _is_valid_token(BOT_TOKEN) and bot is None:
            logger.error("BOT_TOKEN is not configured or invalid and no bot provided.")
            return False

        bot_to_use = _get_bot(bot)
        text = _truncate_message(message)

        # Send main message (use MARKDOWN for formatting)
        await _maybe_send(bot_to_use, chat_id=user_id, text=text, parse_mode=ParseMode.MARKDOWN)

        # Send charts/info messages (text placeholders; real images would be separate logic)
        if chart_data_list:
            for chart_info in chart_data_list:
                try:
                    if isinstance(chart_info, dict) and "error" in chart_info:
                        chart_msg = f"❌ Chart error: {chart_info['error']}"
                        await _maybe_send(bot_to_use, chat_id=user_id, text=_truncate_message(chart_msg))
                    elif isinstance(chart_info, dict):
                        sym = chart_info.get("symbol", "Unknown")
                        tf = chart_info.get("timeframe", "N/A")
                        chart_msg = f"📊 Chart for {sym} ({tf}m)"
                        await _maybe_send(bot_to_use, chat_id=user_id, text=_truncate_message(chart_msg))
                    else:
                        await _maybe_send(bot_to_use, chat_id=user_id, text=_truncate_message(f"📊 Chart: {chart_info}"))
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
    Format alert message for presentation.
    """
    if not isinstance(message, str):
        message = str(message)
    # leave room for escaping/formatting later
    return message


async def send_alert_notification(
    user_id: int,
    alert_message: str,
    chart_data_list: Optional[List[Dict[str, Any]]] = None,
    bot: Optional[Bot] = None
) -> bool:
    """
    High-level API used by the alert service to notify users.
    Accepts optional `bot` so callers can pass `context.bot` or the app's bot instance.
    """
    try:
        if chart_data_list:
            return await notify_user_with_charts(user_id, alert_message, chart_data_list, bot=bot)
        else:
            formatted_message = format_alert_message(alert_message)
            return await notify_user(user_id, formatted_message, ParseMode.MARKDOWN, bot=bot)
    except Exception as e:
        logger.exception("Error sending alert notification to user %s: %s", user_id, e)
        return False
