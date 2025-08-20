# services/notify_user.py
import logging
import inspect
import asyncio
import io
from typing import Optional, List, Dict, Any, Iterable, Union
from telegram import Bot, InputMediaPhoto
from telegram.constants import ParseMode
from config import BOT_TOKEN
from services.chart_service import generate_chart

logger = logging.getLogger(__name__)

TELEGRAM_MAX_MSG_LEN = 4096

# Lazily created shared Bot instance (fallback if caller doesn't pass context.bot)
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


async def _maybe_send(callable_or_method, /, *args, **kwargs):
    """
    Call and await if coroutine; otherwise run in thread. Works for Bot methods or other callables.
    """
    try:
        if inspect.iscoroutinefunction(callable_or_method):
            return await callable_or_method(*args, **kwargs)
        result = callable_or_method(*args, **kwargs)
        if inspect.isawaitable(result):
            return await result
        return await asyncio.to_thread(lambda: result)
    except TypeError:
        # Fallback to running in thread for some edge-case callables
        return await asyncio.to_thread(lambda: callable_or_method(*args, **kwargs))


def _escape_for_telegram_markdown_v1(text: str) -> str:
    """
    Escape text for Telegram Markdown (v1). Escape backslash first, then the characters
    that can create entities in Markdown v1: _ * ` [ ]
    This keeps messages safe when using ParseMode.MARKDOWN.
    """
    if not isinstance(text, str):
        text = str(text)
    # Escape backslash first
    text = text.replace('\\', '\\\\')
    for ch in ['_', '*', '`', '[', ']']:
        text = text.replace(ch, '\\' + ch)
    return text


async def notify_user(user_id: int, message: str, parse_mode: Optional[str] = None, bot: Optional[Bot] = None) -> bool:
    """
    Send a plain text notification to a Telegram user.
    If using ParseMode.MARKDOWN, message will be safely escaped for Markdown v1.
    """
    try:
        if not _is_valid_token(BOT_TOKEN) and bot is None:
            logger.error("BOT_TOKEN is not configured or invalid and no bot provided.")
            return False

        bot_to_use = _get_bot(bot)
        text = _truncate_message(message)

        # If caller intends Markdown (v1), escape the message to prevent parse errors
        if parse_mode == ParseMode.MARKDOWN:
            try:
                text = _escape_for_telegram_markdown_v1(text)
            except Exception:
                logger.exception("Markdown escaping failed; sending raw truncated message instead.")
                text = _truncate_message(message)

        send_fn = getattr(bot_to_use, "send_message")
        await _maybe_send(send_fn, chat_id=user_id, text=text, parse_mode=parse_mode)
        logger.info("Successfully notified user %s", user_id)
        return True
    except Exception as e:
        logger.exception("Failed to notify user %s: %s", user_id, e)
        return False


async def _send_media_group_nonblocking(bot: Bot, chat_id: int, media: List[InputMediaPhoto]):
    """
    Helper to call send_media_group safely (await or run in thread).
    """
    send_fn = getattr(bot, "send_media_group")
    return await _maybe_send(send_fn, chat_id=chat_id, media=media)


def _is_file_like(obj) -> bool:
    return hasattr(obj, "read") and hasattr(obj, "seek")


async def notify_user_with_charts(
    user_id: int,
    message: str,
    chart_data_list: Optional[List[Dict[str, Any]]] = None,
    bot: Optional[Bot] = None
) -> bool:
    """
    Send a message and optional chart images to a Telegram user.

    chart_data_list entries may be:
      - file-like objects (io.BytesIO, open file) or raw bytes/bytearray
      - dicts containing {"buffer": <file-like or bytes>} or {"symbol": "...", "timeframe": 15}
      - dicts containing {"error": "reason"} to be reported as text
    """
    try:
        if not _is_valid_token(BOT_TOKEN) and bot is None:
            logger.error("BOT_TOKEN is not configured or invalid and no bot provided.")
            return False

        bot_to_use = _get_bot(bot)
        text = _truncate_message(message)

        # 1) send main message first (with Markdown escaping to avoid parse errors)
        try:
            send_msg_fn = getattr(bot_to_use, "send_message")
            safe_text = _escape_for_telegram_markdown_v1(text)
            await _maybe_send(send_msg_fn, chat_id=user_id, text=safe_text, parse_mode=ParseMode.MARKDOWN)
        except Exception:
            # don't abort on message failure; continue to attempt charts
            logger.exception("Failed to send main alert message to user %s", user_id)

        if not chart_data_list:
            logger.info("No chart data to send for user %s", user_id)
            return True

        # Normalize and collect buffers
        buffers_to_send: List[io.BytesIO] = []
        errors_to_send: List[str] = []
        symbol_groups: Dict[int, List[str]] = {}  # timeframe -> list of symbols

        def _ensure_bytesio(obj: Union[bytes, bytearray, io.BytesIO, Any]) -> Optional[io.BytesIO]:
            # Convert bytes/bytearray to BytesIO, accept BytesIO directly, or attempt to read file-like
            if isinstance(obj, (bytes, bytearray)):
                return io.BytesIO(bytes(obj))
            if _is_file_like(obj):
                try:
                    obj.seek(0)
                except Exception:
                    pass
                # If it's already a BytesIO we can use as-is; otherwise read into BytesIO to be safe
                if isinstance(obj, io.BytesIO):
                    return obj
                try:
                    data = obj.read()
                    return io.BytesIO(data if data is not None else b"")
                except Exception:
                    return None
            return None

        # First pass: extract any pre-generated buffers and collect symbol/timeframe requests
        for entry in chart_data_list:
            if entry is None:
                continue
            # raw bytes / bytearray
            if isinstance(entry, (bytes, bytearray)):
                b = _ensure_bytesio(entry)
                if b:
                    buffers_to_send.append(b)
                else:
                    errors_to_send.append("Invalid bytes chart buffer")
                continue

            # file-like directly
            if _is_file_like(entry):
                b = _ensure_bytesio(entry)
                if b:
                    buffers_to_send.append(b)
                else:
                    errors_to_send.append("Invalid file-like chart buffer")
                continue

            # dict-case
            if isinstance(entry, dict):
                # explicit buffer field
                if "buffer" in entry:
                    buf = entry["buffer"]
                    b = _ensure_bytesio(buf)
                    if b:
                        buffers_to_send.append(b)
                    else:
                        errors_to_send.append(f"Invalid buffer for entry {entry.get('symbol') or 'unknown'}")
                    continue
                # explicit error
                if "error" in entry:
                    errors_to_send.append(str(entry["error"]))
                    continue
                # symbol/timeframe -> queue for generation
                sym = entry.get("symbol")
                tf = entry.get("timeframe")
                if sym and tf:
                    try:
                        tf_key = int(tf)
                    except Exception:
                        try:
                            tf_key = int(tf) if isinstance(tf, int) else 0
                        except Exception:
                            tf_key = 0
                    symbol_groups.setdefault(tf_key, []).append(sym)
                    continue
                # fallback: stringify entry
                errors_to_send.append(str(entry))
                continue

            # anything else -> stringify and add as error
            errors_to_send.append(str(entry))

        # Generate charts for grouped timeframes
        for tf, syms in symbol_groups.items():
            try:
                if inspect.iscoroutinefunction(generate_chart):
                    chart_buffers = await generate_chart(syms, timeframe=tf)
                else:
                    chart_buffers = await asyncio.to_thread(generate_chart, syms, tf)

                # chart_buffers expected to be iterable
                for b in chart_buffers:
                    # accept bytes, bytearray, or file-like
                    converted = _ensure_bytesio(b)
                    if converted:
                        buffers_to_send.append(converted)
                    else:
                        errors_to_send.append(f"Chart generator returned non-buffer for tf={tf}, syms={syms}")
            except Exception as e:
                logger.exception("Chart generation failed for timeframe %s symbols %s: %s", tf, syms, e)
                errors_to_send.append(f"Chart generation failed for {syms} @ {tf}m: {e}")

        # Send error messages (if any) as plain text so user knows
        for err in errors_to_send:
            try:
                await _maybe_send(getattr(bot_to_use, "send_message"), chat_id=user_id, text=_truncate_message(f"❌ {err}"))
            except Exception:
                logger.exception("Failed to send chart error message to user %s: %s", user_id, err)

        if not buffers_to_send:
            logger.info("No chart buffers to send for user %s", user_id)
            return True

        # Telegram send_media_group limit: max 10 items per request
        MAX_MEDIA = 10
        batches: List[List[io.BytesIO]] = [buffers_to_send[i:i + MAX_MEDIA] for i in range(0, len(buffers_to_send), MAX_MEDIA)]

        for batch in batches:
            media_items: List[InputMediaPhoto] = []
            for buf in batch:
                try:
                    buf.seek(0)
                except Exception:
                    pass
                # InputMediaPhoto accepts file-like objects; use the BytesIO directly
                media_items.append(InputMediaPhoto(media=buf))

            # TRY: attempt send_media_group, with a short retry on transient failure
            try:
                await _send_media_group_nonblocking(bot_to_use, chat_id=user_id, media=media_items)
            except Exception as e:
                # Log a concise warning (no huge stack trace) and attempt one quick retry
                logger.warning("send_media_group failed for user %s (will retry once then fallback): %s", user_id, str(e))
                # short backoff
                try:
                    await asyncio.sleep(0.5)
                    await _send_media_group_nonblocking(bot_to_use, chat_id=user_id, media=media_items)
                except Exception as e2:
                    # Final fallback: send images one-by-one. Log concise message (no full traceback).
                    logger.warning("send_media_group retry failed for user %s; falling back to send_photo per image: %s", user_id, str(e2))
                    for buf in batch:
                        try:
                            await _maybe_send(getattr(bot_to_use, "send_photo"), chat_id=user_id, photo=buf)
                        except Exception:
                            logger.exception("Failed to send individual chart photo to user %s", user_id)
                            continue

        logger.info("Successfully notified user %s with %d chart(s)", user_id, len(buffers_to_send))
        return True

    except Exception as e:
        logger.exception("Failed to notify user %s with charts: %s", user_id, e)
        return False


def format_alert_message(message: str) -> str:
    """
    Format alert message for presentation (escape for Markdown v1 by default).
    """
    if not isinstance(message, str):
        message = str(message)
    try:
        return _escape_for_telegram_markdown_v1(message)
    except Exception:
        return message


async def send_alert_notification(
    user_id: int,
    alert_message: str,
    chart_data_list: Optional[List[Dict[str, Any]]] = None,
    bot: Optional[Bot] = None
) -> bool:
    """
    High-level API used by the alert service to notify users.
    If chart_data_list provided, tries to send charts after the main message.
    """
    try:
        if chart_data_list:
            # notify_user_with_charts will escape the message for Markdown (safe)
            return await notify_user_with_charts(user_id, alert_message, chart_data_list, bot=bot)
        else:
            formatted_message = format_alert_message(alert_message)
            # notify_user expects parse_mode ParseMode.MARKDOWN to be passed so Telegram uses Markdown,
            # but we've already escaped so the content won't break entity parsing.
            return await notify_user(user_id, formatted_message, ParseMode.MARKDOWN, bot=bot)
    except Exception as e:
        logger.exception("Error sending alert notification to user %s: %s", user_id, e)
        return False
