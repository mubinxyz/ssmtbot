# handlers/forex_handler.py
import logging
from contextlib import suppress
from typing import List

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import ContextTypes

import services.alert_service as alert_service
import services.chart_service as chart_service

logger = logging.getLogger(__name__)

GROUP_ID_SET = {
    "dxy_eu_gu_chf",
    "dxy_aud_nzd",
}

FALLBACK_SYMBOLS = {
    "dxy_eu_gu_chf": ["USDX", "EURUSD", "GBPUSD", "USDCHF"],
    "dxy_aud_nzd": ["USDX", "AUD", "NZD"],
}

# timeframe choices (minutes). 1440 == 1 day
TIMEFRAMES = [5, 15, 60, 240]


def _is_alert_active(user_id: int, gid: str, tf: int) -> bool:
    """
    Return True if an active alert exists for this (user, group_id, timeframe).
    Uses alert_service.get_active_alerts() which returns only active alerts.
    """
    try:
        active_alerts = alert_service.get_active_alerts() or {}
        for a in active_alerts.values():
            try:
                if int(a.get("user_id")) == int(user_id) and a.get("group_id") == gid and int(a.get("timeframe_min", -1)) == int(tf):
                    return True
            except Exception:
                # if any cast fails, skip this alert entry
                continue
    except Exception:
        logger.debug("_is_alert_active: error while checking alerts", exc_info=True)
    return False


def _menu_kb() -> InlineKeyboardMarkup:
    buttons = [
        [InlineKeyboardButton("DXY / EURUSD / GBPUSD / CHFUSD", callback_data="dxy_eu_gu_chf")],
        # [InlineKeyboardButton("DXY / CHF / JPY", callback_data="dxy_chf_jpy")],
        [InlineKeyboardButton("DXY / AUD / NZD", callback_data="dxy_aud_nzd")],
        [InlineKeyboardButton("🔙 Back", callback_data="back_to_main")],
    ]
    return InlineKeyboardMarkup(buttons)


def _timeframe_kb(gid: str, user_id: int) -> InlineKeyboardMarkup:
    """
    Build timeframe keyboard and append a small alert symbol inline:
      - 🔔 = active alert for this (user, group, timeframe)
      - 🔕 = inactive
    """
    rows = []
    row = []
    for tf in TIMEFRAMES:
        # Show a filled bell when active, muted bell when inactive
        icon = "🔔" if _is_alert_active(user_id, gid, tf) else "🔕"
        label = f"{tf}m {icon}" if tf < 1440 else f"1d {icon}"
        row.append(InlineKeyboardButton(label, callback_data=f"timeframe::{gid}::{tf}"))
        if len(row) == 2:
            rows.append(row)
            row = []
    if row:
        rows.append(row)
    rows.append([InlineKeyboardButton("⬅️ Back", callback_data="menu_forex")])
    return InlineKeyboardMarkup(rows)


def _action_kb(label: str, gid: str, tf_default: int = 15) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(f"View charts ({tf_default}m) - {label}", callback_data=f"charts::{gid}::{tf_default}")],
        [InlineKeyboardButton(f"Activate alert ({tf_default}m)", callback_data=f"activate::{gid}::{tf_default}")],
        [InlineKeyboardButton(f"Deactivate alert ({tf_default}m)", callback_data=f"deactivate::{gid}::{tf_default}")],
        [InlineKeyboardButton("⬅️ Back", callback_data="menu_forex")],
    ])


def _resolve_label_and_symbols(gid: str):
    try:
        cat, canonical = alert_service._find_group_category_by_group_id_flexible(gid)
        if cat and canonical:
            gd = alert_service.find_group(cat, canonical)
            if gd:
                return gd.get("label", canonical), gd.get("symbols", [])
    except Exception:
        logger.debug("could not resolve group %s in alert_service", gid, exc_info=True)
    return gid.replace("_", " ").upper(), FALLBACK_SYMBOLS.get(gid, [])


# Handlers

async def forex_menu_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    await q.edit_message_text("Forex groups:", reply_markup=_menu_kb())


async def group_select_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    gid = q.data
    if gid not in GROUP_ID_SET:
        await q.edit_message_text("Unsupported group.")
        return
    label, _ = _resolve_label_and_symbols(gid)
    user_id = update.effective_user.id
    await q.edit_message_text(f"{label}\nChoose a timeframe:", reply_markup=_timeframe_kb(gid, user_id))


async def timeframe_select_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    parts = (q.data or "").split("::")
    if len(parts) != 3:
        await q.edit_message_text("Invalid timeframe selection.")
        return
    _, gid, tf_s = parts
    if gid not in GROUP_ID_SET:
        await q.edit_message_text("Unsupported group.")
        return
    try:
        tf = int(tf_s)
    except Exception:
        await q.edit_message_text("Invalid timeframe.")
        return
    label, _ = _resolve_label_and_symbols(gid)
    user_id = update.effective_user.id
    is_active = _is_alert_active(user_id, gid, tf)
    status_text = "ACTIVE" if is_active else "INACTIVE"
    await q.edit_message_text(f"{label} — {tf}min\nStatus: {status_text}\nChoose an action:", reply_markup=_action_kb(label, gid, tf_default=tf))


async def charts_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    parts = q.data.split("::")
    if len(parts) != 3:
        await q.edit_message_text("Invalid charts command.")
        return
    _, gid, tf_s = parts
    if gid not in GROUP_ID_SET:
        return
    try:
        tf = int(tf_s)
    except Exception:
        await q.edit_message_text("Invalid timeframe.")
        return

    label, symbols = _resolve_label_and_symbols(gid)
    if not symbols:
        symbols = FALLBACK_SYMBOLS.get(gid, [])

    await q.edit_message_text(f"Generating {tf}min charts for {label} ...")
    try:
        bufs: List = await chart_service.generate_chart(symbols, timeframe=tf)
        if not bufs:
            await q.edit_message_text("No charts available.")
            return
        chat_id = q.message.chat_id
        for buf in bufs:
            buf.seek(0)
            await context.bot.send_photo(chat_id=chat_id, photo=buf)
        with suppress(Exception):
            await q.edit_message_text(f"Charts for {label} sent.")
    except Exception as e:
        logger.exception("charts failed for %s: %s", gid, e)
        await q.edit_message_text(f"Failed to generate charts: {e}")


async def activate_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    parts = q.data.split("::")
    if len(parts) != 3:
        await q.edit_message_text("Invalid activate command.")
        return
    _, gid, tf_s = parts
    if gid not in GROUP_ID_SET:
        return
    try:
        tf = int(tf_s)
    except Exception:
        await q.edit_message_text("Invalid timeframe.")
        return
    user_id = update.effective_user.id
    try:
        alert_service.set_ssmt_alert(user_id=user_id, group_id=gid, timeframe_min=tf)
        # include a back button that returns to timeframe selection for this gid
        back_kb = InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Back", callback_data=gid)]])
        await q.edit_message_text(f"✅ Alert activated for {gid} ({tf}min).", reply_markup=back_kb)
    except Exception as e:
        logger.exception("activate failed: %s", e)
        back_kb = InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Back", callback_data=gid)]])
        await q.edit_message_text(f"Failed to activate alert: {e}", reply_markup=back_kb)


async def deactivate_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    parts = q.data.split("::")
    if len(parts) != 3:
        await q.edit_message_text("Invalid deactivate command.")
        return
    _, gid, tf_s = parts
    if gid not in GROUP_ID_SET:
        return
    try:
        tf = int(tf_s)
    except Exception:
        await q.edit_message_text("Invalid timeframe.")
        return
    user_id = update.effective_user.id
    ok = alert_service.deactivate_ssmt_alert(user_id=user_id, group_id=gid, timeframe_min=tf)
    back_kb = InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Back", callback_data=gid)]])
    if ok:
        await q.edit_message_text(f"⛔ Alert deactivated for {gid} ({tf}min).", reply_markup=back_kb)
    else:
        await q.edit_message_text("No active alert found to deactivate.", reply_markup=back_kb)
