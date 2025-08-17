# handlers/metals_handler.py
import logging
from contextlib import suppress
from typing import List

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import ContextTypes

import services.alert_service as alert_service
import services.chart_service as chart_service

logger = logging.getLogger(__name__)

GROUP_ID_SET = {
    "dxy_xau_xag_aud",
    "xau_xag_aud",
    "dxy_xau_aud",
}

# Use LiteFinance-friendly symbols (we normalize elsewhere)
FALLBACK_SYMBOLS = {
    "dxy_xau_xag_aud": ["USDX", "XAU", "XAG", "AUD"],
    "xau_xag_aud": ["XAU", "XAG", "AUD"],
    "dxy_xau_aud": ["USDX", "XAU", "AUD"],
}

# Which timeframes the user can choose from (minutes)
TIMEFRAMES = [5, 15, 60, 240]

def _menu_kb() -> InlineKeyboardMarkup:
    buttons = [
        [InlineKeyboardButton("DXY / XAU / XAG / AUD", callback_data="dxy_xau_xag_aud")],
        [InlineKeyboardButton("XAU / XAG / AUD", callback_data="xau_xag_aud")],
        [InlineKeyboardButton("DXY / XAU / AUD", callback_data="dxy_xau_aud")],
        [InlineKeyboardButton("🔙 Back", callback_data="back_to_main")],
    ]
    return InlineKeyboardMarkup(buttons)


def _timeframe_kb(gid: str) -> InlineKeyboardMarkup:
    """
    Build a keyboard that lets the user pick a timeframe for the selected group.
    Callback: timeframe::{gid}::{tf}
    """
    rows = []
    # make 2 buttons per row for nicer layout
    row = []
    for i, tf in enumerate(TIMEFRAMES):
        row.append(InlineKeyboardButton(f"{tf}m" if tf < 1440 else "1d", callback_data=f"timeframe::{gid}::{tf}"))
        if len(row) == 2:
            rows.append(row)
            row = []
    if row:
        rows.append(row)
    # navigation
    rows.append([InlineKeyboardButton("⬅️ Back", callback_data="menu_metals")])
    return InlineKeyboardMarkup(rows)


def _action_kb(label: str, gid: str, tf_default: int = 15) -> InlineKeyboardMarkup:
    """
    After timeframe selected, show actions for that gid+timeframe.
    Callback patterns:
      charts::{gid}::{tf}
      activate::{gid}::{tf}
      deactivate::{gid}::{tf}
    """
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(f"View charts ({tf_default}m) - {label}", callback_data=f"charts::{gid}::{tf_default}")],
        [InlineKeyboardButton(f"Activate alert ({tf_default}m)", callback_data=f"activate::{gid}::{tf_default}")],
        [InlineKeyboardButton(f"Deactivate alert ({tf_default}m)", callback_data=f"deactivate::{gid}::{tf_default}")],
        [InlineKeyboardButton("⬅️ Back", callback_data="menu_metals")],
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
async def metals_menu_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    await q.edit_message_text("Metals groups:", reply_markup=_menu_kb())


async def group_select_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    User clicked a group. Instead of showing actions immediately, show timeframe picker first.
    """
    q = update.callback_query
    await q.answer()
    gid = q.data
    if gid not in GROUP_ID_SET:
        await q.edit_message_text("Unsupported group.")
        return

    label, _ = _resolve_label_and_symbols(gid)
    # Show timeframe options first
    await q.edit_message_text(f"{label}\nChoose a timeframe:", reply_markup=_timeframe_kb(gid))


async def timeframe_select_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Called when user picks a timeframe.
    Callback format: timeframe::{gid}::{tf}
    Show the action keyboard for that gid+tf.
    """
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
    await q.edit_message_text(f"{label} — {tf}min\nChoose an action:", reply_markup=_action_kb(label, gid, tf_default=tf))


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
        await q.edit_message_text(f"✅ Alert activated for {gid} ({tf}min).")
    except Exception as e:
        logger.exception("activate failed: %s", e)
        await q.edit_message_text(f"Failed to activate alert: {e}")


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
    if ok:
        await q.edit_message_text(f"⛔ Alert deactivated for {gid} ({tf}min).")
    else:
        await q.edit_message_text("No active alert found to deactivate.")
