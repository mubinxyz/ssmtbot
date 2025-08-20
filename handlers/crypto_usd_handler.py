# handlers/crypto_usd_handler.py
import logging
from contextlib import suppress
from typing import List

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import ContextTypes

import services.alert_service as alert_service
import services.chart_service as chart_service

logger = logging.getLogger(__name__)

GROUP_ID_SET = {
    "btc_eth_xrp",
    "btc_eth_total",
    "btc_xrp_doge",
}

FALLBACK_SYMBOLS = {
    "btc_eth_xrp": ["BTCUSD", "ETHUSD", "XRPUSD"],
    # "btc_eth_total": ["BTCUSD", "ETHUSD", "TOTAL"],
    # "btc_xrp_doge": ["BTCUSD", "XRPUSD", "DOGEUSD"],
}

# timeframe choices (in minutes). 1440 == 1 day
TIMEFRAMES = [5, 15, 60, 240]

def _menu_kb() -> InlineKeyboardMarkup:
    buttons = [
        [InlineKeyboardButton("BTCUSD / ETHUSD / XRPUSD", callback_data="btc_eth_xrp")],
        # [InlineKeyboardButton("BTCUSD / ETHUSD / TOTAL", callback_data="btc_eth_total")],
        # [InlineKeyboardButton("BTCUSD / XRPUSD / DOGEUSD", callback_data="btc_xrp_doge")],
        [InlineKeyboardButton("🔙 Back", callback_data="back_to_main")],
    ]
    return InlineKeyboardMarkup(buttons)


def _timeframe_kb(gid: str, user_id: int) -> InlineKeyboardMarkup:
    """
    Build a timeframe keyboard showing per-user active/inactive icons.
    """
    active_alerts = alert_service.get_active_alerts() or {}
    active_tfs = set()
    if user_id is not None:
        for a in active_alerts.values():
            try:
                if int(a.get("user_id")) == int(user_id) and a.get("group_id") == gid:
                    active_tfs.add(int(a.get("timeframe_min", -1)))
            except Exception:
                continue

    rows = []
    row = []
    for tf in TIMEFRAMES:
        icon = "🔔" if tf in active_tfs else "🔕"
        label = f"{tf}m {icon}" if tf < 1440 else f"1d {icon}"
        row.append(InlineKeyboardButton(label, callback_data=f"timeframe::{gid}::{tf}"))
        if len(row) == 2:
            rows.append(row)
            row = []
    if row:
        rows.append(row)
    rows.append([InlineKeyboardButton("⬅️ Back", callback_data="menu_crypto_usd")])
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
        [InlineKeyboardButton("⬅️ Back", callback_data="menu_crypto_usd")],
    ])


def _resolve_label_and_symbols(gid: str):
    """
    Try to resolve group label/symbols via alert_service (canonical groups).
    Falls back to FALLBACK_SYMBOLS if not found.
    """
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

async def crypto_usd_menu_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    await q.edit_message_text("Crypto (USD) groups:", reply_markup=_menu_kb())


async def group_select_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    gid = q.data
    if gid not in GROUP_ID_SET:
        await q.edit_message_text("Unsupported group.")
        return
    label, _ = _resolve_label_and_symbols(gid)
    user_id = update.effective_user.id if update.effective_user else None
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

    user_id = update.effective_user.id if update.effective_user else None
    is_active = False
    try:
        active_alerts = alert_service.get_active_alerts() or {}
        for a in active_alerts.values():
            try:
                if int(a.get("user_id")) == int(user_id) and a.get("group_id") == gid and int(a.get("timeframe_min", -1)) == tf:
                    is_active = True
                    break
            except Exception:
                continue
    except Exception:
        logger.debug("timeframe_select_handler: error checking status", exc_info=True)

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
        for buf in bufs:
            buf.seek(0)
            await context.bot.send_photo(chat_id=q.message.chat_id, photo=buf)
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
