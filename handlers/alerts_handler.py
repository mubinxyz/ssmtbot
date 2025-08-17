# handlers/alerts_handler.py
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes

from services import alert_service
from keyboards.alerts_menu import get_alerts_menu, get_trio_groups_menu
from typing import List, Tuple, Optional
import logging
import asyncio

logger = logging.getLogger(__name__)


async def menu_alerts_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    cq = update.callback_query
    await cq.answer()
    await cq.edit_message_text(
        text="🔔 Alerts — choose alert type:",
        reply_markup=get_alerts_menu()
    )


async def alerts_one_symbol_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    cq = update.callback_query
    await cq.answer()
    await cq.edit_message_text(
        text="🔔 One-symbol alerts are coming soon.\n\nYou'll be able to set alerts per single symbol here.",
        reply_markup=get_alerts_menu()
    )


async def alerts_trio_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Show only active trio alerts for this user.
    If none exist, show a message with a button to list all trio groups.
    """
    cq = update.callback_query
    await cq.answer()

    user_id = update.effective_user.id

    # find active alerts for this user that correspond to trio groups
    active = alert_service.get_active_alerts()
    user_alerts = [
        a for a in active
        if a.get("user_id") == int(user_id)
    ]

    # Filter to only groups that exist in GROUPS (defensive)
    user_trio_rows: List[List[InlineKeyboardButton]] = []
    for a in user_alerts:
        cat = a.get("category")
        gid = a.get("group_id")
        tf = a.get("timeframe_min")
        # find group label
        group_def = alert_service.find_group(cat, gid) if cat and gid else None
        label = group_def.get("label") if group_def else f"{gid}"
        btn_text = f"{label} — {tf}min"
        cb = f"manage_active_trio::{cat}::{gid}::{tf}"
        user_trio_rows.append([InlineKeyboardButton(btn_text, callback_data=cb)])

    if user_trio_rows:
        # add navigation buttons
        user_trio_rows.append([InlineKeyboardButton("🔙 Back to Alerts", callback_data="menu_alerts")])
        user_trio_rows.append([InlineKeyboardButton("📋 Show all trio groups", callback_data="trio_group_list_all")])
        kb = InlineKeyboardMarkup(user_trio_rows)
        await cq.edit_message_text(
            text="🔔 Your active Trio alerts:",
            reply_markup=kb
        )
        return

    # if no active trio alerts, present message with option to view all trios
    kb = InlineKeyboardMarkup([
        [InlineKeyboardButton("📋 Show all trio groups", callback_data="trio_group_list_all")],
        [InlineKeyboardButton("🔙 Back", callback_data="menu_alerts")],
    ])
    await cq.edit_message_text(
        text="You don't have any active Trio alerts yet.\n\nTap below to view all available trio groups and create alerts.",
        reply_markup=kb
    )


async def trio_group_list_all_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Show the full trio groups list (fallback when user has no active trio alerts).
    """
    cq = update.callback_query
    await cq.answer()
    await cq.edit_message_text(
        text="🔔 Trio alerts — available groups:",
        reply_markup=get_trio_groups_menu()
    )


async def manage_active_trio_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Manage a single active trio alert.
    Callback format: manage_active_trio::{category}::{group_id}::{timeframe}
    Shows options: View charts, Deactivate alert, Back to Alerts.
    """
    cq = update.callback_query
    await cq.answer()
    data = cq.data or ""
    parts = data.split("::")
    if len(parts) < 4:
        await cq.edit_message_text(text="Invalid selection.", reply_markup=get_alerts_menu())
        return

    _, category, group_id, timeframe = parts[0], parts[1], parts[2], parts[3]
    group_def = alert_service.find_group(category, group_id)
    label = group_def.get("label") if group_def else group_id
    tf = int(timeframe)

    kb = InlineKeyboardMarkup([
        [InlineKeyboardButton(f"🖼 View charts ({tf}min)", callback_data=f"manage_active_trio_view::{category}::{group_id}::{tf}")],
        [InlineKeyboardButton(f"⛔ Deactivate alert ({tf}min)", callback_data=f"manage_active_trio_deactivate::{category}::{group_id}::{tf}")],
        [InlineKeyboardButton("🔙 Back to Alerts", callback_data="menu_alerts")],
    ])

    await cq.edit_message_text(
        text=f"Manage alert: {label} — {tf}min",
        reply_markup=kb
    )


async def manage_active_trio_view_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Generate and send charts for the selected trio alert.
    Uses services.chart_service.generate_chart(...) — this may be IO heavy.
    """
    cq = update.callback_query
    await cq.answer()

    data = cq.data or ""
    parts = data.split("::")
    if len(parts) < 4:
        await cq.edit_message_text(text="Invalid request.", reply_markup=get_trio_groups_menu())
        return

    _, category, group_id, timeframe = parts[0], parts[1], parts[2], parts[3]
    tf = int(timeframe)
    group_def = alert_service.find_group(category, group_id)
    if group_def is None:
        await cq.edit_message_text(text="Group not found.", reply_markup=get_trio_groups_menu())
        return

    symbols = group_def.get("symbols", [])
    # generate charts using the chart service module (can be heavy so use to_thread if needed)
    try:
        # chart_service.generate_chart is async in our design; import and await
        from services import chart_service
        charts = await chart_service.generate_chart(symbols, timeframe=tf)
    except Exception as e:
        logger.exception("Chart generation failed: %s", e)
        charts = None

    # send charts as photos below the edited message (keep the edited message as-is)
    if charts:
        for buf in charts:
            try:
                buf.seek(0)
                await cq.message.reply_photo(photo=buf)
            except Exception as e:
                logger.exception("Failed to send chart image: %s", e)

    # keep the manage menu visible
    await cq.edit_message_text(
        text=f"Sent charts for {group_def.get('label')} ({tf}min).",
        reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("🔙 Back to Alerts", callback_data="menu_alerts")]
        ])
    )


async def manage_active_trio_deactivate_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Deactivate an active alert for this user.
    Callback: manage_active_trio_deactivate::{category}::{group_id}::{timeframe}
    """
    cq = update.callback_query
    await cq.answer()

    user_id = update.effective_user.id
    data = cq.data or ""
    parts = data.split("::")
    if len(parts) < 4:
        await cq.edit_message_text(text="Invalid request.", reply_markup=get_trio_groups_menu())
        return

    _, category, group_id, timeframe = parts[0], parts[1], parts[2], parts[3]
    tf = int(timeframe)

    ok = alert_service.deactivate_ssmt_alert(user_id=user_id, category=category, group_id=group_id, timeframe_min=tf)
    if ok:
        await cq.edit_message_text(
            text=f"✅ Alert deactivated for {group_id} ({tf}min).",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🔙 Back to Alerts", callback_data="menu_alerts")],
                [InlineKeyboardButton("🏠 Main menu", callback_data="back_to_main")]
            ])
        )
    else:
        await cq.edit_message_text(
            text="Could not deactivate (alert not found).",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🔙 Back to Alerts", callback_data="menu_alerts")]
            ])
        )
