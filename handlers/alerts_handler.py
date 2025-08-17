# handlers/alerts_handler.py
import logging
import json
import os
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import ContextTypes
from services.alert_service import load_alerts, save_alerts, GROUP_CHARACTERISTICS

logger = logging.getLogger(__name__)

ALERTS_STORE_FILE = "alerts_store.json"

def _format_timeframe(minutes):
    """Convert minutes to readable format."""
    if minutes < 60:
        return f"{minutes}m"
    elif minutes == 60:
        return "1h"
    elif minutes == 240:
        return "4h"
    elif minutes == 1440:
        return "1d"
    else:
        hours = minutes // 60
        return f"{hours}h"

def _get_group_label(group_id):
    """Get human-readable label for group ID."""
    if group_id in GROUP_CHARACTERISTICS:
        return GROUP_CHARACTERISTICS[group_id].get("label", group_id)
    return group_id.replace("_", " ").upper()

def _get_user_alerts(user_id):
    """Get all alerts for a specific user."""
    alerts = load_alerts()
    user_alerts = []
    
    for key, alert_data in alerts.items():
        if alert_data.get("user_id") == user_id:
            user_alerts.append((key, alert_data))
    
    return user_alerts

def _alerts_menu_kb(user_id) -> InlineKeyboardMarkup:
    """Create the alerts menu keyboard with user's alerts."""
    user_alerts = _get_user_alerts(user_id)
    
    buttons = []
    
    if user_alerts:
        buttons.append([InlineKeyboardButton("🔔 Your Active Alerts", callback_data="dummy")])
        
        for key, alert in user_alerts:
            if alert.get("active", False):
                group_label = _get_group_label(alert["group_id"])
                timeframe = _format_timeframe(alert["timeframe_min"])
                status = "✅" if alert.get("active") else "❌"
                
                # Alert display button
                alert_text = f"{status} {group_label} ({timeframe})"
                buttons.append([InlineKeyboardButton(alert_text, callback_data=f"view_alert::{key}")])
                
                # Delete button for each alert
                delete_text = f"🗑️ Delete {group_label} ({timeframe})"
                buttons.append([InlineKeyboardButton(delete_text, callback_data=f"delete_alert::{key}")])
    else:
        buttons.append([InlineKeyboardButton("📭 No active alerts", callback_data="dummy")])
    
    # Add navigation buttons
    buttons.append([InlineKeyboardButton("➕ Add New Alert", callback_data="menu_forex")])
    buttons.append([InlineKeyboardButton("🔙 Back to Main Menu", callback_data="back_to_main")])
    
    return InlineKeyboardMarkup(buttons)

async def alerts_menu_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle the alerts menu display."""
    query = update.callback_query
    await query.answer()
    
    user_id = update.effective_user.id
    user_alerts = _get_user_alerts(user_id)
    
    active_count = sum(1 for _, alert in user_alerts if alert.get("active", False))
    
    menu_text = f"🔔 Alerts Management\n\n"
    menu_text += f"📊 Active Alerts: {active_count}\n"
    menu_text += f"👤 User ID: {user_id}\n\n"
    menu_text += "Select an alert to view details or delete it:"
    
    await query.edit_message_text(
        text=menu_text,
        reply_markup=_alerts_menu_kb(user_id)
    )

async def delete_alert_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle alert deletion."""
    query = update.callback_query
    await query.answer()
    
    user_id = update.effective_user.id
    data = query.data
    
    if not data.startswith("delete_alert::"):
        await query.edit_message_text("Invalid delete request.")
        return
    
    alert_key = data.split("::", 1)[1]
    
    # Load alerts
    alerts = load_alerts()
    
    # Check if alert exists and belongs to user
    if alert_key not in alerts:
        await query.edit_message_text("Alert not found.")
        return
    
    alert_data = alerts[alert_key]
    if alert_data.get("user_id") != user_id:
        await query.edit_message_text("You don't have permission to delete this alert.")
        return
    
    # Get alert details for confirmation message
    group_label = _get_group_label(alert_data["group_id"])
    timeframe = _format_timeframe(alert_data["timeframe_min"])
    
    # Remove the alert
    del alerts[alert_key]
    save_alerts(alerts)
    
    confirmation_text = f"✅ Alert deleted successfully!\n\n"
    confirmation_text += f"Group: {group_label}\n"
    confirmation_text += f"Timeframe: {timeframe}\n"
    
    # Show confirmation and refresh alerts menu
    await query.edit_message_text(confirmation_text)
    
    # After a short delay, show the alerts menu again
    import asyncio
    await asyncio.sleep(1.5)
    
    # Refresh the alerts menu
    user_alerts = _get_user_alerts(user_id)
    active_count = sum(1 for _, alert in user_alerts if alert.get("active", False))
    
    menu_text = f"🔔 Alerts Management\n\n"
    menu_text += f"📊 Active Alerts: {active_count}\n"
    menu_text += f"👤 User ID: {user_id}\n\n"
    menu_text += "Select an alert to view details or delete it:"
    
    await query.edit_message_text(
        text=menu_text,
        reply_markup=_alerts_menu_kb(user_id)
    )

async def view_alert_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle alert viewing (optional detail view)."""
    query = update.callback_query
    await query.answer()
    
    user_id = update.effective_user.id
    data = query.data
    
    if not data.startswith("view_alert::"):
        await query.edit_message_text("Invalid view request.")
        return
    
    alert_key = data.split("::", 1)[1]
    
    # Load alerts
    alerts = load_alerts()
    
    # Check if alert exists and belongs to user
    if alert_key not in alerts:
        await query.edit_message_text("Alert not found.")
        return
    
    alert_data = alerts[alert_key]
    if alert_data.get("user_id") != user_id:
        await query.edit_message_text("You don't have permission to view this alert.")
        return
    
    # Display alert details
    group_label = _get_group_label(alert_data["group_id"])
    timeframe = _format_timeframe(alert_data["timeframe_min"])
    status = "✅ Active" if alert_data.get("active") else "❌ Inactive"
    
    detail_text = f"🔍 Alert Details\n\n"
    detail_text += f"Group: {group_label}\n"
    detail_text += f"Timeframe: {timeframe}\n"
    detail_text += f"Status: {status}\n"
    detail_text += f"Category: {alert_data['category']}\n"
    detail_text += f"Created: {alert_data['created_at']}\n\n"
    
    # Add back button
    back_kb = InlineKeyboardMarkup([[
        InlineKeyboardButton("⬅️ Back to Alerts", callback_data="menu_alerts")
    ]])
    
    await query.edit_message_text(
        text=detail_text,
        reply_markup=back_kb
    )

async def ssmt_alerts_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle SSMT Alerts selection - redirect to forex menu for now."""
    query = update.callback_query
    await query.answer()
    
    # For now, redirect to forex menu since that's where SSMT alerts are implemented
    from handlers.forex_handler import forex_menu_handler
    await forex_menu_handler(update, context)

async def single_symbol_alerts_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle Single Symbol Alerts selection - show coming soon message."""
    query = update.callback_query
    await query.answer()
    
    await query.edit_message_text(
        text="🔹 Single Symbol Alerts\n\n🚧 Coming Soon!\n\nStay tuned for individual symbol alert functionality.",
        reply_markup=InlineKeyboardMarkup([[
            InlineKeyboardButton("⬅️ Back to Alerts", callback_data="menu_alerts")
        ]])
    )