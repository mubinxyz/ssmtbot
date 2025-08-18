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
    """Create the alerts menu keyboard with alert type selection."""
    buttons = []
    
    # Alert type selection buttons
    buttons.append([InlineKeyboardButton("🔹 Single Alerts (Coming Soon)", callback_data="single_alerts")])
    buttons.append([InlineKeyboardButton(" Trio/Four Alerts", callback_data="show_trio_alerts")])
    
    # Add navigation buttons
    buttons.append([InlineKeyboardButton("🔙 Back to Main Menu", callback_data="back_to_main")])
    
    return InlineKeyboardMarkup(buttons)

def _trio_alerts_kb(user_id) -> InlineKeyboardMarkup:
    """Create keyboard for trio/four alerts with delete options."""
    user_alerts = _get_user_alerts(user_id)
    
    buttons = []
    
    if user_alerts:
        # Group alerts by category and group_id
        grouped_alerts = {}
        for key, alert in user_alerts:
            if alert.get("active", True):  # Default to True if not specified
                category = alert.get("category", "Unknown")
                group_id = alert.get("group_id", "Unknown")
                group_key = f"{category}::{group_id}"
                
                if group_key not in grouped_alerts:
                    grouped_alerts[group_key] = {
                        'category': category,
                        'group_id': group_id,
                        'alerts': []
                    }
                grouped_alerts[group_key]['alerts'].append((key, alert))
        
        # Display grouped alerts
        for group_key, group_data in grouped_alerts.items():
            category = group_data['category']
            group_id = group_data['group_id']
            alerts_list = group_data['alerts']
            
            if alerts_list:
                # Show group header
                group_label = _get_group_label(group_id)
                buttons.append([InlineKeyboardButton(f"📊 {group_label}", callback_data="dummy")])
                
                # Show all alerts in this group
                alert_lines = []
                for _, alert in alerts_list:
                    timeframe = _format_timeframe(alert["timeframe_min"])
                    status = "✅" if alert.get("active", True) else "❌"
                    alert_lines.append(f"  {status} {timeframe}")
                
                # Display all timeframes in the group
                if alert_lines:
                    group_text = "\n".join(alert_lines)
                    buttons.append([InlineKeyboardButton(group_text, callback_data="dummy")])
                
                # Delete button for the entire group
                delete_text = f"🗑️ Delete Group"
                buttons.append([InlineKeyboardButton(delete_text, callback_data=f"delete_group::{group_key}")])
    else:
        buttons.append([InlineKeyboardButton("📭 No active trio/four alerts", callback_data="dummy")])
    
    # Navigation buttons
    buttons.append([InlineKeyboardButton("⬅️ Back to Alerts Menu", callback_data="menu_alerts")])
    
    return InlineKeyboardMarkup(buttons)

async def alerts_menu_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle the alerts menu display."""
    query = update.callback_query
    await query.answer()
    
    user_id = update.effective_user.id
    
    menu_text = "🔔 Alerts Management\n\n"
    menu_text += "Select the type of alerts you want to manage:\n\n"
    menu_text += "🔹 Single Alerts - Individual symbol alerts (Coming Soon)\n"
    menu_text += " Trio/Four Alerts - Group alerts for multiple symbols\n"
    
    await query.edit_message_text(
        text=menu_text,
        reply_markup=_alerts_menu_kb(user_id)
    )

async def show_trio_alerts_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle trio/four alerts display."""
    query = update.callback_query
    await query.answer()
    
    user_id = update.effective_user.id
    user_alerts = _get_user_alerts(user_id)
    
    # Count unique groups
    unique_groups = set()
    active_count = 0
    
    for _, alert in user_alerts:
        if alert.get("active", True):
            unique_groups.add(f"{alert.get('category', 'Unknown')}::{alert.get('group_id', 'Unknown')}")
            active_count += 1
    
    group_count = len(unique_groups)
    
    menu_text = f" Trio/Four Alerts Management\n\n"
    menu_text += f"📊 Active Alert Groups: {group_count}\n"
    menu_text += f"📈 Total Active Alerts: {active_count}\n"
    menu_text += f"👤 User ID: {user_id}\n\n"
    menu_text += "Your grouped alerts are displayed below with delete options:"
    
    await query.edit_message_text(
        text=menu_text,
        reply_markup=_trio_alerts_kb(user_id)
    )

async def delete_group_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle group deletion."""
    query = update.callback_query
    await query.answer()
    
    user_id = update.effective_user.id
    data = query.data
    
    if not data.startswith("delete_group::"):
        await query.edit_message_text("Invalid delete request.")
        return
    
    group_key = data.split("::", 1)[1]
    
    # Parse the group key to get category and group_id
    try:
        category, group_id = group_key.split("::", 1)
    except ValueError:
        await query.edit_message_text("Invalid group key format.")
        return
    
    # Load alerts
    alerts = load_alerts()
    
    # Find and delete all alerts in this group for this user
    deleted_count = 0
    keys_to_delete = []
    
    for key, alert_data in alerts.items():
        if (alert_data.get("user_id") == user_id and 
            alert_data.get("category") == category and 
            alert_data.get("group_id") == group_id):
            keys_to_delete.append(key)
    
    # Delete the alerts
    for key in keys_to_delete:
        if key in alerts:
            del alerts[key]
            deleted_count += 1
    
    save_alerts(alerts)
    
    # Get group label for confirmation message
    group_label = _get_group_label(group_id)
    
    confirmation_text = f"✅ Group deleted successfully!\n\n"
    confirmation_text += f"Group: {group_label}\n"
    confirmation_text += f"Deleted Alerts: {deleted_count}\n"
    
    # Show confirmation
    await query.edit_message_text(confirmation_text)
    
    # After a short delay, show the trio alerts menu again
    import asyncio
    await asyncio.sleep(1.5)
    
    # Refresh the trio alerts menu
    user_alerts = _get_user_alerts(user_id)
    unique_groups = set()
    active_count = 0
    
    for _, alert in user_alerts:
        if alert.get("active", True):
            unique_groups.add(f"{alert.get('category', 'Unknown')}::{alert.get('group_id', 'Unknown')}")
            active_count += 1
    
    group_count = len(unique_groups)
    
    menu_text = f" Trio/Four Alerts Management\n\n"
    menu_text += f"📊 Active Alert Groups: {group_count}\n"
    menu_text += f"📈 Total Active Alerts: {active_count}\n"
    menu_text += f"👤 User ID: {user_id}\n\n"
    menu_text += "Your grouped alerts are displayed below with delete options:"
    
    await query.edit_message_text(
        text=menu_text,
        reply_markup=_trio_alerts_kb(user_id)
    )

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

# Keep the other handlers for backward compatibility
async def delete_alert_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle individual alert deletion (kept for compatibility)."""
    query = update.callback_query
    await query.answer()
    
    await query.edit_message_text(
        text="This function has been updated. Please use the group deletion feature.",
        reply_markup=InlineKeyboardMarkup([[
            InlineKeyboardButton("⬅️ Back to Trio Alerts", callback_data="show_trio_alerts")
        ]])
    )

async def view_alert_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle alert viewing (kept for compatibility)."""
    query = update.callback_query
    await query.answer()
    
    await query.edit_message_text(
        text="This function has been updated. Group alerts are displayed on the trio alerts page.",
        reply_markup=InlineKeyboardMarkup([[
            InlineKeyboardButton("⬅️ Back to Trio Alerts", callback_data="show_trio_alerts")
        ]])
    )

async def ssmt_alerts_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle SSMT Alerts selection - redirect to forex menu for now."""
    query = update.callback_query
    await query.answer()
    
    # For now, redirect to forex menu since that's where SSMT alerts are implemented
    from handlers.forex_handler import forex_menu_handler
    await forex_menu_handler(update, context)