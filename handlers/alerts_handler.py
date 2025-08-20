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
    buttons.append([InlineKeyboardButton(" Trio/Four Alerts", callback_data="show_alerts")])

    # Add navigation buttons
    buttons.append([InlineKeyboardButton("🔙 Back to Main Menu", callback_data="back_to_main")])

    return InlineKeyboardMarkup(buttons)


def _build_alerts_list_and_kb(user_id, context) -> (str, InlineKeyboardMarkup):
    """Build the textual list of alerts and an inline keyboard with delete buttons.

    Because Telegram callback_data is limited in size, we create a short integer id per
    alert and store a mapping in context.user_data['alerts_map'] so delete callbacks can
    reference alerts safely.
    """
    user_alerts = _get_user_alerts(user_id)
    if not user_alerts:
        text = "🔔 Trio/Four Alerts Management\n\n📭 You have no alerts registered."
        kb = InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Back to Alerts Menu", callback_data="menu_alerts")]])
        # ensure map is empty
        context.user_data['alerts_map'] = {}
        return text, kb

    lines = ["🔔 Trio/Four Alerts Management\n"]
    lines.append(f"👤 User ID: {user_id}\n")

    # We'll list alerts as numbered items
    alerts_map = {}
    buttons = []
    idx = 1

    for key, alert in sorted(user_alerts, key=lambda x: (x[1].get('category', ''), x[1].get('group_id', ''), x[1].get('timeframe_min', 0))):
        category = alert.get('category', 'Unknown')
        group_id = alert.get('group_id', 'Unknown')
        timeframe = _format_timeframe(alert.get('timeframe_min', 0))
        status = "✅ Active" if alert.get('active', True) else "❌ Inactive"
        group_label = _get_group_label(group_id)
        created = alert.get('created_at', '')

        lines.append(f"{idx}. {group_label} — {timeframe} — {category} — {status}")

        # map short id to the actual store key
        alerts_map[str(idx)] = key

        # add delete button for this alert (one per row)
        buttons.append([InlineKeyboardButton(f"🗑️ Delete {idx}", callback_data=f"delete_alert::{idx}")])

        idx += 1

    # Add a navigation row at the end
    buttons.append([InlineKeyboardButton("⬅️ Back to Alerts Menu", callback_data="menu_alerts")])

    # Store the mapping in user_data so delete handler can find the real key
    context.user_data['alerts_map'] = alerts_map

    text = "\n".join(lines)
    kb = InlineKeyboardMarkup(buttons)
    return text, kb


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


async def show_alerts_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show all trio/four alerts for the user with per-alert delete buttons."""
    query = update.callback_query
    await query.answer()

    user_id = update.effective_user.id
    text, kb = _build_alerts_list_and_kb(user_id, context)

    await query.edit_message_text(
        text=text,
        reply_markup=kb
    )


async def delete_alert_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle individual alert deletion.

    Expects callback_data like "delete_alert::<short_id>" where <short_id> maps to the
    real key via context.user_data['alerts_map'].
    """
    query = update.callback_query
    await query.answer()

    user_id = update.effective_user.id
    data = query.data or ""

    if not data.startswith("delete_alert::"):
        await query.edit_message_text("Invalid delete request.")
        return

    short_id = data.split("::", 1)[1]
    alerts_map = context.user_data.get('alerts_map', {})
    real_key = alerts_map.get(short_id)

    if not real_key:
        # mapping lost or invalid id; try to be helpful by attempting to find a matching alert
        alerts = load_alerts()
        # Try to find a unique alert that matches the displayed index by ordering
        user_alerts = _get_user_alerts(user_id)
        try:
            idx = int(short_id) - 1
            candidate_key = sorted(user_alerts, key=lambda x: (x[1].get('category', ''), x[1].get('group_id', ''), x[1].get('timeframe_min', 0)))[idx][0]
            real_key = candidate_key
        except Exception:
            await query.edit_message_text("Could not locate the alert to delete (it may be stale). Please open the alerts menu again.")
            return

    # Load alerts and delete if owner matches
    alerts = load_alerts()
    alert_data = alerts.get(real_key)
    if not alert_data or alert_data.get('user_id') != user_id:
        await query.edit_message_text("Alert not found or you don't have permission to delete it.")
        return

    # Perform deletion
    try:
        del alerts[real_key]
        save_alerts(alerts)
    except KeyError:
        await query.edit_message_text("Failed to delete alert: not found.")
        return
    except Exception as e:
        logger.exception("Error deleting alert: %s", e)
        await query.edit_message_text("Unexpected error while deleting the alert.")
        return

    # Rebuild the list and keyboard (this will also refresh the mapping)
    text, kb = _build_alerts_list_and_kb(user_id, context)

    confirmation_text = "✅ Alert deleted.\n\n" + text

    # Edit message to show updated list
    await query.edit_message_text(
        text=confirmation_text,
        reply_markup=kb
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
async def view_alert_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle alert viewing (kept for compatibility)."""
    query = update.callback_query
    await query.answer()

    await query.edit_message_text(
        text="This function has been updated. Group alerts are displayed on the trio alerts page.",
        reply_markup=InlineKeyboardMarkup([[
            InlineKeyboardButton("⬅️ Back to Trio Alerts", callback_data="show_alerts")
        ]])
    )


async def ssmt_alerts_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle SSMT Alerts selection - redirect to forex menu for now."""
    query = update.callback_query
    await query.answer()

    # For now, redirect to forex menu since that's where SSMT alerts are implemented
    from handlers.forex_handler import forex_menu_handler
    await forex_menu_handler(update, context)
