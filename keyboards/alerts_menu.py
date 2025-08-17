# keyboards/alerts_menu.py
from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from typing import List, Tuple

# builds static alerts menu + dynamic trio-groups menu (reads GROUPS from services.alert_service)
def get_alerts_menu() -> InlineKeyboardMarkup:
    """
    Alerts submenu:
      - One symbol alerts (placeholder)
      - Trio alerts (navigates to the trio groups list)
    """
    buttons = [
        [InlineKeyboardButton("🔔 One symbol alerts", callback_data="alerts_one_symbol")],
        [InlineKeyboardButton("🔔 Trio alerts", callback_data="alerts_trio")],
        [InlineKeyboardButton("Back", callback_data="back_to_main")],
    ]
    return InlineKeyboardMarkup(buttons)


def _flatten_groups_for_trio() -> List[Tuple[str, str, str]]:
    """
    Return list of tuples (category, group_id, label) for all groups.
    """
    try:
        from services import alert_service
    except Exception:
        return []

    out = []
    for category, groups in alert_service.GROUPS.items():
        for g in groups:
            # we show every group — caller can filter if needed
            out.append((category, g["id"], g.get("label", g["id"])))
    return out


def get_trio_groups_menu() -> InlineKeyboardMarkup:
    """
    Build an InlineKeyboardMarkup listing all trio groups configured in alert_service.GROUPS.
    Each button callback_data: "trio_group::category::group_id"
    """
    rows = []
    for category, gid, label in _flatten_groups_for_trio():
        # show label with (category) suffix for clarity
        text = f"{label} — {category}"
        cb = f"trio_group::{category}::{gid}"
        rows.append([InlineKeyboardButton(text, callback_data=cb)])

    # add navigation row(s)
    rows.append([InlineKeyboardButton("🔙 Back to Alerts", callback_data="menu_alerts")])
    rows.append([InlineKeyboardButton("🏠 Main menu", callback_data="back_to_main")])
    return InlineKeyboardMarkup(rows)
