# services/alert_service.py

# This is a simplified version. You can expand it with real-time data and scheduling later.

# In-memory alert storage (replace with DB or persistent store in production)
active_alerts = {}

def set_ssmt_alert(user_id: int, group: str, timeframe: str) -> None:
    """
    Stores an alert configuration for a user.
    """
    key = f"{user_id}:{group}:{timeframe}"
    active_alerts[key] = {
        "user_id": user_id,
        "group": group,
        "timeframe": timeframe,
        "active": True
    }

def deactivate_ssmt_alert(user_id: int, group: str, timeframe: str) -> None:
    """
    Deactivates an alert configuration.
    """
    key = f"{user_id}:{group}:{timeframe}"
    if key in active_alerts:
        active_alerts[key]["active"] = False

def check_ssmt_violation(group: str, timeframe: str, market_data: dict) -> bool:
    """
    Applies your SSMT logic to determine if a rule is violated.
    This is a placeholder—you'll define the real logic later.
    """
    # Example logic:
    # Compare first quarter low/high with second quarter movement
    # If violation detected, return True
    return True  # Simulate a violation for now

def get_active_alerts() -> list:
    """
    Returns all currently active alerts.
    """
    return [alert for alert in active_alerts.values() if alert["active"]]
