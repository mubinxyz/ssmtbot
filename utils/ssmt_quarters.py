# utils/ssmt_quarters.py

from datetime import datetime, timedelta
import pandas as pd

def get_ssmt_quarters(date: datetime) -> list[pd.Timestamp]:
    """
    Returns 5 timestamps dividing the day into 4 quarters in UTC+3:30.
    """
    base = date.replace(hour=1, minute=30, second=0, microsecond=0)
    offsets = [0, 6, 12, 18, 24]  # hours from 01:30
    quarters = [base + timedelta(hours=h) for h in offsets]
    return pd.to_datetime(quarters)
