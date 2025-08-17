# services/alert_service.py

import json
import logging
import os
import pandas as pd
from datetime import datetime, timedelta, timezone
import pytz
from utils.get_data import get_ohlc
from utils.normalize_data import normalize_symbol

# --- Import for chart generation ---
from services.chart_service import generate_chart
import asyncio

# --- Configuration ---
ALERTS_STORE_FILE = "alerts_store.json"
TIMEZONE_LITEFINANCE = pytz.timezone('Etc/GMT-3')  # UTC+3
TIMEZONE_TARGET = pytz.timezone('Asia/Tehran')     # UTC+3:30 (Example, adjust if needed)
# TIMEZONE_TARGET = pytz.FixedOffset(210)          # Direct UTC+3:30 offset if preferred

logger = logging.getLogger(__name__)

# --- Alert Storage Management ---
def load_alerts():
    """Loads alerts from the JSON file."""
    if not os.path.exists(ALERTS_STORE_FILE):
        return {}
    try:
        with open(ALERTS_STORE_FILE, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"Error loading alerts from {ALERTS_STORE_FILE}: {e}")
        return {}

def save_alerts(alerts):
    """Saves alerts to the JSON file."""
    try:
        with open(ALERTS_STORE_FILE, 'w') as f:
            json.dump(alerts, f, indent=2)
    except IOError as e:
        logger.error(f"Error saving alerts to {ALERTS_STORE_FILE}: {e}")

# --- Alert Definition Management ---
def _generate_alert_key(user_id, category, group_id, timeframe_min):
    """Generates a unique key for an alert."""
    return f"{user_id}::{category}::{group_id}::{timeframe_min}"

def set_ssmt_alert(user_id: int, group_id: str, timeframe_min: int, category: str = "FOREX CURRENCIES"):
    """
    Sets or activates an alert for a user.
    If the alert already exists, it's reactivated.
    If not, a new alert is created and activated.
    """
    alerts = load_alerts()
    key = _generate_alert_key(user_id, category, group_id, timeframe_min)
    if key in alerts:
        # Reactivate existing alert
        alerts[key]["active"] = True
        alerts[key]["updated_at"] = datetime.now(timezone.utc).isoformat() + 'Z'
    else:
        # Create new alert
        alerts[key] = {
            "user_id": user_id,
            "category": category,
            "group_id": group_id,
            "timeframe_min": timeframe_min,
            "active": True,
            "created_at": datetime.now(timezone.utc).isoformat() + 'Z'
        }
    save_alerts(alerts)
    logger.info(f"Alert set/activated for user {user_id}, group {group_id}, timeframe {timeframe_min}min.")

def deactivate_ssmt_alert(user_id: int, group_id: str, timeframe_min: int, category: str = "FOREX CURRENCIES") -> bool:
    """
    Deactivates an alert for a user.
    Returns True if the alert was found and deactivated, False otherwise.
    """
    alerts = load_alerts()
    key = _generate_alert_key(user_id, category, group_id, timeframe_min)
    if key in alerts and alerts[key]["active"]:
        alerts[key]["active"] = False
        alerts[key]["updated_at"] = datetime.now(timezone.utc).isoformat() + 'Z'
        save_alerts(alerts)
        logger.info(f"Alert deactivated for user {user_id}, group {group_id}, timeframe {timeframe_min}min.")
        return True
    return False

def get_active_alerts():
    """Retrieves all currently active alerts."""
    alerts = load_alerts()
    return {k: v for k, v in alerts.items() if v.get("active", False)}

# --- Alert Logic Definition ---
# Define group characteristics
GROUP_CHARACTERISTICS = {
    # --- Existing Test Groups (from tests) ---
    "test_move_group": {
        "label": "Test Move Together",
        "symbols": ["AAA", "BBB", "CCC"],
        "type": "move_together"
    },
    "test_reverse_group": {
        "label": "Test Reverse Moving",
        "symbols": ["DDD", "EEE", "FFF"],
        "type": "reverse_moving"
    },
    # --- Real Groups ---
    # Forex
    "dxy_eu_gu": {
        "label": "DXY / EURUSD / GBPUSD",
        "symbols": ["USDX", "EURUSD", "GBPUSD"],
        "type": "reverse_moving"
    },
    # Crypto
    "btc_eth_xrp": {
        "label": "BTCUSD / ETHUSD / XRPUSD",
        "symbols": ["BTCUSD", "ETHUSD", "XRPUSD"],
        "type": "move_together"
    },
    # Futures
    "spx_nq_ym": {
        "label": "S&P500 (SPX) / NASDAQ (NQ) / DOW (YM)",
        "symbols": ["SPX", "NQ", "YM"],
        "type": "move_together"
    },
    # Metals
    "dxy_xau_xag_aud": {
        "label": "DXY / XAU / XAG / AUD",
        "symbols": ["USDX", "XAU", "XAG", "AUD"],
        "type": "reverse_moving"
    }
}

def _find_group_category_by_group_id_flexible(gid: str):
    """Finds category and canonical group ID."""
    if gid in GROUP_CHARACTERISTICS:
        return "FOREX CURRENCIES", gid
    return None, None

def find_group(category: str, group_id: str):
    """Finds group definition."""
    if category == "FOREX CURRENCIES" and group_id in GROUP_CHARACTERISTICS:
        return GROUP_CHARACTERISTICS[group_id]
    return None

# --- Data Processing and Alert Checking ---
def _convert_lf_time_to_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts LiteFinance timestamps (assumed UTC+3) to target timezone (UTC+3:30).
    LiteFinance timestamps are likely in seconds.
    """
    if df.empty or 'timestamp' not in df.columns:
        return df
    # Convert timestamp from seconds to datetime, assuming it's UTC+3
    df['datetime_utc3'] = pd.to_datetime(df['timestamp'], unit='s', utc=True).dt.tz_convert(TIMEZONE_LITEFINANCE)
    # Convert to target timezone (UTC+3:30)
    df['datetime_target'] = df['datetime_utc3'].dt.tz_convert(TIMEZONE_TARGET)
    # Update the main timestamp column to be in the target timezone (as epoch seconds)
    df['timestamp'] = df['datetime_target'].apply(lambda dt: int(dt.timestamp()))
    return df

def _get_data_for_alert_checking(symbols: list[str], timeframe_min: int) -> dict[str, pd.DataFrame]:
    """
    Fetches and preprocesses data for alert checking based on timeframe.
    """
    data_map = {}
    now_target = datetime.now(TIMEZONE_TARGET)
    
    # Special handling for 5-minute alerts
    if timeframe_min == 5:
        # Calculate time window: 1:30 to 7:30 in UTC+3:30
        today = now_target.date()
        start_time = TIMEZONE_TARGET.localize(datetime.combine(today, datetime.min.time().replace(hour=1, minute=30)))
        end_time = TIMEZONE_TARGET.localize(datetime.combine(today, datetime.min.time().replace(hour=7, minute=30)))
        
        # Convert to LiteFinance timezone (UTC+3)
        from_time_lf = start_time.astimezone(TIMEZONE_LITEFINANCE)
        to_time_lf = end_time.astimezone(TIMEZONE_LITEFINANCE)
        
        from_timestamp_lf = int(from_time_lf.timestamp())
        to_timestamp_lf = int(to_time_lf.timestamp())
    else:
        # Original logic for other timeframes
        if timeframe_min <= 15:
            data_resolution = 1
            period_minutes = 6.5 * 24 * 60 if timeframe_min == 15 else 4 * timeframe_min
        else:
            data_resolution = 5
            period_minutes = 24 * (timeframe_min // data_resolution)
        
        from_time_target = now_target - timedelta(minutes=period_minutes)
        from_timestamp_lf = int(from_time_target.astimezone(TIMEZONE_LITEFINANCE).timestamp())
        to_timestamp_lf = int(now_target.astimezone(TIMEZONE_LITEFINANCE).timestamp())
    
    # Determine data resolution
    data_resolution = 1 if timeframe_min <= 15 else 5
    
    for symbol in symbols:
        logger.debug(f"Fetching {data_resolution}min data for {symbol}")
        df = get_ohlc(symbol, timeframe=data_resolution, from_date=from_timestamp_lf, to_date=to_timestamp_lf)
        if not df.empty:
            df = _convert_lf_time_to_target(df)
            data_map[symbol] = df
        else:
            logger.warning(f"Failed to fetch data for {symbol}")
    return data_map

def _resample_to_alert_timeframe(df_1min_or_5min: pd.DataFrame, alert_timeframe_min: int) -> pd.DataFrame:
    """
    Resamples 1min or 5min data to the alert's timeframe.
    Assumes the input DataFrame has 'timestamp', 'open', 'high', 'low', 'close' columns.
    """
    if df_1min_or_5min.empty:
        return df_1min_or_5min
    
    # Convert timestamp to datetime for resampling
    df_resample = df_1min_or_5min.copy()
    df_resample['datetime'] = pd.to_datetime(df_resample['timestamp'], unit='s', utc=True).dt.tz_localize(None)
    df_resample.set_index('datetime', inplace=True)
    
    # Perform resampling
    resampled_df = df_resample.resample(f'{alert_timeframe_min}T', label='right', closed='right').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'timestamp': 'last'
    }).dropna()
    
    # Reset index and process columns
    resampled_df.reset_index(inplace=True)
    resampled_df.rename(columns={'timestamp': 'timestamp_end'}, inplace=True)
    resampled_df.rename(columns={'datetime': 'datetime_start'}, inplace=True)
    resampled_df['timestamp'] = resampled_df['timestamp_end']
    resampled_df.drop(columns=['timestamp_end'], inplace=True)
    return resampled_df

# --- Store triggered alerts with charts ---
TRIGGERED_ALERTS_WITH_CHARTS_FILE = "triggered_alerts_with_charts.json"

def load_triggered_alerts_with_charts():
    """Loads triggered alerts with chart data from the JSON file."""
    if not os.path.exists(TRIGGERED_ALERTS_WITH_CHARTS_FILE):
        return []
    try:
        with open(TRIGGERED_ALERTS_WITH_CHARTS_FILE, 'r') as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            else:
                logger.warning(f"Data in {TRIGGERED_ALERTS_WITH_CHARTS_FILE} is not a list. Returning empty list.")
                return []
    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"Error loading triggered alerts from {TRIGGERED_ALERTS_WITH_CHARTS_FILE}: {e}")
        return []

def save_triggered_alert_with_charts(triggered_alert_data):
    """Appends a triggered alert with chart data to the JSON file."""
    all_triggered = load_triggered_alerts_with_charts()
    all_triggered.append(triggered_alert_data)
    try:
        with open(TRIGGERED_ALERTS_WITH_CHARTS_FILE, 'w') as f:
            json.dump(all_triggered, f, indent=2)
    except IOError as e:
        logger.error(f"Error saving triggered alert to {TRIGGERED_ALERTS_WITH_CHARTS_FILE}: {e}")

async def _generate_charts_for_symbols(symbols: list[str], timeframe_min: int) -> list:
    """Helper function to generate charts for a list of symbols."""
    try:
        chart_buffers = await generate_chart(symbols, timeframe=timeframe_min)
        chart_data_list = []
        for i, buf in enumerate(chart_buffers):
            chart_info = {
                "symbol": symbols[i] if i < len(symbols) else "unknown",
                "timeframe": timeframe_min,
                "generated": True,
            }
            chart_data_list.append(chart_info)
        return chart_data_list
    except Exception as e:
        logger.error(f"Error generating charts for symbols {symbols}: {e}")
        return [{"error": f"Failed to generate chart: {str(e)}"} for _ in symbols]

def check_alert_conditions(alert_key: str, alert_details: dict):
    """
    Checks if the conditions for a specific alert are met.
    Returns a tuple (is_triggered: bool, message: str | None, chart_data_list: list | None).
    """
    user_id = alert_details['user_id']
    group_id = alert_details['group_id']
    timeframe_min = alert_details['timeframe_min']
    category = alert_details['category']
    group_info = find_group(category, group_id)
    
    if not group_info:
        logger.warning(f"Group info not found for alert {alert_key}")
        return False, None, None
    
    symbols = group_info.get('symbols', [])
    group_type = group_info.get('type', 'unknown')
    
    if not symbols:
        logger.warning(f"No symbols defined for group {group_id}")
        return False, None, None
    
    logger.info(f"Checking alert {alert_key} for group {group_id} ({group_type}) with symbols {symbols}")
    
    # 1. Fetch data
    data_map_raw = _get_data_for_alert_checking(symbols, timeframe_min)
    if not data_map_raw:
        logger.error("No data fetched for alert check.")
        return False, None, None
    
    # 2. Resample data to alert timeframe
    data_map = {}
    for symbol, df in data_map_raw.items():
        resampled_df = _resample_to_alert_timeframe(df, timeframe_min)
        if not resampled_df.empty:
            data_map[symbol] = resampled_df
        else:
            logger.warning(f"Resampling failed or resulted in empty data for {symbol}")
    
    if not data_map or len(data_map) != len(symbols):
        logger.error("Insufficient data after resampling for alert check.")
        return False, None, None
    
    # 3. Check conditions based on group type
    required_bars = 2
    min_bars_available = min(len(df) for df in data_map.values())
    
    if min_bars_available < required_bars:
        logger.info("Insufficient historical data for Q1/Q2 comparison.")
        return False, None, None
    
    # Get the last two bars (Q1 and Q2) for each symbol
    q1_q2_data = {}
    for symbol, df in data_map.items():
        if len(df) >= 2:
            q1_data = df.iloc[-2]
            q2_data = df.iloc[-1]
            q1_q2_data[symbol] = {'Q1': q1_data, 'Q2': q2_data}
        else:
            logger.warning(f"Not enough bars for {symbol} to perform Q1/Q2 check.")
            return False, None, None
    
    if not q1_q2_data:
        logger.error("Failed to extract Q1/Q2 data for any symbol.")
        return False, None, None
    
    # Determine the primary symbol (for reverse moving groups)
    primary_symbol = symbols[0]
    other_symbols = symbols[1:]
    
    # --- Condition Logic ---
    # High Break Scenario
    q1_high_primary = q1_q2_data[primary_symbol]['Q1']['high']
    q2_high_primary = q1_q2_data[primary_symbol]['Q2']['high']
    
    if q2_high_primary > q1_high_primary:
        if group_type == "move_together":
            others_broke_high = True
            for sym in other_symbols:
                if q1_q2_data[sym]['Q2']['high'] <= q1_q2_data[sym]['Q1']['high']:
                    others_broke_high = False
                    break
            if not others_broke_high:
                msg = (f"⚠️ Alert Triggered ({group_id}, {timeframe_min}m)!\n"
                       f"Group Type: {group_type}\n"
                       f"{primary_symbol} broke Q1 high ({q1_high_primary:.5f}) in Q2 ({q2_high_primary:.5f}), "
                       f"but not all others followed.")
                logger.info(f"Triggered: {msg}")
                chart_data_list = asyncio.run(_generate_charts_for_symbols(symbols, timeframe_min))
                triggered_data = {
                    "timestamp": datetime.now(timezone.utc).isoformat() + 'Z',
                    "alert_key": alert_key,
                    "alert_details": alert_details,
                    "message": msg,
                    "charts": chart_data_list
                }
                save_triggered_alert_with_charts(triggered_data)
                return True, msg, chart_data_list
        elif group_type == "reverse_moving":
            others_failed_low = True
            for sym in other_symbols:
                if q1_q2_data[sym]['Q2']['low'] <= q1_q2_data[sym]['Q1']['low']:
                    others_failed_low = False
                    break
            if others_failed_low:
                msg = (f"⚠️ Alert Triggered ({group_id}, {timeframe_min}m)!\n"
                       f"Group Type: {group_type}\n"
                       f"{primary_symbol} broke Q1 high ({q1_high_primary:.5f}) in Q2 ({q2_high_primary:.5f}), "
                       f"while others held their Q1 lows.")
                logger.info(f"Triggered: {msg}")
                chart_data_list = asyncio.run(_generate_charts_for_symbols(symbols, timeframe_min))
                triggered_data = {
                    "timestamp": datetime.now(timezone.utc).isoformat() + 'Z',
                    "alert_key": alert_key,
                    "alert_details": alert_details,
                    "message": msg,
                    "charts": chart_data_list
                }
                save_triggered_alert_with_charts(triggered_data)
                return True, msg, chart_data_list
    
    # Low Break Scenario
    q1_low_primary = q1_q2_data[primary_symbol]['Q1']['low']
    q2_low_primary = q1_q2_data[primary_symbol]['Q2']['low']
    
    if q2_low_primary < q1_low_primary:
        if group_type == "move_together":
            others_broke_low = True
            for sym in other_symbols:
                if q1_q2_data[sym]['Q2']['low'] >= q1_q2_data[sym]['Q1']['low']:
                    others_broke_low = False
                    break
            if not others_broke_low:
                msg = (f"⚠️ Alert Triggered ({group_id}, {timeframe_min}m)!\n"
                       f"Group Type: {group_type}\n"
                       f"{primary_symbol} broke Q1 low ({q1_low_primary:.5f}) in Q2 ({q2_low_primary:.5f}), "
                       f"but not all others followed.")
                logger.info(f"Triggered: {msg}")
                chart_data_list = asyncio.run(_generate_charts_for_symbols(symbols, timeframe_min))
                triggered_data = {
                    "timestamp": datetime.now(timezone.utc).isoformat() + 'Z',
                    "alert_key": alert_key,
                    "alert_details": alert_details,
                    "message": msg,
                    "charts": chart_data_list
                }
                save_triggered_alert_with_charts(triggered_data)
                return True, msg, chart_data_list
        elif group_type == "reverse_moving":
            others_failed_high = True
            for sym in other_symbols:
                if q1_q2_data[sym]['Q2']['high'] >= q1_q2_data[sym]['Q1']['high']:
                    others_failed_high = False
                    break
            if others_failed_high:
                msg = (f"⚠️ Alert Triggered ({group_id}, {timeframe_min}m)!\n"
                       f"Group Type: {group_type}\n"
                       f"{primary_symbol} broke Q1 low ({q1_low_primary:.5f}) in Q2 ({q2_low_primary:.5f}), "
                       f"while others held their Q1 highs.")
                logger.info(f"Triggered: {msg}")
                chart_data_list = asyncio.run(_generate_charts_for_symbols(symbols, timeframe_min))
                triggered_data = {
                    "timestamp": datetime.now(timezone.utc).isoformat() + 'Z',
                    "alert_key": alert_key,
                    "alert_details": alert_details,
                    "message": msg,
                    "charts": chart_data_list
                }
                save_triggered_alert_with_charts(triggered_data)
                return True, msg, chart_data_list
    
    logger.debug("Alert conditions not met.")
    return False, None, None