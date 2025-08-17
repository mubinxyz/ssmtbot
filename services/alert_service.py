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
    "dxy_eu_gu": { # Assuming this is the canonical ID used in handlers
        "label": "DXY / EURUSD / GBPUSD",
        "symbols": ["USDX", "EURUSD", "GBPUSD"], # Symbols as expected by get_data
        "type": "reverse_moving" # DXY often moves inversely to EU/GU
    },
    # Crypto
    "btc_eth_xrp": { # Assuming this is the canonical ID used in handlers
        "label": "BTCUSD / ETHUSD / XRPUSD",
        "symbols": ["BTCUSD", "ETHUSD", "XRPUSD"], # Symbols as expected by get_data
        "type": "move_together" # Major cryptos often move together
    },
    # Futures
    "spx_nq_ym": { # Assuming this is the canonical ID used in handlers
         "label": "S&P500 (SPX) / NASDAQ (NQ) / DOW (YM)", # Label as per futures_handler menu
         "symbols": ["SPX", "NQ", "YM"], # Symbols as expected by get_data
         "type": "move_together" # US indices often move together
    },
    # Metals
    "dxy_xau_xag_aud": { # Assuming this is the canonical ID used in handlers
         "label": "DXY / XAU / XAG / AUD", # Label as per metals_handler menu
         "symbols": ["USDX", "XAU", "XAG", "AUD"], # Symbols as expected by get_data
         "type": "reverse_moving" # DXY often moves inversely to precious metals and AUD
    }
    # Add more groups as needed, e.g., if you uncomment btc_eth_total, btc_xrp_doge, etc.
    # "btc_eth_total": {
    #     "label": "BTCUSD / ETHUSD / TOTAL",
    #     "symbols": ["BTCUSD", "ETHUSD", "TOTAL"], # Check if 'TOTAL' is the correct symbol
    #     "type": "move_together"
    # },
    # Uncomment and define others if activated in handlers
    # "dxy_chf_jpy": { ... },
    # "dxy_aud_nzd": { ... },
    # "btc_xrp_doge": { ... },
    # "es_nq_dow": { ... },
    # "spx_dow_nq": { ... },
    # "xau_xag_aud": { ... },
    # "dxy_xau_aud": { ... },
}
def _find_group_category_by_group_id_flexible(gid: str):
    """Finds category and canonical group ID."""
    # This is a simplified version. You might have a more complex structure.
    # For now, assume category is fixed and gid is canonical.
    if gid in GROUP_CHARACTERISTICS:
         # Assuming category is always "FOREX CURRENCIES" for now based on your example
         # You might need to expand this logic if you have multiple categories.
         return "FOREX CURRENCIES", gid
    return None, None
def find_group(category: str, group_id: str):
    """Finds group definition."""
    # Simplified lookup based on the structure defined
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
    # Update the main timestamp column to be in the target timezone (as naive or aware datetime)
    # Using naive datetime for simplicity, assuming downstream processing handles it correctly
    # or keep it as 'datetime_target' aware datetime.
    # Let's replace the original 'timestamp' column with the timestamp in the target timezone (as epoch seconds)
    df['timestamp'] = df['datetime_target'].apply(lambda dt: int(dt.timestamp()))
    # Keep the datetime_target column for reference if needed
    # df.drop(columns=['datetime_utc3'], inplace=True) # Optional: remove intermediate column
    return df
def _get_data_for_alert_checking(symbols: list[str], timeframe_min: int) -> dict[str, pd.DataFrame]:
    """
    Fetches and preprocesses data for alert checking based on timeframe.
    """
    data_map = {}
    now_target = datetime.now(TIMEZONE_TARGET)
    # Determine data resolution and period based on alert timeframe
    if timeframe_min <= 15:
        data_resolution = 1 # 1-minute candles
        # Fetch a week of data for 15m alerts (as per requirement)
        # 1 week = 7 days * 24 hours * 60 minutes = 10080 minutes
        # 10080 points at 1min resolution is well within the 10000 limit
        period_minutes = 7 * 24 * 60 if timeframe_min == 15 else 4 * timeframe_min
    else: # 60, 240
        data_resolution = 5 # 5-minute candles
        # 4h = 240 minutes. 240 / 5 = 48 points per 4h bar.
        # Fetch enough data points. E.g., for 240min alert, fetch several days.
        # 48 * 24 = 1152 points for a day's worth of 4h bars, still within limit.
        period_minutes = 24 * (timeframe_min // data_resolution) # Approx 1 day of higher TF bars
    from_time_target = now_target - timedelta(minutes=period_minutes)
    from_timestamp_lf = int(from_time_target.astimezone(TIMEZONE_LITEFINANCE).timestamp())
    to_timestamp_lf = int(now_target.astimezone(TIMEZONE_LITEFINANCE).timestamp())
    for symbol in symbols:
        logger.debug(f"Fetching {data_resolution}min data for {symbol} from {from_time_target} (target TZ)")
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
    df_resample['datetime'] = pd.to_datetime(df_resample['timestamp'], unit='s', utc=True).dt.tz_localize(None) # Make naive
    df_resample.set_index('datetime', inplace=True)
    # Perform resampling
    resampled_df = df_resample.resample(f'{alert_timeframe_min}T', label='right', closed='right').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'timestamp': 'last' # Use the timestamp of the last sub-period
    }).dropna()
    # Reset index and ensure columns are in the expected order
    resampled_df.reset_index(inplace=True)
    # Convert datetime back to timestamp if needed, or just use the 'timestamp' column preserved
    # Let's use the preserved 'timestamp' from the last sub-period
    resampled_df.rename(columns={'timestamp': 'timestamp_end'}, inplace=True) # Rename to avoid confusion
    # The 'datetime' column is now the start of the resampled period. We need the end timestamp.
    # We already have 'timestamp_end' which is the end timestamp of the last sub-bar.
    # Let's make 'timestamp' represent the end of the resampled bar for consistency.
    resampled_df.rename(columns={'datetime': 'datetime_start'}, inplace=True)
    resampled_df['timestamp'] = resampled_df['timestamp_end']
    resampled_df.drop(columns=['timestamp_end'], inplace=True) # Drop if not needed
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
            # Ensure it's a list
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
    # Optional: Keep only the last N triggered alerts to prevent file bloat
    # MAX_STORED_TRIGGERED = 100
    # all_triggered = all_triggered[-MAX_STORED_TRIGGERED:]
    try:
        with open(TRIGGERED_ALERTS_WITH_CHARTS_FILE, 'w') as f:
            json.dump(all_triggered, f, indent=2)
    except IOError as e:
        logger.error(f"Error saving triggered alert to {TRIGGERED_ALERTS_WITH_CHARTS_FILE}: {e}")

async def _generate_charts_for_symbols(symbols: list[str], timeframe_min: int) -> list:
    """Helper function to generate charts for a list of symbols."""
    try:
        chart_buffers = await generate_chart(symbols, timeframe=timeframe_min)
        # Convert BytesIO buffers to base64 strings or paths if needed for JSON storage.
        # For simplicity here, we'll just store the fact that charts were generated.
        # A more robust solution would serialize the image data.
        chart_data_list = []
        for i, buf in enumerate(chart_buffers):
            # Example: Save to a temporary file or encode to base64
            # Here we just acknowledge the chart was created.
            chart_info = {
                "symbol": symbols[i] if i < len(symbols) else "unknown",
                "timeframe": timeframe_min,
                "generated": True,
                # In a real implementation, you might store base64 or a file path:
                # "data": base64.b64encode(buf.getvalue()).decode('utf-8')
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
    # We need at least 2 completed bars to compare Q1 and Q2
    # Q1 is the bar before the last one, Q2 is the last completed bar.
    required_bars = 2
    # Check if all symbols have enough data
    min_bars_available = min(len(df) for df in data_map.values())
    if min_bars_available < required_bars:
        logger.info("Insufficient historical data for Q1/Q2 comparison.")
        return False, None, None
    # Get the last two bars (Q1 and Q2) for each symbol
    q1_q2_data = {}
    for symbol, df in data_map.items():
        if len(df) >= 2:
            # df.iloc[-2] is Q1, df.iloc[-1] is Q2 (current incomplete bar is not included if resampled correctly)
            q1_data = df.iloc[-2]
            q2_data = df.iloc[-1]
            q1_q2_data[symbol] = {'Q1': q1_data, 'Q2': q2_data}
        else:
            logger.warning(f"Not enough bars for {symbol} to perform Q1/Q2 check.")
            return False, None, None # If any symbol lacks data, abort check
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
    if q2_high_primary > q1_high_primary: # Primary symbol breaks its Q1 high in Q2
        if group_type == "move_together":
            # Check if OTHER symbols also broke their Q1 highs
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
                # --- Generate Charts ---
                chart_data_list = asyncio.run(_generate_charts_for_symbols(symbols, timeframe_min))
                # --- Save triggered alert with charts ---
                triggered_data = {
                    "timestamp": datetime.now(timezone.utc).isoformat() + 'Z',
                    "alert_key": alert_key,
                    "alert_details": alert_details,
                    "message": msg,
                    "charts": chart_data_list
                }
                save_triggered_alert_with_charts(triggered_data)
                # ---
                return True, msg, chart_data_list
        elif group_type == "reverse_moving":
            # Check if OTHER symbols failed to break their Q1 *lows*
            others_failed_low = True
            for sym in other_symbols:
                if q1_q2_data[sym]['Q2']['low'] <= q1_q2_data[sym]['Q1']['low']: # They broke low (bad!)
                    others_failed_low = False
                    break
            if not others_failed_low:
                 # Optional: Check if they actually broke the Q1 low, making divergence stronger
                 # This is a stricter check. The base requirement is they "can't break their Q1 lows"
                 # which could mean they didn't make a new high *or* they made a new low.
                 # Let's stick to the "can't break Q1 lows" -> they made a new low or equalled it.
                 pass # Condition already met if others_failed_low is False
            if others_failed_low: # They did NOT break their Q1 lows (good for trigger)
                msg = (f"⚠️ Alert Triggered ({group_id}, {timeframe_min}m)!\n"
                       f"Group Type: {group_type}\n"
                       f"{primary_symbol} broke Q1 high ({q1_high_primary:.5f}) in Q2 ({q2_high_primary:.5f}), "
                       f"while others held their Q1 lows.")
                logger.info(f"Triggered: {msg}")
                # --- Generate Charts ---
                chart_data_list = asyncio.run(_generate_charts_for_symbols(symbols, timeframe_min))
                # --- Save triggered alert with charts ---
                triggered_data = {
                    "timestamp": datetime.now(timezone.utc).isoformat() + 'Z',
                    "alert_key": alert_key,
                    "alert_details": alert_details,
                    "message": msg,
                    "charts": chart_data_list
                }
                save_triggered_alert_with_charts(triggered_data)
                # ---
                return True, msg, chart_data_list
    # Low Break Scenario
    q1_low_primary = q1_q2_data[primary_symbol]['Q1']['low']
    q2_low_primary = q1_q2_data[primary_symbol]['Q2']['low']
    if q2_low_primary < q1_low_primary: # Primary symbol breaks its Q1 low in Q2
        if group_type == "move_together":
            # Check if OTHER symbols also broke their Q1 lows
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
                # --- Generate Charts ---
                chart_data_list = asyncio.run(_generate_charts_for_symbols(symbols, timeframe_min))
                # --- Save triggered alert with charts ---
                triggered_data = {
                    "timestamp": datetime.now(timezone.utc).isoformat() + 'Z',
                    "alert_key": alert_key,
                    "alert_details": alert_details,
                    "message": msg,
                    "charts": chart_data_list
                }
                save_triggered_alert_with_charts(triggered_data)
                # ---
                return True, msg, chart_data_list
        elif group_type == "reverse_moving":
            # Check if OTHER symbols failed to break their Q1 *highs*
            others_failed_high = True
            for sym in other_symbols:
                if q1_q2_data[sym]['Q2']['high'] >= q1_q2_data[sym]['Q1']['high']: # They broke high (bad!)
                    others_failed_high = False
                    break
            if not others_failed_high:
                # Similar stricter check possible here if needed
                pass
            if others_failed_high: # They did NOT break their Q1 highs (good for trigger)
                msg = (f"⚠️ Alert Triggered ({group_id}, {timeframe_min}m)!\n"
                       f"Group Type: {group_type}\n"
                       f"{primary_symbol} broke Q1 low ({q1_low_primary:.5f}) in Q2 ({q2_low_primary:.5f}), "
                       f"while others held their Q1 highs.")
                logger.info(f"Triggered: {msg}")
                # --- Generate Charts ---
                chart_data_list = asyncio.run(_generate_charts_for_symbols(symbols, timeframe_min))
                # --- Save triggered alert with charts ---
                triggered_data = {
                    "timestamp": datetime.now(timezone.utc).isoformat() + 'Z',
                    "alert_key": alert_key,
                    "alert_details": alert_details,
                    "message": msg,
                    "charts": chart_data_list
                }
                save_triggered_alert_with_charts(triggered_data)
                # ---
                return True, msg, chart_data_list
    logger.debug("Alert conditions not met.")
    return False, None, None
