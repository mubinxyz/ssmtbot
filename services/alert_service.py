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
# --- Import for notifications ---
from services.notify_user import send_alert_notification

# --- Configuration ---
ALERTS_STORE_FILE = "alerts_store.json"
TIMEZONE_LITEFINANCE = pytz.timezone('Etc/GMT-3')  # UTC+3
TIMEZONE_TARGET = pytz.timezone('Asia/Tehran')     # UTC+3:30
TIMEZONE_UTC = pytz.utc

logger = logging.getLogger(__name__)
# --- Force DEBUG level for this module (optional, remove if not needed) ---
# logger.setLevel(logging.DEBUG)
# logger.propagate = True

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
    Converts LiteFinance datetimes (assumed UTC+3) to target timezone (UTC+3:30).
    LiteFinance datetimes are expected to be in a 'datetime' column and tz-aware UTC.
    This function adds/updates a 'timestamp' column (epoch seconds in TARGET_TZ).
    """
    if df.empty:
        logger.debug("_convert_lf_time_to_target: Received empty DataFrame.")
        # Ensure 'timestamp' column exists even if empty
        if 'timestamp' not in df.columns:
            df['timestamp'] = pd.Series(dtype='int64')
        return df

    if 'datetime' not in df.columns:
        logger.warning("_convert_lf_time_to_target: 'datetime' column missing from DataFrame.")
        # Ensure 'timestamp' column exists
        if 'timestamp' not in df.columns:
            df['timestamp'] = pd.Series(dtype='int64')
        return df

    try:
        # Ensure 'datetime' column is datetime and tz-aware UTC
        df_datetime = pd.to_datetime(df['datetime'], utc=True)
        if df_datetime.dt.tz is None:
            df_datetime = df_datetime.dt.tz_localize('UTC')
        elif df_datetime.dt.tz != pytz.utc:
            df_datetime = df_datetime.dt.tz_convert('UTC')

        # Convert to LiteFinance timezone (UTC+3) conceptually
        df_datetime_utc3 = df_datetime.dt.tz_convert(TIMEZONE_LITEFINANCE)
        # Convert to target timezone (UTC+3:30)
        df_datetime_target = df_datetime_utc3.dt.tz_convert(TIMEZONE_TARGET)
        # Add/update the 'timestamp' column to be epoch seconds in the target timezone
        df['timestamp'] = df_datetime_target.apply(lambda dt: int(dt.timestamp()))
        return df
    except Exception as e:
        logger.error(f"_convert_lf_time_to_target: Error processing datetime column: {e}")
        # Ensure 'timestamp' column exists on error
        if 'timestamp' not in df.columns:
            df['timestamp'] = pd.Series(dtype='int64')
        return df

def _get_data_for_alert_checking(symbols: list[str], timeframe_min: int) -> dict[str, pd.DataFrame]:
    """
    Fetches and preprocesses data for alert checking based on timeframe.
    """
    data_map = {}
    now_target = datetime.now(TIMEZONE_TARGET)

    # Determine the lookback period and data resolution based on alert timeframe
    if timeframe_min == 5:
        # 5m alerts: 
        period_minutes = 0.146 * 24 * 60  # ~3.5 hours
        data_resolution = 1
    elif timeframe_min == 15:
        # 15m alerts: Check previous ~16.8 hours, fetch 1m candles
        period_minutes = 0.7 * 24 * 60  
        data_resolution = 1
    elif timeframe_min == 60:  # 1h
        # 1h alerts: Check previous 2.5 days, fetch 5m candles
        period_minutes = 2.5 * 24 * 60  
        data_resolution = 5
    elif timeframe_min == 240:  # 4h
        # 4h alerts: Check previous 17.5 days, fetch 5m candles
        period_minutes = 17.5 * 24 * 60
        data_resolution = 5
    else:
        # Default: use 1m data for shorter timeframes, 5m for longer ones
        data_resolution = 1 if timeframe_min <= 15 else 5
        period_minutes = 4 * timeframe_min

    # Calculate time range
    from_time_target = now_target - timedelta(minutes=period_minutes)
    
    # Convert to LiteFinance timezone (UTC+3) for API request
    from_time_lf = from_time_target.astimezone(TIMEZONE_LITEFINANCE)
    to_time_lf = now_target.astimezone(TIMEZONE_LITEFINANCE)

    from_timestamp_lf = int(from_time_lf.timestamp())
    to_timestamp_lf = int(to_time_lf.timestamp())

    for symbol in symbols:
        logger.debug(f"Fetching {data_resolution}min data for {symbol} from {from_timestamp_lf} to {to_timestamp_lf}")
        df = get_ohlc(symbol, timeframe=data_resolution, from_date=from_timestamp_lf, to_date=to_timestamp_lf)
        if not df.empty:
            # Ensure 'timestamp' column exists for subsequent processing
            df = _convert_lf_time_to_target(df)
            data_map[symbol] = df
        else:
            logger.warning(f"Failed to fetch data for {symbol}")
    return data_map

def _resample_to_alert_timeframe(df_1min_or_5min: pd.DataFrame, alert_timeframe_min: int) -> pd.DataFrame:
    """
    Resamples 1min or 5min data to the alert's timeframe.
    Assumes the input DataFrame has 'datetime' (tz-aware UTC) and 'open', 'high', 'low', 'close' columns.
    Adds/ensures a 'timestamp' column (epoch seconds, assumed to be in TARGET_TZ from _convert_lf_time_to_target).
    """
    if df_1min_or_5min.empty:
        # Return empty DataFrame with expected columns
        return pd.DataFrame(columns=['datetime', 'open', 'high', 'low', 'close', 'timestamp'])

    # Work on a copy
    df_resample = df_1min_or_5min.copy()

    # Ensure 'datetime' is datetime and tz-aware UTC (as expected from get_ohlc/chart_service)
    if 'datetime' not in df_resample.columns:
         logger.error("_resample_to_alert_timeframe: Input DataFrame missing 'datetime' column.")
         return pd.DataFrame() # Return empty DataFrame on critical error

    try:
        df_resample['datetime'] = pd.to_datetime(df_resample['datetime'], utc=True)
        if df_resample['datetime'].dt.tz is None:
            df_resample['datetime'] = df_resample['datetime'].dt.tz_localize('UTC')
    except Exception as e:
         logger.error(f"_resample_to_alert_timeframe: Error parsing 'datetime' column: {e}")
         return pd.DataFrame()

    # Set 'datetime' as the index for resampling
    df_resample.set_index('datetime', inplace=True)

    # Perform resampling based on the alert timeframe
    try:
        # Use 'min' instead of deprecated 'T'
        resampled_df = df_resample.resample(f'{alert_timeframe_min}min', label='right', closed='right').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            # Aggregate timestamp: take the last one from the resampled period
            # This timestamp should correspond to the end of the resampled bar
            # and should already be in target tz epoch seconds from _convert_lf_time_to_target
            'timestamp': 'last'
        }).dropna()
    except Exception as e:
        logger.error(f"_resample_to_alert_timeframe: Error during resampling for {alert_timeframe_min}min: {e}")
        return pd.DataFrame() # Return empty DataFrame on resampling error

    # Reset index to make 'datetime' a column again
    resampled_df.reset_index(inplace=True)
    # Rename columns appropriately if needed (chart_service expects 'datetime' as index, but we need it as column here)
    # The resampling preserves the 'datetime' index name, which becomes a column on reset_index()

    # Ensure no rows with missing critical data
    resampled_df.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)

    # Ensure 'timestamp' column exists and is valid
    if 'timestamp' not in resampled_df.columns or resampled_df['timestamp'].isna().all():
         logger.warning("_resample_to_alert_timeframe: 'timestamp' column missing or all NaN after resampling. Recreating from 'datetime'.")
         # Convert the 'datetime' (which is tz-aware UTC) back to target timezone and then to epoch seconds
         try:
             resampled_df['timestamp'] = resampled_df['datetime'].dt.tz_convert(TIMEZONE_TARGET).apply(lambda dt: int(dt.timestamp()))
         except Exception as e:
             logger.error(f"_resample_to_alert_timeframe: Error recreating 'timestamp' column: {e}")
             # Add empty timestamp column if recreation fails
             if 'timestamp' not in resampled_df.columns:
                 resampled_df['timestamp'] = pd.Series(dtype='int64')

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

# --- Helper for Quarter Index (Copied from chart_service.py) ---
def _quarter_index_from_boundary(boundary_utc: pd.Timestamp, tz_name: str = "Asia/Tehran", timeframe: int = 15) -> int:
    """
    Given a boundary timestamp in UTC, return the quarter index (0..3) based
    on its local time in tz_name (Asia/Tehran) and the timeframe.

    For 5m:  each 15m-quarter is divided into four 90-minute segments:
           q1 = start + 0..90min, q2 = start+90..start+180, ...
    For 15m: Q1=01:30, Q2=07:30, Q3=13:30, Q4=19:30
    For 1h:  Q1=MON, Q2=TUE, Q3=WED, Q4=THU
    For 4h:  Q1=WEEK1, Q2=WEEK2, Q3=WEEK3, Q4=WEEK4
    """
    if boundary_utc is None:
        return 0

    try:
        if getattr(boundary_utc, "tzinfo", None) is None:
            b_utc = pd.Timestamp(boundary_utc).tz_localize("UTC")
        else:
            b_utc = pd.Timestamp(boundary_utc).tz_convert("UTC")
        # Import dateutil.tz here or ensure it's available
        from dateutil import tz as dateutil_tz
        local = b_utc.tz_convert(tz_name)

        if timeframe == 5:
            # Determine which 90-minute segment within the 15m quarter
            h = local.hour
            m = local.minute
            mins = h * 60 + m

            def mm(hh, mm_): return hh * 60 + mm_
            q1_start = mm(1, 30)
            q2_start = mm(7, 30)
            q3_start = mm(13, 30)
            q4_start = mm(19, 30)

            if q1_start <= mins < q2_start:
                quarter_start = q1_start
            elif q2_start <= mins < q3_start:
                quarter_start = q2_start
            elif q3_start <= mins < q4_start:
                quarter_start = q3_start
            else:
                quarter_start = q4_start

            # segment: 0..3 each = 90 minutes
            segment_position = (mins - quarter_start) // 90
            return int(segment_position % 4)

        elif timeframe == 15:
            h = local.hour
            m = local.minute
            mins = h * 60 + m
            def mm(hh, mm_): return hh * 60 + mm_
            q1_start = mm(1, 30)
            q2_start = mm(7, 30)
            q3_start = mm(13, 30)
            q4_start = mm(19, 30)
            if q1_start <= mins < q2_start:
                return 0
            if q2_start <= mins < q3_start:
                return 1
            if q3_start <= mins < q4_start:
                return 2
            return 3

        elif timeframe == 60:
            weekday = local.weekday()
            if weekday == 0:  # Monday
                return 0
            elif weekday == 1:  # Tuesday
                return 1
            elif weekday == 2:  # Wednesday
                return 2
            elif weekday == 3:  # Thursday
                return 3
            else:
                # Default to Q4 if outside Mon-Thu
                return 3

        elif timeframe == 240:
            # Week of month calculation
            week_of_month = (local.day - 1) // 7 + 1
            week_of_month = min(week_of_month, 4)
            return week_of_month - 1

        else:
            # Default to Q1 for unknown timeframes
            return 0
    except Exception as e:
        logger.error(f"_quarter_index_from_boundary error: {e}")
        return 0 # Default to Q1 on error

def _check_all_quarter_pairs(data_map: dict, group_id: str, group_type: str, primary_symbol: str, other_symbols: list, timeframe_min: int):
    """
    Check the most recent quarter pair for alert conditions.
    Returns list of triggered alerts [(is_triggered, message, chart_data_list), ...]
    """
    triggered_alerts = []

    # Ensure all symbols have data and same length
    lengths = [len(df) for df in data_map.values()] if data_map else [0]
    if not lengths or min(lengths) < 2:
        logger.info("Insufficient historical data for quarter comparison (need at least 2 data points).")
        return triggered_alerts

    # --- Only check the most recent pair ---
    # The last two data points are at indices -2 (Q1) and -1 (Q2)
    q1_index = -2
    q2_index = -1
    # min_length = min(lengths) # Not used anymore for numbering

    # Get the Q1 and Q2 data for each symbol
    q1_q2_data = {}
    valid_data = True

    for symbol in [primary_symbol] + other_symbols: # Include primary symbol first
        df = data_map.get(symbol)
        if df is None or len(df) < 2: # Need at least 2 points
             logger.warning(f"_check_all_quarter_pairs: Insufficient data for symbol {symbol}")
             valid_data = False
             break
        try:
            q1_data = df.iloc[q1_index] # -2 (Second to last bar)
            q2_data = df.iloc[q2_index] # -1 (Last bar)
            q1_q2_data[symbol] = {'Q1': q1_data, 'Q2': q2_data}
        except IndexError as e:
             logger.warning(f"_check_all_quarter_pairs: IndexError accessing data for symbol {symbol} at indices {q1_index}, {q2_index}: {e}")
             valid_data = False
             break

    # --- Corrected line below ---
    if not valid_data or not q1_q2_data:
        return triggered_alerts # Return empty list if data is bad

    # --- Get the end times of Q1 and Q2 bars to determine their quarter labels ---
    # The 'datetime' column in the resampled data represents the *end* time of the bar.
    # We use the end time to determine which quarter the bar *ended in*.
    q1_end_time_utc = None
    q2_end_time_utc = None
    try:
        # Get Q1 and Q2 bar data for primary symbol (timing should be consistent)
        q1_bar_data = q1_q2_data[primary_symbol]['Q1']
        q2_bar_data = q1_q2_data[primary_symbol]['Q2']

        # Ensure 'datetime' is datetime and tz-aware UTC
        q1_end_time_raw = q1_bar_data['datetime']
        q2_end_time_raw = q2_bar_data['datetime']

        q1_end_time_utc = pd.to_datetime(q1_end_time_raw, utc=True) # Q1 end time
        q2_end_time_utc = pd.to_datetime(q2_end_time_raw, utc=True) # Q2 end time

        logger.debug(f"Q1 Bar End Time (UTC): {q1_end_time_utc}")
        logger.debug(f"Q2 Bar End Time (UTC): {q2_end_time_utc}")

    except Exception as e:
        logger.error(f"_check_all_quarter_pairs: Error getting bar end times: {e}", exc_info=True)

    # Check conditions for this most recent pair
    # Pass the END times to determine correct quarter names for labeling
    triggered_result = _check_single_quarter_pair(
        q1_q2_data, group_id, group_type, primary_symbol, other_symbols,
        timeframe_min, q1_end_time_utc, q2_end_time_utc # Pass END times for labeling
    )

    if triggered_result[0]:  # If triggered
        triggered_alerts.append(triggered_result)

    return triggered_alerts

# --- CORRECTED _check_single_quarter_pair function (OR semantics for 'others') ---
def _check_single_quarter_pair(q1_q2_data: dict, group_id: str, group_type: str,
                              primary_symbol: str, other_symbols: list, timeframe_min: int,
                              q1_end_time_utc: pd.Timestamp, q2_end_time_utc: pd.Timestamp):
    """
    Check a single quarter pair (Q1 vs Q2) for alert conditions.
    q1_end_time_utc: End time of the Q1 bar (benchmark bar).
    q2_end_time_utc: End time of the Q2 bar (action bar).
    The message labels Q1 and Q2 refer to the position of the bars in the comparison,
    not necessarily the time-based quarter names from chart_service.

    NOTE: For reverse_moving, the "others" check uses OR: the alert triggers if
    ANY other symbol held the required side. For move_together, the logic
    triggers if ANY other did *not* follow (same as before).
    """
    try:
        q1_label = "Q1"
        q2_label = "Q2"

        # High Break Scenario (primary high break)
        q1_high_primary = float(q1_q2_data[primary_symbol]['Q1']['high'])
        q2_high_primary = float(q1_q2_data[primary_symbol]['Q2']['high'])

        if q2_high_primary > q1_high_primary:
            if group_type == "move_together":
                # move_together: trigger if any other DID NOT break its high
                others_broke_high = True
                for sym in other_symbols:
                    sym_q1_high = float(q1_q2_data[sym]['Q1']['high'])
                    sym_q2_high = float(q1_q2_data[sym]['Q2']['high'])
                    if sym_q2_high <= sym_q1_high:
                        others_broke_high = False
                        break
                if not others_broke_high:
                    msg = (f"⚠️ Alert Triggered ({group_id}, {timeframe_min}m)!\n"
                           f"Group Type: {group_type}\n"
                           f"Quarter Pair: {q1_label}/{q2_label}\n"
                           f"{primary_symbol} broke {q1_label} high ({q1_high_primary:.5f}) in {q2_label} ({q2_high_primary:.5f}), "
                           f"but not all others followed.")
                    logger.info(f"Triggered: {msg}")
                    return True, msg, None

            elif group_type == "reverse_moving":
                # reverse_moving (OR): trigger if ANY other held their Q1 low (i.e., did NOT break low)
                any_other_held_low = False
                for sym in other_symbols:
                    sym_q1_low = float(q1_q2_data[sym]['Q1']['low'])
                    sym_q2_low = float(q1_q2_data[sym]['Q2']['low'])
                    # If this other didn't break its low (sym_q2_low >= sym_q1_low), mark true and stop
                    if sym_q2_low >= sym_q1_low:
                        any_other_held_low = True
                        break
                if any_other_held_low:
                    msg = (f"⚠️ Alert Triggered ({group_id}, {timeframe_min}m)!\n"
                           f"Group Type: {group_type}\n"
                           f"Quarter Pair: {q1_label}/{q2_label}\n"
                           f"{primary_symbol} broke {q1_label} high ({q1_high_primary:.5f}) in {q2_label} ({q2_high_primary:.5f}), "
                           f"while at least one other held their {q1_label} low.")
                    logger.info(f"Triggered: {msg}")
                    return True, msg, None

        # Low Break Scenario (primary low break)
        q1_low_primary = float(q1_q2_data[primary_symbol]['Q1']['low'])
        q2_low_primary = float(q1_q2_data[primary_symbol]['Q2']['low'])

        if q2_low_primary < q1_low_primary:
            if group_type == "move_together":
                # move_together: trigger if any other DID NOT break its low
                others_broke_low = True
                for sym in other_symbols:
                    sym_q1_low = float(q1_q2_data[sym]['Q1']['low'])
                    sym_q2_low = float(q1_q2_data[sym]['Q2']['low'])
                    if sym_q2_low >= sym_q1_low:
                        others_broke_low = False
                        break
                if not others_broke_low:
                    msg = (f"⚠️ Alert Triggered ({group_id}, {timeframe_min}m)!\n"
                           f"Group Type: {group_type}\n"
                           f"Quarter Pair: {q1_label}/{q2_label}\n"
                           f"{primary_symbol} broke {q1_label} low ({q1_low_primary:.5f}) in {q2_label} ({q2_low_primary:.5f}), "
                           f"but not all others followed.")
                    logger.info(f"Triggered: {msg}")
                    return True, msg, None

            elif group_type == "reverse_moving":
                # reverse_moving (OR): trigger if ANY other held their Q1 high (i.e., did NOT break high)
                any_other_held_high = False
                for sym in other_symbols:
                    sym_q1_high = float(q1_q2_data[sym]['Q1']['high'])
                    sym_q2_high = float(q1_q2_data[sym]['Q2']['high'])
                    # If this other didn't break its high (sym_q2_high >= sym_q1_high), mark true and stop
                    if sym_q2_high >= sym_q1_high:
                        any_other_held_high = True
                        break
                if any_other_held_high:
                    msg = (f"⚠️ Alert Triggered ({group_id}, {timeframe_min}m)!\n"
                           f"Group Type: {group_type}\n"
                           f"Quarter Pair: {q1_label}/{q2_label}\n"
                           f"{primary_symbol} broke {q1_label} low ({q1_low_primary:.5f}) in {q2_label} ({q2_low_primary:.5f}), "
                           f"while at least one other held their {q1_label} high.")
                    logger.info(f"Triggered: {msg}")
                    return True, msg, None

    except (KeyError, ValueError, TypeError) as e:
        logger.error(f"_check_single_quarter_pair: Error processing data for pair: {e}")

    return False, None, None
# --- END OF CORRECTED _check_single_quarter_pair function ---

async def check_alert_conditions(alert_key: str, alert_details: dict):
    """
    Checks if the conditions for a specific alert are met.
    Returns a tuple (is_triggered: bool, message: str | None, chart_data_list: list | None).
    For 5m timeframe, checks both time windows separately.
    For other timeframes, checks all consecutive quarter pairs.
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

    # Special handling for 5-minute alerts - time window checking
    if timeframe_min == 5:
        now_target = datetime.now(TIMEZONE_TARGET)
        today = now_target.date()

        # Define the two time windows
        start_time_1 = TIMEZONE_TARGET.localize(datetime.combine(today, datetime.min.time().replace(hour=1, minute=30)))
        end_time_1 = TIMEZONE_TARGET.localize(datetime.combine(today, datetime.min.time().replace(hour=7, minute=30)))
        start_time_2 = TIMEZONE_TARGET.localize(datetime.combine(today, datetime.min.time().replace(hour=7, minute=30)))
        end_time_2 = TIMEZONE_TARGET.localize(datetime.combine(today, datetime.min.time().replace(hour=13, minute=30)))

        # Check which window we're in and process accordingly
        # Only process if within the defined windows
        if start_time_1 <= now_target < end_time_1:
            # Process first window (1:30-7:30)
            logger.info("Processing 5m alert for first window (1:30-7:30 UTC+3:30)")
            triggered_results = await _process_5m_window(alert_details, symbols, group_id, group_type, user_id, timeframe_min, 1)
        elif start_time_2 <= now_target < end_time_2:
            # Process second window (7:30-13:30)
            logger.info("Processing 5m alert for second window (7:30-13:30 UTC+3:30)")
            triggered_results = await _process_5m_window(alert_details, symbols, group_id, group_type, user_id, timeframe_min, 2)
        else:
            logger.info("Outside 5m alert windows (1:30-7:30 or 7:30-13:30 Tehran time), no processing")
            return False, None, None

        # Return first triggered result if any
        if triggered_results:
            # The results from _process_5m_window should already include chart_data_list
            return triggered_results[0]
        return False, None, None

    else:
        # Regular processing for other timeframes (15m, 1h, 4h)
        # 1. Fetch data
        data_map_raw = _get_data_for_alert_checking(symbols, timeframe_min)
        if not data_map_raw:
            logger.error("No data fetched for alert check.")
            return False, None, None

        # 2. Resample data to alert timeframe
        # Ensure all symbols have data after fetching
        if len(data_map_raw) != len(symbols):
             missing_symbols = set(symbols) - set(data_map_raw.keys())
             logger.error(f"Failed to fetch data for symbols: {missing_symbols}")
             return False, None, None

        data_map = {}
        for symbol, df in data_map_raw.items():
            if df.empty:
                 logger.warning(f"Data for {symbol} is empty after fetching.")
                 continue # Skip empty DataFrames
            resampled_df = _resample_to_alert_timeframe(df, timeframe_min)
            if not resampled_df.empty:
                data_map[symbol] = resampled_df
            else:
                logger.warning(f"Resampling failed or resulted in empty data for {symbol}")

        # Ensure all symbols have data after resampling
        if len(data_map) != len(symbols):
             logger.error(f"Insufficient data after resampling for alert check. Expected {len(symbols)}, got {len(data_map)}.")
             return False, None, None

        # 3. Check the most recent consecutive quarter pair
        primary_symbol = symbols[0]
        other_symbols = symbols[1:]

        triggered_results = _check_all_quarter_pairs(
            data_map, group_id, group_type, primary_symbol, other_symbols, timeframe_min
        )

        # Process triggered alerts (send notifications, generate charts)
        final_results = []
        for i, (is_triggered, message, _) in enumerate(triggered_results):
            if is_triggered and message: # Ensure message is not None
                # Generate charts for triggered alerts
                try:
                    chart_data_list = await _generate_charts_for_symbols(symbols, timeframe_min)
                except Exception as chart_error:
                     logger.error(f"Error generating charts for triggered alert: {chart_error}")
                     chart_data_list = [{"error": f"Chart generation failed: {chart_error}"}]

                triggered_data = {
                    "timestamp": datetime.now(timezone.utc).isoformat() + 'Z',
                    "alert_key": alert_key,
                    "alert_details": alert_details,
                    "message": message,
                    "charts": chart_data_list
                }
                try:
                    save_triggered_alert_with_charts(triggered_data)
                except Exception as save_error:
                     logger.error(f"Error saving triggered alert: {save_error}")

                # Send notification to user
                try:
                    await send_alert_notification(user_id, message, chart_data_list)
                except Exception as notify_error:
                     logger.error(f"Error sending notification for triggered alert: {notify_error}")

                final_results.append((True, message, chart_data_list))

        # Return first triggered result if any
        if final_results:
            return final_results[0]

        logger.debug("Alert conditions not met for the most recent quarter pair.")
        return False, None, None

async def _process_5m_window(alert_details: dict, symbols: list, group_id: str, group_type: str,
                            user_id: int, timeframe_min: int, window_number: int):
    """
    Process 5-minute alerts for specific time windows.
    """
    # Fetch data for the specific window (this handles the window time logic)
    data_map_raw = _get_data_for_alert_checking(symbols, timeframe_min)
    if not data_map_raw:
        logger.error("No data fetched for 5m alert check in window.")
        return []

    # Ensure all symbols have data after fetching
    if len(data_map_raw) != len(symbols):
         missing_symbols = set(symbols) - set(data_map_raw.keys())
         logger.error(f"_process_5m_window: Failed to fetch data for symbols in window {window_number}: {missing_symbols}")
         return []

    # Resample data
    data_map = {}
    for symbol, df in data_map_raw.items():
        if df.empty:
             logger.warning(f"_process_5m_window: Data for {symbol} is empty in window {window_number}.")
             continue
        resampled_df = _resample_to_alert_timeframe(df, timeframe_min)
        if not resampled_df.empty:
            data_map[symbol] = resampled_df
        else:
            logger.warning(f"_process_5m_window: Resampling failed for {symbol} in window {window_number}")

    if not data_map or len(data_map) != len(symbols):
        logger.error(f"Insufficient data for 5m window {window_number} processing.")
        return []

    # Check the most recent consecutive quarter pair for this window
    primary_symbol = symbols[0]
    other_symbols = symbols[1:]

    triggered_results = _check_all_quarter_pairs(
        data_map, group_id, group_type, primary_symbol, other_symbols, timeframe_min
    )

    # Process triggered alerts for the 5m window
    final_results = []
    for i, (is_triggered, message, _) in enumerate(triggered_results):
        if is_triggered and message:
            # Generate charts
            try:
                chart_data_list = await _generate_charts_for_symbols(symbols, timeframe_min)
            except Exception as chart_error:
                 logger.error(f"_process_5m_window: Error generating charts: {chart_error}")
                 chart_data_list = [{"error": f"Chart generation failed: {chart_error}"}]

            triggered_data = {
                "timestamp": datetime.now(timezone.utc).isoformat() + 'Z',
                "alert_key": f"{alert_details['user_id']}::{alert_details['category']}::{group_id}::{timeframe_min}",
                "alert_details": alert_details,
                "message": message,
                "charts": chart_data_list,
                "window": window_number
            }
            try:
                save_triggered_alert_with_charts(triggered_data)
            except Exception as save_error:
                 logger.error(f"_process_5m_window: Error saving triggered alert: {save_error}")

            # Send notification
            try:
                await send_alert_notification(user_id, message, chart_data_list)
            except Exception as notify_error:
                 logger.error(f"_process_5m_window: Error sending notification: {notify_error}")

            final_results.append((True, message, chart_data_list))

    return final_results
