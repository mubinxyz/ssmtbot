# services/alert_service.py

import json
import logging
import os
import pandas as pd
from datetime import datetime, timedelta, timezone
import pytz
from utils.get_data import get_ohlc
from utils.normalize_data import normalize_symbol
from services.chart_service import generate_chart
import asyncio
import inspect
import threading
from typing import Optional

# Use the notify_user send function that we updated (it accepts optional bot)
from services.notify_user import send_alert_notification

# --- Configuration ---
ALERTS_STORE_FILE = "alerts_store.json"
TRIGGERED_ALERTS_WITH_CHARTS_FILE = "triggered_alerts_with_charts.json"
CHARTS_OUTPUT_DIR = "charts"  # If you want to persist charts to disk (optional)

TIMEZONE_LITEFINANCE = pytz.timezone('Etc/GMT-3')  # UTC+3
TIMEZONE_TARGET = pytz.timezone('Asia/Tehran')     # UTC+3:30
TIMEZONE_UTC = pytz.utc

# Concurrency/timeouts for fetching & charting
_SYMBOL_FETCH_CONCURRENCY = 6
_PER_SYMBOL_FETCH_TIMEOUT = 18.0  # seconds per symbol for get_ohlc
_PER_ALERT_FETCH_TIMEOUT = 40.0   # total time allowed to fetch & resample for an alert
_CHART_GENERATION_TIMEOUT = 40.0  # seconds (if generate_chart is blocking we use to_thread)
_SAVE_IO_TIMEOUT = 10.0           # seconds for saving triggered alerts

logger = logging.getLogger(__name__)


# --------------------------
# Alert storage helpers
# --------------------------

def load_alerts():
    if not os.path.exists(ALERTS_STORE_FILE):
        return {}
    try:
        with open(ALERTS_STORE_FILE, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"Error loading alerts from {ALERTS_STORE_FILE}: {e}")
        return {}


def save_alerts(alerts):
    try:
        with open(ALERTS_STORE_FILE, 'w') as f:
            json.dump(alerts, f, indent=2)
    except IOError as e:
        logger.error(f"Error saving alerts to {ALERTS_STORE_FILE}: {e}")


def load_triggered_alerts_with_charts():
    if not os.path.exists(TRIGGERED_ALERTS_WITH_CHARTS_FILE):
        return []
    try:
        with open(TRIGGERED_ALERTS_WITH_CHARTS_FILE, 'r') as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"Error loading triggered alerts from {TRIGGERED_ALERTS_WITH_CHARTS_FILE}: {e}")
        return []


def save_triggered_alert_with_charts(triggered_alert_data):
    all_triggered = load_triggered_alerts_with_charts()
    all_triggered.append(triggered_alert_data)
    try:
        with open(TRIGGERED_ALERTS_WITH_CHARTS_FILE, 'w') as f:
            json.dump(all_triggered, f, indent=2)
    except IOError as e:
        logger.error(f"Error saving triggered alert to {TRIGGERED_ALERTS_WITH_CHARTS_FILE}: {e}")


# --------------------------
# Alert definitions & groups
# --------------------------

def _generate_alert_key(user_id, category, group_id, timeframe_min):
    return f"{user_id}::{category}::{group_id}::{timeframe_min}"


def set_ssmt_alert(user_id: int, group_id: str, timeframe_min: int, category: str = "FOREX CURRENCIES"):
    alerts = load_alerts()
    key = _generate_alert_key(user_id, category, group_id, timeframe_min)
    if key in alerts:
        alerts[key]["active"] = True
        alerts[key]["updated_at"] = datetime.now(timezone.utc).isoformat() + 'Z'
    else:
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
    alerts = load_alerts()
    key = _generate_alert_key(user_id, category, group_id, timeframe_min)
    if key in alerts and alerts[key].get("active"):
        alerts[key]["active"] = False
        alerts[key]["updated_at"] = datetime.now(timezone.utc).isoformat() + 'Z'
        save_alerts(alerts)
        logger.info(f"Alert deactivated for user {user_id}, group {group_id}, timeframe {timeframe_min}min.")
        return True
    return False


def get_active_alerts():
    alerts = load_alerts()
    return {k: v for k, v in alerts.items() if v.get("active", False)}

GROUP_CHARACTERISTICS = {
    "test_move_group": {"label": "Test Move Together", "symbols": ["AAA", "BBB", "CCC"], "type": "move_together"},
    "test_reverse_group": {"label": "Test Reverse Moving", "symbols": ["DDD", "EEE", "FFF"], "type": "reverse_moving"},
    "dxy_eu_gu": {"label": "DXY / EURUSD / GBPUSD", "symbols": ["USDX", "EURUSD", "GBPUSD"], "type": "reverse_moving"},
    "btc_eth_xrp": {"label": "BTCUSD / ETHUSD / XRPUSD", "symbols": ["BTCUSD", "ETHUSD", "XRPUSD"], "type": "move_together"},
    "spx_nq_ym": {"label": "S&P500 (SPX) / NASDAQ (NQ) / DOW (YM)", "symbols": ["SPX", "NQ", "YM"], "type": "move_together"},
    "dxy_xau_xag_aud": {"label": "DXY / XAU / XAG / AUD", "symbols": ["USDX", "XAU", "XAG", "AUD"], "type": "reverse_moving"}
}

def find_group(category: str, group_id: str):
    if category == "FOREX CURRENCIES" and group_id in GROUP_CHARACTERISTICS:
        return GROUP_CHARACTERISTICS[group_id]
    return None


# --------------------------
# Time conversion & resampling (CPU-bound) - run in threads
# --------------------------

def _convert_lf_time_to_target(df: pd.DataFrame) -> pd.DataFrame:
    # Same logic as before, keep it sync but we'll call it via to_thread
    if df.empty:
        if 'timestamp' not in df.columns:
            df['timestamp'] = pd.Series(dtype='int64')
        return df
    if 'datetime' not in df.columns:
        if 'timestamp' not in df.columns:
            df['timestamp'] = pd.Series(dtype='int64')
        return df
    try:
        df_datetime = pd.to_datetime(df['datetime'], utc=True)
        if df_datetime.dt.tz is None:
            df_datetime = df_datetime.dt.tz_localize('UTC')
        elif df_datetime.dt.tz != pytz.utc:
            df_datetime = df_datetime.dt.tz_convert('UTC')
        df_datetime_utc3 = df_datetime.dt.tz_convert(TIMEZONE_LITEFINANCE)
        df_datetime_target = df_datetime_utc3.dt.tz_convert(TIMEZONE_TARGET)
        df['timestamp'] = df_datetime_target.apply(lambda dt: int(dt.timestamp()))
        return df
    except Exception as e:
        logger.error(f"_convert_lf_time_to_target: Error: {e}")
        if 'timestamp' not in df.columns:
            df['timestamp'] = pd.Series(dtype='int64')
        return df


def _resample_to_alert_timeframe(df_1min_or_5min: pd.DataFrame, alert_timeframe_min: int) -> pd.DataFrame:
    if df_1min_or_5min.empty:
        return pd.DataFrame(columns=['datetime', 'open', 'high', 'low', 'close', 'timestamp'])
    df_resample = df_1min_or_5min.copy()
    if 'datetime' not in df_resample.columns:
         logger.error("_resample_to_alert_timeframe: missing 'datetime'.")
         return pd.DataFrame()
    try:
        df_resample['datetime'] = pd.to_datetime(df_resample['datetime'], utc=True)
        if df_resample['datetime'].dt.tz is None:
            df_resample['datetime'] = df_resample['datetime'].dt.tz_localize('UTC')
    except Exception as e:
         logger.error(f"_resample_to_alert_timeframe: Error parsing 'datetime': {e}")
         return pd.DataFrame()
    df_resample.set_index('datetime', inplace=True)
    try:
        resampled_df = df_resample.resample(f'{alert_timeframe_min}min', label='right', closed='right').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'timestamp': 'last'
        }).dropna()
    except Exception as e:
        logger.error(f"_resample_to_alert_timeframe: Error during resample: {e}")
        return pd.DataFrame()
    resampled_df.reset_index(inplace=True)
    resampled_df.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)
    if 'timestamp' not in resampled_df.columns or resampled_df['timestamp'].isna().all():
         try:
            resampled_df['timestamp'] = resampled_df['datetime'].dt.tz_convert(TIMEZONE_TARGET).apply(lambda dt: int(dt.timestamp()))
         except Exception as e:
            logger.error(f"_resample_to_alert_timeframe: recreate timestamp error: {e}")
            if 'timestamp' not in resampled_df.columns:
                resampled_df['timestamp'] = pd.Series(dtype='int64')
    return resampled_df


# --------------------------
# Async-safe data fetching: run get_ohlc in threads with timeouts & concurrency
# --------------------------

async def _fetch_symbol_df(symbol: str, data_resolution: int, from_ts: int, to_ts: int, timeout: float) -> Optional[pd.DataFrame]:
    """
    Fetch single symbol in a thread and then convert its times in a thread.
    Returns DataFrame or None on error/timeout.
    """
    try:
        # Run get_ohlc in a thread and wrap with timeout
        coro = asyncio.to_thread(get_ohlc, symbol, data_resolution, from_ts, to_ts)
        df = await asyncio.wait_for(coro, timeout=timeout)
        if df is None or (hasattr(df, 'empty') and df.empty):
            logger.warning("_fetch_symbol_df: get_ohlc returned empty for %s", symbol)
            return None
        # Convert times in a thread (pandas ops)
        df_conv = await asyncio.to_thread(_convert_lf_time_to_target, df)
        return df_conv
    except asyncio.TimeoutError:
        logger.warning("_fetch_symbol_df: timeout fetching %s", symbol)
    except Exception as e:
        logger.exception("_fetch_symbol_df: error fetching %s: %s", symbol, e)
    return None


async def _get_data_for_alert_checking(symbols: list[str], timeframe_min: int) -> dict[str, pd.DataFrame]:
    """
    Async wrapper to fetch multiple symbols concurrently but limited to a semaphore.
    Returns a dict symbol -> DataFrame (only successful ones).
    """
    data_map = {}
    now_target = datetime.now(TIMEZONE_TARGET)

    # determine data_resolution & lookback as before
    if timeframe_min == 5:
        period_minutes = 0.146 * 24 * 60
        data_resolution = 1
    elif timeframe_min == 15:
        period_minutes = 0.7 * 24 * 60
        data_resolution = 1
    elif timeframe_min == 60:
        period_minutes = 2.5 * 24 * 60
        data_resolution = 5
    elif timeframe_min == 240:
        period_minutes = 17.5 * 24 * 60
        data_resolution = 5
    else:
        data_resolution = 1 if timeframe_min <= 15 else 5
        period_minutes = 4 * timeframe_min

    from_time_target = now_target - timedelta(minutes=period_minutes)
    from_time_lf = from_time_target.astimezone(TIMEZONE_LITEFINANCE)
    to_time_lf = now_target.astimezone(TIMEZONE_LITEFINANCE)
    from_timestamp_lf = int(from_time_lf.timestamp())
    to_timestamp_lf = int(to_time_lf.timestamp())

    sem = asyncio.Semaphore(_SYMBOL_FETCH_CONCURRENCY)
    async def _fetch_with_sem(sym):
        async with sem:
            return sym, await _fetch_symbol_df(sym, data_resolution, from_timestamp_lf, to_timestamp_lf, _PER_SYMBOL_FETCH_TIMEOUT)

    pending = [asyncio.create_task(_fetch_with_sem(sym)) for sym in symbols]
    try:
        done, pending2 = await asyncio.wait(pending, timeout=_PER_ALERT_FETCH_TIMEOUT)
        for t in done:
            try:
                sym, df = t.result()
                if df is not None and not df.empty:
                    data_map[sym] = df
            except Exception as e:
                logger.exception("_get_data_for_alert_checking: fetch task exception: %s", e)
        # cancel any remaining pending fetches
        if pending2:
            for t in pending2:
                t.cancel()
            await asyncio.gather(*pending2, return_exceptions=True)
    except Exception as e:
        logger.exception("_get_data_for_alert_checking: unexpected error: %s", e)
        for t in pending:
            if not t.done():
                t.cancel()
        await asyncio.gather(*pending, return_exceptions=True)

    return data_map


# --------------------------
# Chart generation (async-safe)
# --------------------------

async def _generate_charts_for_symbols(symbols: list[str], timeframe_min: int) -> list:
    """
    Attempt to generate charts in a non-blocking way. If generate_chart is async, await it.
    If it's sync, run it in a thread.
    Returns list of dicts: {"symbol":..., "timeframe":..., "buffer": <bytes or file-like>}
    """
    try:
        if inspect.iscoroutinefunction(generate_chart):
            coro = generate_chart(symbols, timeframe=timeframe_min)
            # apply timeout to chart generation
            chart_buffers = await asyncio.wait_for(coro, timeout=_CHART_GENERATION_TIMEOUT)
        else:
            # blocking; run in a thread
            chart_buffers = await asyncio.wait_for(asyncio.to_thread(generate_chart, symbols, timeframe_min), timeout=_CHART_GENERATION_TIMEOUT)

        chart_data_list = []
        # assume generate_chart returns a list/iterable of buffers (bytes or file-like)
        for i, buf in enumerate(chart_buffers):
            chart_data_list.append({
                "symbol": symbols[i] if i < len(symbols) else "unknown",
                "timeframe": timeframe_min,
                "buffer": buf
            })
        return chart_data_list
    except asyncio.TimeoutError:
        logger.warning("_generate_charts_for_symbols: timeout generating charts for %s", symbols)
        return [{"error": "chart generation timeout", "symbol": symbols[i] if i < len(symbols) else "unknown"} for i in range(len(symbols))]
    except Exception as e:
        logger.exception("_generate_charts_for_symbols: error: %s", e)
        return [{"error": f"chart generation failed: {e}", "symbol": symbols[i] if i < len(symbols) else "unknown"} for i in range(len(symbols))]


# --------------------------
# Background task scheduling helpers (safe)
# --------------------------

def _bg_done_callback(task: asyncio.Task, name: str):
    try:
        exc = task.exception()
        if exc:
            logger.exception("Background task '%s' raised: %s", name, exc)
    except asyncio.CancelledError:
        logger.warning("Background task '%s' cancelled", name)
    except Exception as e:
        logger.exception("Error inspecting background task '%s': %s", name, e)


def _schedule_background_task(coro, name: str):
    try:
        task = asyncio.create_task(coro)
        task.add_done_callback(lambda t: _bg_done_callback(t, name))
        return task
    except RuntimeError as e:
        logger.warning("No running loop to create background task '%s': %s", name, e)
        def runner():
            try:
                asyncio.run(coro)
            except Exception as ex:
                logger.exception("Background-thread runner for '%s' errored: %s", name, ex)
        t = threading.Thread(target=runner, daemon=True)
        t.start()
        return t
    except Exception as e:
        logger.exception("Failed scheduling background task '%s': %s", name, e)
        return None


# --------------------------
# Quarter logic helpers (same semantics)
# --------------------------

def _quarter_index_from_boundary(boundary_utc: pd.Timestamp, tz_name: str = "Asia/Tehran", timeframe: int = 15) -> int:
    if boundary_utc is None:
        return 0
    try:
        if getattr(boundary_utc, "tzinfo", None) is None:
            b_utc = pd.Timestamp(boundary_utc).tz_localize("UTC")
        else:
            b_utc = pd.Timestamp(boundary_utc).tz_convert("UTC")
        local = b_utc.tz_convert(tz_name)
        if timeframe == 5:
            h = local.hour; m = local.minute; mins = h * 60 + m
            def mm(hh, mm_): return hh * 60 + mm_
            q1_start = mm(1, 30); q2_start = mm(7, 30); q3_start = mm(13, 30); q4_start = mm(19, 30)
            if q1_start <= mins < q2_start: quarter_start = q1_start
            elif q2_start <= mins < q3_start: quarter_start = q2_start
            elif q3_start <= mins < q4_start: quarter_start = q3_start
            else: quarter_start = q4_start
            segment_position = (mins - quarter_start) // 90
            return int(segment_position % 4)
        elif timeframe == 15:
            h = local.hour; m = local.minute; mins = h * 60 + m
            def mm(hh, mm_): return hh * 60 + mm_
            q1_start = mm(1, 30); q2_start = mm(7, 30); q3_start = mm(13, 30); q4_start = mm(19, 30)
            if q1_start <= mins < q2_start: return 0
            if q2_start <= mins < q3_start: return 1
            if q3_start <= mins < q4_start: return 2
            return 3
        elif timeframe == 60:
            weekday = local.weekday()
            return {0:0,1:1,2:2,3:3}.get(weekday, 3)
        elif timeframe == 240:
            week_of_month = (local.day - 1) // 7 + 1
            week_of_month = min(week_of_month, 4)
            return week_of_month - 1
        else:
            return 0
    except Exception as e:
        logger.error(f"_quarter_index_from_boundary error: {e}")
        return 0


# --------------------------
# Quarter pair check logic (keeps OR semantics for reverse_moving)
# --------------------------

def _check_all_quarter_pairs(data_map: dict, group_id: str, group_type: str, primary_symbol: str, other_symbols: list, timeframe_min: int):
    triggered_alerts = []
    lengths = [len(df) for df in data_map.values()] if data_map else [0]
    if not lengths or min(lengths) < 2:
        logger.info("Insufficient historical data for quarter comparison (need at least 2 data points).")
        return triggered_alerts
    q1_index = -2; q2_index = -1
    q1_q2_data = {}
    valid_data = True
    for symbol in [primary_symbol] + other_symbols:
        df = data_map.get(symbol)
        if df is None or len(df) < 2:
             logger.warning(f"_check_all_quarter_pairs: Insufficient data for symbol {symbol}")
             valid_data = False; break
        try:
            q1_data = df.iloc[q1_index]; q2_data = df.iloc[q2_index]
            q1_q2_data[symbol] = {'Q1': q1_data, 'Q2': q2_data}
        except IndexError as e:
            logger.warning(f"_check_all_quarter_pairs: IndexError for {symbol}: {e}")
            valid_data = False; break
    if not valid_data or not q1_q2_data:
        return triggered_alerts
    q1_end_time_utc = q2_end_time_utc = None
    try:
        q1_bar_data = q1_q2_data[primary_symbol]['Q1']; q2_bar_data = q1_q2_data[primary_symbol]['Q2']
        q1_end_time_utc = pd.to_datetime(q1_bar_data['datetime'], utc=True)
        q2_end_time_utc = pd.to_datetime(q2_bar_data['datetime'], utc=True)
    except Exception as e:
        logger.error(f"_check_all_quarter_pairs: Error getting bar end times: {e}", exc_info=True)

    triggered_result = _check_single_quarter_pair(
        q1_q2_data, group_id, group_type, primary_symbol, other_symbols,
        timeframe_min, q1_end_time_utc, q2_end_time_utc
    )

    if triggered_result[0]:
        # triggered_result is (True, message, q2_end_time_utc) now
        triggered_alerts.append(triggered_result)
    return triggered_alerts


def _check_single_quarter_pair(q1_q2_data: dict, group_id: str, group_type: str,
                              primary_symbol: str, other_symbols: list, timeframe_min: int,
                              q1_end_time_utc: pd.Timestamp, q2_end_time_utc: pd.Timestamp):
    """
    Returns (is_triggered: bool, message: str|None, q2_end_time_utc: pd.Timestamp|None)
    """
    try:
        q1_label = "Q1"; q2_label = "Q2"
        q1_high_primary = float(q1_q2_data[primary_symbol]['Q1']['high'])
        q2_high_primary = float(q1_q2_data[primary_symbol]['Q2']['high'])
        if q2_high_primary > q1_high_primary:
            if group_type == "move_together":
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
                    return True, msg, q2_end_time_utc
            elif group_type == "reverse_moving":
                any_other_held_low = False
                for sym in other_symbols:
                    sym_q1_low = float(q1_q2_data[sym]['Q1']['low'])
                    sym_q2_low = float(q1_q2_data[sym]['Q2']['low'])
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
                    return True, msg, q2_end_time_utc
        q1_low_primary = float(q1_q2_data[primary_symbol]['Q1']['low'])
        q2_low_primary = float(q1_q2_data[primary_symbol]['Q2']['low'])
        if q2_low_primary < q1_low_primary:
            if group_type == "move_together":
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
                    return True, msg, q2_end_time_utc
            elif group_type == "reverse_moving":
                any_other_held_high = False
                for sym in other_symbols:
                    sym_q1_high = float(q1_q2_data[sym]['Q1']['high'])
                    sym_q2_high = float(q1_q2_data[sym]['Q2']['high'])
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
                    return True, msg, q2_end_time_utc
    except (KeyError, ValueError, TypeError) as e:
        logger.error(f"_check_single_quarter_pair: Error processing data for pair: {e}")
    return False, None, None


# --------------------------
# Background trigger handler (does long work off-loop)
# --------------------------

async def _handle_trigger_actions(alert_key: str, alert_details: dict, message: str, symbols: list, timeframe_min: int, user_id: int, bot: Optional[object] = None):
    """
    Generate charts, save JSON, and send notifications in background.

    Charts are sent as separate messages: first the textual alert, then one message per chart.

    All blocking bits run in thread workers via asyncio.to_thread.
    """
    try:
        chart_data_list = await _generate_charts_for_symbols(symbols, timeframe_min)

        triggered_data = {
            "timestamp": datetime.now(timezone.utc).isoformat() + 'Z',
            "alert_key": alert_key,
            "alert_details": alert_details,
            "message": message,
            # Do not attempt to persist binary buffers into JSON - instead store metadata
            "charts": [{"symbol": c.get('symbol'), "timeframe": c.get('timeframe'), "status": 'generated' if c.get('buffer') else 'error'} for c in chart_data_list]
        }

        # Save triggered alert (file I/O) in thread and with timeout
        try:
            await asyncio.wait_for(asyncio.to_thread(save_triggered_alert_with_charts, triggered_data), timeout=_SAVE_IO_TIMEOUT)
        except asyncio.TimeoutError:
            logger.warning("_handle_trigger_actions: saving triggered alert timed out for %s", alert_key)
        except Exception as e:
            logger.exception("_handle_trigger_actions: save triggered alert error: %s", e)

        # Send textual notification first
        try:
            # send only the text in the first message
            await send_alert_notification(user_id, message, [], bot=bot)
        except Exception as e:
            logger.exception("_handle_trigger_actions: notifying user (message) failed: %s", e)

        # Then send each chart in a separate message (if the send_alert_notification supports receiving a single chart buffer in the list)
        for chart_info in chart_data_list:
            try:
                buf = chart_info.get('buffer')
                sym = chart_info.get('symbol')
                tf = chart_info.get('timeframe')
                caption = f"Chart: {sym} {tf}m"
                # send the chart as a separate message; we pass a single-item list containing the buffer
                await send_alert_notification(user_id, caption, [buf] if buf is not None else [], bot=bot)
            except Exception as e:
                logger.exception("_handle_trigger_actions: sending chart for %s failed: %s", chart_info.get('symbol'), e)
    except Exception as e:
        logger.exception("_handle_trigger_actions: unexpected error: %s", e)


# --------------------------
# Main check function (async; accepts optional bot)
# --------------------------

async def check_alert_conditions(alert_key: str, alert_details: dict, bot: Optional[object] = None):
    """
    Returns (is_triggered: bool, message: str|None, chart_data_list: list|None)

    This function itself tries to be fast: heavy work (charting/sending/saving) is delegated
    to background tasks by calling _schedule_background_task on _handle_trigger_actions.

    To avoid repeated alerts for the same bar/condition we store a "last_trigger_signature"
    on the alert and only trigger again when the signature changes (e.g. new q2 bar time).
    """
    user_id = alert_details.get('user_id')
    group_id = alert_details.get('group_id')
    timeframe_min = alert_details.get('timeframe_min')
    category = alert_details.get('category')
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

    # Helper to check & update the last trigger signature to avoid duplicates
    def _should_trigger_and_mark(signature: str) -> bool:
        alerts = load_alerts()
        alert = alerts.get(alert_key, {})
        last_sig = alert.get('last_trigger_signature')
        if last_sig == signature:
            logger.info(f"Skipping trigger for {alert_key}: same signature {signature}")
            return False
        # mark now (persist immediately) so concurrent checks won't double-send
        alert['last_trigger_signature'] = signature
        alert['updated_at'] = datetime.now(timezone.utc).isoformat() + 'Z'
        alerts[alert_key] = alert
        save_alerts(alerts)
        return True

    # 5m special windows (reuse existing behavior)
    if timeframe_min == 5:
        now_target = datetime.now(TIMEZONE_TARGET)
        today = now_target.date()
        def _localized(h, m): return TIMEZONE_TARGET.localize(datetime.combine(today, datetime.min.time().replace(hour=h, minute=m)))
        start_time_1 = _localized(1, 30); end_time_1 = _localized(7, 30)
        start_time_2 = _localized(7, 30); end_time_2 = _localized(13, 30)
        triggered_results = []
        if start_time_1 <= now_target < end_time_1:
            triggered_results = await _process_5m_window(alert_details, symbols, group_id, group_type, user_id, timeframe_min, 1, bot)
        elif start_time_2 <= now_target < end_time_2:
            triggered_results = await _process_5m_window(alert_details, symbols, group_id, group_type, user_id, timeframe_min, 2, bot)
        else:
            logger.info("Outside 5m alert windows, no processing")
            return False, None, None

        # triggered_results items are (True, message, signature)
        for (is_triggered, message, signature) in triggered_results:
            if is_triggered and message:
                sig = signature or f"msghash:{hash(message)}"
                if _should_trigger_and_mark(sig):
                    bg_name = f"trigger_handler::{alert_key}"
                    _schedule_background_task(_handle_trigger_actions(alert_key, alert_details, message, symbols, timeframe_min, user_id, bot=bot), bg_name)
                    return True, message, None
                else:
                    return False, None, None
        return False, None, None

    # For other timeframes: fetch data asynchronously
    data_map_raw = await _get_data_for_alert_checking(symbols, timeframe_min)
    if not data_map_raw:
        logger.error("No data fetched for alert check.")
        return False, None, None

    if len(data_map_raw) != len(symbols):
         missing_symbols = set(symbols) - set(data_map_raw.keys())
         logger.error(f"Failed to fetch data for symbols: {missing_symbols}")
         return False, None, None

    # Resample each DataFrame off the event loop concurrently
    resample_tasks = []
    for symbol, df in data_map_raw.items():
        resample_tasks.append(asyncio.create_task(asyncio.to_thread(_resample_to_alert_timeframe, df, timeframe_min)))
    # gather with timeout to avoid blocking
    try:
        resampled_results = await asyncio.wait_for(asyncio.gather(*resample_tasks, return_exceptions=True), timeout=_PER_ALERT_FETCH_TIMEOUT)
    except asyncio.TimeoutError:
        logger.error("Resampling timeout during alert check.")
        for t in resample_tasks:
            if not t.done():
                t.cancel()
        return False, None, None

    data_map = {}
    for sym, res in zip(data_map_raw.keys(), resampled_results):
        if isinstance(res, Exception):
            logger.exception("_resample_to_alert_timeframe produced exception for %s: %s", sym, res)
        elif res is None or (hasattr(res, 'empty') and res.empty):
            logger.warning("Resampling for %s produced empty result", sym)
        else:
            data_map[sym] = res

    if len(data_map) != len(symbols):
         logger.error(f"Insufficient data after resampling. Expected {len(symbols)}, got {len(data_map)}.")
         return False, None, None

    primary_symbol = symbols[0]
    other_symbols = symbols[1:]

    triggered_results = _check_all_quarter_pairs(
        data_map, group_id, group_type, primary_symbol, other_symbols, timeframe_min
    )

    for (is_triggered, message, q2_end_time_utc) in triggered_results:
        if is_triggered and message:
            # create a deterministic signature for the triggering bar if available
            if q2_end_time_utc is not None:
                try:
                    sig = f"bar:{int(pd.to_datetime(q2_end_time_utc, utc=True).timestamp())}"
                except Exception:
                    sig = f"msghash:{hash(message)}"
            else:
                sig = f"msghash:{hash(message)}"

            if _should_trigger_and_mark(sig):
                bg_name = f"trigger_handler::{alert_key}"
                _schedule_background_task(_handle_trigger_actions(alert_key, alert_details, message, symbols, timeframe_min, user_id, bot=bot), bg_name)
                return True, message, None
            else:
                return False, None, None

    logger.debug("Alert conditions not met for the most recent quarter pair.")
    return False, None, None


# 5m window processing reuses async fetch/resample & scheduling
async def _process_5m_window(alert_details: dict, symbols: list, group_id: str, group_type: str,
                            user_id: int, timeframe_min: int, window_number: int, bot: Optional[object] = None):
    data_map_raw = await _get_data_for_alert_checking(symbols, timeframe_min)
    if not data_map_raw:
        logger.error("No data fetched for 5m alert check in window.")
        return []
    if len(data_map_raw) != len(symbols):
         missing_symbols = set(symbols) - set(data_map_raw.keys())
         logger.error(f"_process_5m_window: Failed to fetch data for symbols in window {window_number}: {missing_symbols}")
         return []

    # resample concurrently
    resample_tasks = [asyncio.create_task(asyncio.to_thread(_resample_to_alert_timeframe, df, timeframe_min)) for df in data_map_raw.values()]
    try:
        resampled_results = await asyncio.wait_for(asyncio.gather(*resample_tasks, return_exceptions=True), timeout=_PER_ALERT_FETCH_TIMEOUT)
    except asyncio.TimeoutError:
        logger.error("_process_5m_window: resampling timeout")
        for t in resample_tasks:
            if not t.done():
                t.cancel()
        return []

    data_map = {}
    for sym, res in zip(data_map_raw.keys(), resampled_results):
        if isinstance(res, Exception):
            logger.exception("_process_5m_window: resample exception for %s: %s", sym, res)
        elif res is None or (hasattr(res, 'empty') and res.empty):
            logger.warning("_process_5m_window: resample empty for %s", sym)
        else:
            data_map[sym] = res

    if not data_map or len(data_map) != len(symbols):
        logger.error(f"Insufficient data for 5m window {window_number}.")
        return []

    primary_symbol = symbols[0]
    other_symbols = symbols[1:]

    triggered_results = _check_all_quarter_pairs(
        data_map, group_id, group_type, primary_symbol, other_symbols, timeframe_min
    )

    final_results = []
    for i, (is_triggered, message, q2_end_time_utc) in enumerate(triggered_results):
        if is_triggered and message:
            # return signature as third element so caller can dedupe
            if q2_end_time_utc is not None:
                try:
                    sig = f"bar:{int(pd.to_datetime(q2_end_time_utc, utc=True).timestamp())}"
                except Exception:
                    sig = f"msghash:{hash(message)}"
            else:
                sig = f"msghash:{hash(message)}"
            final_results.append((True, message, sig))
    return final_results
