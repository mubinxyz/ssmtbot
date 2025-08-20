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
import io
import math
import re

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

# Telegram message length guard
_TELEGRAM_MAX_MESSAGE_LEN = 3900

logger = logging.getLogger(__name__)


# --------------------------
# Utility helpers
# --------------------------
def _atomic_write_json(path: str, data):
    tmp = f"{path}.tmp"
    try:
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
    except Exception as e:
        logger.exception("_atomic_write_json failed for %s: %s", path, e)
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass

def _backup_corrupt_file(path: str) -> None:
    try:
        if os.path.exists(path):
            ts = datetime.now().strftime('%Y%m%dT%H%M%S')
            bak = f"{path}.corrupt.{ts}"
            os.replace(path, bak)
            logger.warning("Backed up corrupt file %s -> %s", path, bak)
    except Exception as e:
        logger.exception("_backup_corrupt_file failed for %s: %s", path, e)

def _escape_for_telegram_markdown_v1(text: str) -> str:
    """
    Escape text for Telegram Markdown (v1). The notify_user appears to use ParseMode.MARKDOWN,
    so escape the characters that can create entities there: _ * ` [ ]
    Also escape backslash first.
    """
    if not isinstance(text, str):
        text = str(text)
    text = text.replace('\\', '\\\\')
    for ch in ['_', '*', '`', '[', ']']:
        text = text.replace(ch, '\\' + ch)
    return text

def _truncate_message(text: str, max_len: int = _TELEGRAM_MAX_MESSAGE_LEN) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + '...'


# --------------------------
# Alert storage helpers (robust against corrupted JSON)
# --------------------------
def load_alerts():
    if not os.path.exists(ALERTS_STORE_FILE):
        return {}
    try:
        with open(ALERTS_STORE_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"Error loading alerts from {ALERTS_STORE_FILE}: {e}")
        _backup_corrupt_file(ALERTS_STORE_FILE)
        return {}

def save_alerts(alerts):
    try:
        _atomic_write_json(ALERTS_STORE_FILE, alerts)
    except IOError as e:
        logger.error(f"Error saving alerts to {ALERTS_STORE_FILE}: {e}")

def load_triggered_alerts_with_charts():
    """
    Load triggered alerts list robustly. If the JSON is malformed we try two fallbacks:
      1) try to parse as newline-delimited JSON (NDJSON)
      2) backup corrupt file and return empty list
    """
    if not os.path.exists(TRIGGERED_ALERTS_WITH_CHARTS_FILE):
        return []
    try:
        with open(TRIGGERED_ALERTS_WITH_CHARTS_FILE, 'r', encoding='utf-8') as f:
            raw = f.read()
            data = json.loads(raw)
            return data if isinstance(data, list) else []
    except json.JSONDecodeError as e:
        logger.error("Error loading triggered alerts from %s: %s", TRIGGERED_ALERTS_WITH_CHARTS_FILE, e)
        # try NDJSON parse (one JSON object per line)
        try:
            parsed = []
            with open(TRIGGERED_ALERTS_WITH_CHARTS_FILE, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parsed.append(json.loads(line))
            logger.warning("Parsed %d entries from NDJSON fallback for %s", len(parsed), TRIGGERED_ALERTS_WITH_CHARTS_FILE)
            return parsed
        except Exception:
            _backup_corrupt_file(TRIGGERED_ALERTS_WITH_CHARTS_FILE)
            return []
    except IOError as e:
        logger.error(f"Error loading triggered alerts from {TRIGGERED_ALERTS_WITH_CHARTS_FILE}: {e}")
        return []

def save_triggered_alert_with_charts(triggered_alert_data):
    all_triggered = load_triggered_alerts_with_charts()
    all_triggered.append(triggered_alert_data)
    try:
        _atomic_write_json(TRIGGERED_ALERTS_WITH_CHARTS_FILE, all_triggered)
    except Exception as e:
        logger.exception("Error saving triggered alert to %s: %s", TRIGGERED_ALERTS_WITH_CHARTS_FILE, e)


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
    try:
        coro = asyncio.to_thread(get_ohlc, symbol, data_resolution, from_ts, to_ts)
        df = await asyncio.wait_for(coro, timeout=timeout)
        if df is None or (hasattr(df, 'empty') and df.empty):
            logger.warning("_fetch_symbol_df: get_ohlc returned empty for %s", symbol)
            return None
        df_conv = await asyncio.to_thread(_convert_lf_time_to_target, df)
        return df_conv
    except asyncio.TimeoutError:
        logger.warning("_fetch_symbol_df: timeout fetching %s", symbol)
    except Exception as e:
        logger.exception("_fetch_symbol_df: error fetching %s: %s", symbol, e)
    return None

async def _get_data_for_alert_checking(symbols: list[str], timeframe_min: int) -> dict[str, pd.DataFrame]:
    data_map = {}
    now_target = datetime.now(TIMEZONE_TARGET)

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
    try:
        if inspect.iscoroutinefunction(generate_chart):
            coro = generate_chart(symbols, timeframe=timeframe_min)
            chart_buffers = await asyncio.wait_for(coro, timeout=_CHART_GENERATION_TIMEOUT)
        else:
            chart_buffers = await asyncio.wait_for(asyncio.to_thread(generate_chart, symbols, timeframe_min), timeout=_CHART_GENERATION_TIMEOUT)

        chart_data_list = []
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

def _format_ts_for_target(ts_utc: pd.Timestamp | None) -> str:
    if ts_utc is None:
        return 'n/a'
    try:
        t = pd.to_datetime(ts_utc, utc=True).tz_convert(TIMEZONE_TARGET)
        return t.strftime('%Y-%m-%d %H:%M:%S %Z')
    except Exception:
        return str(ts_utc)


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
        triggered_alerts.append(triggered_result)
    return triggered_alerts

def _check_single_quarter_pair(q1_q2_data: dict, group_id: str, group_type: str,
                              primary_symbol: str, other_symbols: list, timeframe_min: int,
                              q1_end_time_utc: pd.Timestamp, q2_end_time_utc: pd.Timestamp):
    """
    Use the later bar (q2_end_time_utc) to determine the quarter index for Q2, and label Q1 as the previous quarter.
    This avoids cases where both timestamps fall within the same quarter due to boundaries, but the bars are
    still consecutive (e.g. Q3 -> Q4).
    """
    try:
        # Determine quarter indices based on the q2 (latest) timestamp, then set q1 as previous
        if q2_end_time_utc is not None:
            q2_idx = _quarter_index_from_boundary(q2_end_time_utc, timeframe=timeframe_min)
            q1_idx = (q2_idx - 1) % 4
        else:
            # fallback: compute from q1 and then set q2 = next
            q1_idx = _quarter_index_from_boundary(q1_end_time_utc, timeframe=timeframe_min)
            q2_idx = (q1_idx + 1) % 4

        q1_label = f"Q{q1_idx + 1}"
        q2_label = f"Q{q2_idx + 1}"

        q1_high_primary = float(q1_q2_data[primary_symbol]['Q1']['high'])
        q2_high_primary = float(q1_q2_data[primary_symbol]['Q2']['high'])

        per_symbol_lines = []
        for sym, d in q1_q2_data.items():
            try:
                q1h = float(d['Q1']['high']); q1l = float(d['Q1']['low'])
                q2h = float(d['Q2']['high']); q2l = float(d['Q2']['low'])
                per_symbol_lines.append(f"{sym}: {q1_label}(H/L)={q1h:.5f}/{q1l:.5f}, {q2_label}(H/L)={q2h:.5f}/{q2l:.5f}")
            except Exception:
                per_symbol_lines.append(f"{sym}: data unavailable or invalid")

        q1_time_str = _format_ts_for_target(q1_end_time_utc)
        q2_time_str = _format_ts_for_target(q2_end_time_utc)

        header = (
            f"⚠️ Alert Triggered — Group: {group_id} ({group_type}) — {timeframe_min}m\n"
            f"Primary: {primary_symbol} | {q1_label} end: {q1_time_str} | {q2_label} end: {q2_time_str}\n"
        )

        # check highs: primary's q2 > q1 high
        if q2_high_primary > q1_high_primary:
            if group_type == "move_together":
                others_follow = []
                others_not = []
                for sym in other_symbols:
                    sym_q1_high = float(q1_q2_data[sym]['Q1']['high'])
                    sym_q2_high = float(q1_q2_data[sym]['Q2']['high'])
                    if sym_q2_high <= sym_q1_high:
                        others_not.append(sym)
                    else:
                        others_follow.append(sym)
                if others_not:
                    msg = (
                        header +
                        f"Condition: Primary broke {q1_label} high ({q1_high_primary:.5f} → {q2_high_primary:.5f}) in {q2_label}.\n"
                        f"Others that followed: {', '.join(others_follow) if others_follow else 'none'}.\n"
                        f"Others that DID NOT follow (held or didn't exceed {q1_label} high): {', '.join(others_not) if others_not else 'none'}.\n\n"
                        f"Per-symbol summary:\n" + "\n".join(per_symbol_lines)
                    )
                    logger.info(f"Triggered: {msg}")
                    return True, msg, q2_end_time_utc
            elif group_type == "reverse_moving":
                held_low_syms = []
                for sym in other_symbols:
                    sym_q1_low = float(q1_q2_data[sym]['Q1']['low'])
                    sym_q2_low = float(q1_q2_data[sym]['Q2']['low'])
                    if sym_q2_low >= sym_q1_low:
                        held_low_syms.append(sym)
                if held_low_syms:
                    msg = (
                        header +
                        f"Condition: Primary broke {q1_label} high ({q1_high_primary:.5f} → {q2_high_primary:.5f}) in {q2_label},\n"
                        f"while at least one other held their {q1_label} low: {', '.join(held_low_syms)}.\n\n"
                        f"Per-symbol summary:\n" + "\n".join(per_symbol_lines)
                    )
                    logger.info(f"Triggered: {msg}")
                    return True, msg, q2_end_time_utc

        # check lows: primary's q2 < q1 low
        q1_low_primary = float(q1_q2_data[primary_symbol]['Q1']['low'])
        q2_low_primary = float(q1_q2_data[primary_symbol]['Q2']['low'])
        if q2_low_primary < q1_low_primary:
            if group_type == "move_together":
                others_follow = []
                others_not = []
                for sym in other_symbols:
                    sym_q1_low = float(q1_q2_data[sym]['Q1']['low'])
                    sym_q2_low = float(q1_q2_data[sym]['Q2']['low'])
                    if sym_q2_low >= sym_q1_low:
                        others_not.append(sym)
                    else:
                        others_follow.append(sym)
                if others_not:
                    msg = (
                        header +
                        f"Condition: Primary broke {q1_label} low ({q1_low_primary:.5f} → {q2_low_primary:.5f}) in {q2_label}.\n"
                        f"Others that followed lower: {', '.join(others_follow) if others_follow else 'none'}.\n"
                        f"Others that DID NOT follow (held or didn't drop below {q1_label} low): {', '.join(others_not) if others_not else 'none'}.\n\n"
                        f"Per-symbol summary:\n" + "\n".join(per_symbol_lines)
                    )
                    logger.info(f"Triggered: {msg}")
                    return True, msg, q2_end_time_utc
            elif group_type == "reverse_moving":
                held_high_syms = []
                for sym in other_symbols:
                    sym_q1_high = float(q1_q2_data[sym]['Q1']['high'])
                    sym_q2_high = float(q1_q2_data[sym]['Q2']['high'])
                    if sym_q2_high >= sym_q1_high:
                        held_high_syms.append(sym)
                if held_high_syms:
                    msg = (
                        header +
                        f"Condition: Primary broke {q1_label} low ({q1_low_primary:.5f} → {q2_low_primary:.5f}) in {q2_label},\n"
                        f"while at least one other held their {q1_label} high: {', '.join(held_high_syms)}.\n\n"
                        f"Per-symbol summary:\n" + "\n".join(per_symbol_lines)
                    )
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

    NOTE: We send charts separately (one message per chart) and send the textual alert first.
    This avoids Telegram entity parsing errors by escaping for Markdown v1.
    """
    try:
        chart_data_list = await _generate_charts_for_symbols(symbols, timeframe_min)

        triggered_data = {
            "timestamp": datetime.now(timezone.utc).isoformat() + 'Z',
            "alert_key": alert_key,
            "alert_details": alert_details,
            "message": message,
            "charts": [{"symbol": c.get('symbol'), "timeframe": c.get('timeframe'), "status": 'generated' if c.get('buffer') else c.get('error', 'error')} for c in chart_data_list]
        }

        # Save triggered alert (file I/O) in thread and with timeout
        try:
            await asyncio.wait_for(asyncio.to_thread(save_triggered_alert_with_charts, triggered_data), timeout=_SAVE_IO_TIMEOUT)
        except asyncio.TimeoutError:
            logger.warning("_handle_trigger_actions: saving triggered alert timed out for %s", alert_key)
        except Exception as e:
            logger.exception("_handle_trigger_actions: save triggered alert error: %s", e)

        # Prepare image buffers
        image_buffers = []
        for c in chart_data_list:
            buf = c.get('buffer')
            if buf is None:
                continue
            if hasattr(buf, 'read'):
                try:
                    b = buf.read()
                    image_buffers.append(b)
                except Exception:
                    try:
                        buf.seek(0)
                        image_buffers.append(buf.read())
                    except Exception:
                        logger.warning("_handle_trigger_actions: unable to read buffer for chart %s", c.get('symbol'))
            elif isinstance(buf, (bytes, bytearray)):
                image_buffers.append(bytes(buf))
            else:
                logger.warning("_handle_trigger_actions: chart buffer is unknown type for %s", c.get('symbol'))

        # Sanitize and truncate message to avoid Telegram parse errors (ParseMode.MARKDOWN in notify_user)
        try:
            safe_message = _escape_for_telegram_markdown_v1(message)
            safe_message = _truncate_message(safe_message)
        except Exception as e:
            logger.exception("_handle_trigger_actions: message sanitization failed: %s", e)
            safe_message = _truncate_message(str(message))

        # Send textual notification first (sanitized for Markdown v1)
        try:
            await send_alert_notification(user_id, safe_message, [], bot=bot)
        except Exception as e:
            logger.exception("_handle_trigger_actions: notifying user (text) failed: %s", e)
            # try plain fallback (no escaping)
            try:
                await send_alert_notification(user_id, _truncate_message(str(message)), [], bot=bot)
            except Exception as e2:
                logger.exception("_handle_trigger_actions: notifying user (plain text fallback) failed: %s", e2)

        # Then send each chart in a separate message (each as its own file message)
        for chart_info in chart_data_list:
            try:
                buf = chart_info.get('buffer')
                if buf is None:
                    continue
                if hasattr(buf, 'read'):
                    try:
                        b = buf.read()
                    except Exception:
                        try:
                            buf.seek(0)
                            b = buf.read()
                        except Exception:
                            logger.warning("_handle_trigger_actions: unable to read buffer for chart %s", chart_info.get('symbol'))
                            continue
                elif isinstance(buf, (bytes, bytearray)):
                    b = bytes(buf)
                else:
                    logger.warning("_handle_trigger_actions: chart buffer is unknown type for %s", chart_info.get('symbol'))
                    continue

                # send each chart as its own notification (no text)
                try:
                    await send_alert_notification(user_id, '', [b], bot=bot)
                except Exception as e:
                    logger.exception("_handle_trigger_actions: sending chart for %s failed: %s", chart_info.get('symbol'), e)
            except Exception as e:
                logger.exception("_handle_trigger_actions: unexpected error while sending chart: %s", e)

    except Exception as e:
        logger.exception("_handle_trigger_actions: unexpected error: %s", e)


# --------------------------
# Main check function (async; accepts optional bot)
# --------------------------
async def check_alert_conditions(alert_key: str, alert_details: dict, bot: Optional[object] = None):
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

    def _should_trigger_and_mark(signature: str) -> bool:
        alerts = load_alerts()
        alert = alerts.get(alert_key, {})
        last_sig = alert.get('last_trigger_signature')
        if last_sig == signature:
            logger.info(f"Skipping trigger for {alert_key}: same signature {signature}")
            return False
        alert['last_trigger_signature'] = signature
        alert['updated_at'] = datetime.now(timezone.utc).isoformat() + 'Z'
        alerts[alert_key] = alert
        save_alerts(alerts)
        return True

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

    data_map_raw = await _get_data_for_alert_checking(symbols, timeframe_min)
    if not data_map_raw:
        logger.error("No data fetched for alert check.")
        return False, None, None

    if len(data_map_raw) != len(symbols):
         missing_symbols = set(symbols) - set(data_map_raw.keys())
         logger.error(f"Failed to fetch data for symbols: {missing_symbols}")
         return False, None, None

    resample_tasks = []
    for symbol, df in data_map_raw.items():
        resample_tasks.append(asyncio.create_task(asyncio.to_thread(_resample_to_alert_timeframe, df, timeframe_min)))
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
            if q2_end_time_utc is not None:
                try:
                    sig = f"bar:{int(pd.to_datetime(q2_end_time_utc, utc=True).timestamp())}"
                except Exception:
                    sig = f"msghash:{hash(message)}"
            else:
                sig = f"msghash:{hash(message)}"
            final_results.append((True, message, sig))
    return final_results
