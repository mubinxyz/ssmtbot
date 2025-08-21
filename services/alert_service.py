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
from typing import Optional, Union
import io
import math
import random
import re
from ssl import SSLError as SSL_ERROR

from services.notify_user import send_alert_notification

# --- Configuration ---
ALERTS_STORE_FILE = "alerts_store.json"
TRIGGERED_ALERTS_WITH_CHARTS_FILE = "triggered_alerts_with_charts.json"
CHARTS_OUTPUT_DIR = "charts"  # optional persist

# NOTE: LiteFinance provider uses UTC+3 according to your comment.
# Using pytz.FixedOffset avoids potential sign confusion with 'Etc/GMT-3'.
# FixedOffset takes minutes east of UTC, so 180 -> UTC+3.
TIMEZONE_LITEFINANCE = pytz.FixedOffset(180)  # UTC+3 (unambiguous)
TIMEZONE_TARGET = pytz.timezone('Asia/Tehran')     # UTC+3:30
TIMEZONE_UTC = pytz.utc

# Concurrency/timeouts for fetching & charting
_SYMBOL_FETCH_CONCURRENCY = 3
_PER_SYMBOL_FETCH_TIMEOUT = 30.0
_PER_ALERT_FETCH_TIMEOUT = 80.0
_CHART_GENERATION_TIMEOUT = 60.0
_SAVE_IO_TIMEOUT = 15.0

_TELEGRAM_MAX_MESSAGE_LEN = 3900

# Small epsilon to avoid micro-breaks due to floating precision / rounding noise
EPS = 1e-6

logger = logging.getLogger(__name__)

# --------------------------
# Existing group characteristics
# --------------------------
GROUP_CHARACTERISTICS = {
    "dxy_eu_gu": {"label": "DXY / EURUSD / GBPUSD", "symbols": ["USDX", "EURUSD", "GBPUSD"], "type": "reverse_moving"},
    "btc_eth_xrp": {"label": "BTCUSD / ETHUSD / XRPUSD", "symbols": ["BTCUSD", "ETHUSD", "XRPUSD"], "type": "move_together"},
    "spx_nq_ym": {"label": "S&P500 (SPX) / NASDAQ (NQ) / DOW (YM)", "symbols": ["SPX", "NQ", "YM"], "type": "move_together"},
    "dxy_xau_xag_aud": {"label": "DXY / XAU / XAG / AUD", "symbols": ["USDX", "XAU", "XAG", "AUD"], "type": "reverse_moving"}
}

def find_group(category: str, group_id: str):
    if category == "FOREX CURRENCIES" and group_id in GROUP_CHARACTERISTICS:
        return GROUP_CHARACTERISTICS[group_id]
    return None

# schedule background coroutine safely (put this near other helpers / after imports)
def _schedule_background_task(coro, name: Optional[str] = None) -> Union[asyncio.Task, threading.Thread]:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None and loop.is_running():
        try:
            task = asyncio.create_task(coro)
            try:
                if name:
                    task.set_name(name)
            except Exception:
                pass
            return task
        except Exception:
            logger.exception("_schedule_background_task: failed to create asyncio task, falling back to thread runner")

    def _runner():
        try:
            asyncio.run(coro)
        except Exception:
            logger.exception("_schedule_background_task: background thread runner exception for %s", name)

    t = threading.Thread(target=_runner, daemon=True, name=(name or "bg_task_thread"))
    t.start()
    return t

# --------------------------
# Simple atomic file helper (unchanged)
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

# --------------------------
# Alerts storage (unchanged)
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
    if not os.path.exists(TRIGGERED_ALERTS_WITH_CHARTS_FILE):
        return []
    try:
        with open(TRIGGERED_ALERTS_WITH_CHARTS_FILE, 'r', encoding='utf-8') as f:
            raw = f.read()
            data = json.loads(raw)
            return data if isinstance(data, list) else []
    except json.JSONDecodeError as e:
        logger.error("Error loading triggered alerts from %s: %s", TRIGGERED_ALERTS_WITH_CHARTS_FILE, e)
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
# Utility helpers (escape/truncation)
# --------------------------
def _escape_for_telegram_markdown_v1(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = text.replace('\\', '\\\\')
    for ch in ['_', '*', '`', '[', ']']:
        text = text.replace(ch, '\\' + ch)
    return text

def _truncate_message(text: str, max_len: int = _TELEGRAM_MAX_MESSAGE_LEN) -> str:
    if text is None:
        return ''
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + '...'

# --------------------------
# Time conversion & resampling (kept as-is)
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
# Robust fetch wrapper with retries + cooldown
# --------------------------
_SYMBOL_FAILURES: dict[str, int] = {}
_SYMBOL_COOLDOWN_UNTIL: dict[str, float] = {}
_FAILURES_BEFORE_COOLDOWN = 3
_COOLDOWN_SECONDS = 60 * 3

async def _fetch_symbol_df(symbol: str, data_resolution: int, from_ts: int, to_ts: int, timeout: float) -> Optional[pd.DataFrame]:
    now_ts = asyncio.get_event_loop().time()
    cooldown_until = _SYMBOL_COOLDOWN_UNTIL.get(symbol)
    if cooldown_until and now_ts < cooldown_until:
        logger.warning("_fetch_symbol_df: skipping %s due to cooldown until %.1f (now %.1f)", symbol, cooldown_until, now_ts)
        return None

    max_attempts = 4
    base_backoff = 0.8
    attempt = 0
    start_time = asyncio.get_event_loop().time()
    last_exc = None

    while attempt < max_attempts:
        attempt += 1
        elapsed = asyncio.get_event_loop().time() - start_time
        remaining_total = timeout - elapsed
        if remaining_total <= 0:
            logger.warning("_fetch_symbol_df: overall timeout exhausted for %s after %d attempts", symbol, attempt - 1)
            break

        per_attempt_timeout = min(remaining_total, timeout)
        try:
            coro = asyncio.to_thread(get_ohlc, symbol, data_resolution, from_ts, to_ts)
            df = await asyncio.wait_for(coro, timeout=per_attempt_timeout)

            if df is None:
                last_exc = RuntimeError("get_ohlc returned None")
                logger.warning("_fetch_symbol_df: get_ohlc returned None for %s (attempt %d)", symbol, attempt)
                raise last_exc
            if hasattr(df, "empty") and df.empty:
                last_exc = RuntimeError("empty dataframe")
                logger.warning("_fetch_symbol_df: get_ohlc returned empty for %s (attempt %d)", symbol, attempt)
                raise last_exc

            df_conv = await asyncio.to_thread(_convert_lf_time_to_target, df)
            _SYMBOL_FAILURES.pop(symbol, None)
            _SYMBOL_COOLDOWN_UNTIL.pop(symbol, None)
            return df_conv

        except asyncio.TimeoutError as e:
            last_exc = e
            logger.warning("_fetch_symbol_df: attempt %d timeout for %s (per-attempt %ss)", attempt, symbol, per_attempt_timeout)
        except SSL_ERROR as e:
            last_exc = e
            logger.warning("_fetch_symbol_df: SSL error on attempt %d for %s: %s", attempt, symbol, e)
        except Exception as e:
            last_exc = e
            logger.warning("_fetch_symbol_df: attempt %d error fetching %s: %s", attempt, symbol, e)

        prev_fail = _SYMBOL_FAILURES.get(symbol, 0)
        _SYMBOL_FAILURES[symbol] = prev_fail + 1

        if _SYMBOL_FAILURES[symbol] >= _FAILURES_BEFORE_COOLDOWN:
            until = asyncio.get_event_loop().time() + _COOLDOWN_SECONDS
            _SYMBOL_COOLDOWN_UNTIL[symbol] = until
            logger.warning("_fetch_symbol_df: symbol %s is on cooldown for %ds after %d failures", symbol, _COOLDOWN_SECONDS, _SYMBOL_FAILURES[symbol])
            break

        backoff = base_backoff * (2 ** (attempt - 1))
        jitter = random.uniform(0, backoff * 0.3)
        sleep_for = min(backoff + jitter, max(0.0, remaining_total - 0.01))
        if sleep_for > 0:
            try:
                await asyncio.sleep(sleep_for)
            except Exception:
                pass

    logger.warning("_fetch_symbol_df: all attempts failed for %s (last error: %s). consecutive failures=%d", symbol, repr(last_exc), _SYMBOL_FAILURES.get(symbol, 0))
    return None

# --------------------------
# Chart generation (unchanged)
# --------------------------
async def _generate_charts_for_symbols(symbols: list[str], timeframe_min: int) -> list:
    try:
        if inspect.iscoroutinefunction(generate_chart):
            coro = generate_chart(symbols, timeframe=timeframe_min)
            chart_buffers = await asyncio.wait_for(coro, timeout=_CHART_GENERATION_TIMEOUT)
        else:
            chart_buffers = await asyncio.wait_for(asyncio.to_thread(generate_chart, symbols, timeframe_min), timeout=_CHART_GENERATION_TIMEOUT)

        chart_data_list = []
        combined_label = "__combined__"
        n_symbols = len(symbols) if symbols else 0
        n_buffers = len(chart_buffers) if chart_buffers else 0

        for i, buf in enumerate(chart_buffers):
            if i < n_symbols:
                sym_label = symbols[i]
            else:
                if i == n_symbols and n_buffers == n_symbols + 1:
                    sym_label = combined_label
                else:
                    sym_label = "unknown"
            chart_data_list.append({
                "symbol": sym_label,
                "timeframe": timeframe_min,
                "buffer": buf
            })
        return chart_data_list
    except asyncio.TimeoutError:
        logger.warning("_generate_charts_for_symbols: timeout generating charts for %s", symbols)
        return [{"error": "chart generation timeout", "symbol": symbols[i] if (symbols and i < len(symbols)) else "unknown"} for i in range(max(1, len(symbols)))]
    except Exception as e:
        logger.exception("_generate_charts_for_symbols: error: %s", e)
        return [{"error": f"chart generation failed: {e}", "symbol": symbols[i] if (symbols and i < len(symbols)) else "unknown"} for i in range(max(1, len(symbols)))]

# --------------------------
# Quarter logic helpers (unchanged)
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
# NEW: Revised pretty alert message builder (English-only, per-case did-not-break lists)
# --------------------------

def _build_pretty_alert_message(group_id: str, group_type: str, timeframe_min: int,
                               primary_symbol: str, q1_label: str, q2_label: str,
                               q1_time_str: str, q2_time_str: str,
                               action_verb: str,     # "HIGH" or "LOW"
                               q1_val: float, q2_val: float,
                               primary_direction_emoji: str,
                               didnt_break_high_list: list, didnt_break_low_list: list) -> str:
    """
    Build a compact English-only alert message.

    - didnt_break_high_list: symbols whose Q2.high <= Q1.high (i.e. DID NOT break HIGH)
    - didnt_break_low_list: symbols whose Q2.low >= Q1.low (i.e. DID NOT break LOW)

    Returns a Markdown-v1 string (not escaped).
    """
    label = GROUP_CHARACTERISTICS.get(group_id, {}).get("label", group_id)
    title = f"🔔 *SSMT Alert — {label}*"
    meta = f"Group type: `{group_type}` | Timeframe: `{timeframe_min}m`"
    time_span = f"{q1_label} end: {q1_time_str}  →  {q2_label} end: {q2_time_str}"

    # Primary action line (mentions the actual breaker symbol)
    action_line = f"{primary_direction_emoji} *{primary_symbol} broke previous quarter {action_verb}* — `{q1_val:.5f}` → `{q2_val:.5f}`"

    dh = ', '.join(didnt_break_high_list) if didnt_break_high_list else "None"
    dl = ', '.join(didnt_break_low_list) if didnt_break_low_list else "None"

    didnt_high_line = f"• Didn't break previous-quarter HIGH: {dh}"
    didnt_low_line = f"• Didn't break previous-quarter LOW: {dl}"

    parts = [
        title,
        meta,
        time_span,
        "――――――――――――――――――――――――――",
        action_line,
        didnt_high_line,
        didnt_low_line
    ]
    return "\n".join(parts)


# --------------------------
# Quarter pair check logic (reworked to return relevant didn't-break lists)
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

    # candidate iteration includes primary + others (allows any symbol to be the one that broke)
    candidate_list = [primary_symbol] + [s for s in other_symbols if s != primary_symbol]

    for candidate in candidate_list:
        others_for_candidate = [s for s in candidate_list if s != candidate]
        try:
            q1_bar_data = q1_q2_data[candidate]['Q1']; q2_bar_data = q1_q2_data[candidate]['Q2']
            q1_end_time_utc = pd.to_datetime(q1_bar_data['datetime'], utc=True)
            q2_end_time_utc = pd.to_datetime(q2_bar_data['datetime'], utc=True)
        except Exception as e:
            logger.error(f"_check_all_quarter_pairs: Error getting bar end times for candidate {candidate}: {e}", exc_info=True)
            continue

        triggered_result = _check_single_quarter_pair(
            q1_q2_data, group_id, group_type, candidate, others_for_candidate, timeframe_min,
            q1_end_time_utc, q2_end_time_utc
        )

        if triggered_result and triggered_result[0]:
            triggered_alerts.append(triggered_result)

    return triggered_alerts

def _safe_float(val, fallback=None):
    try:
        return float(val)
    except Exception:
        return fallback

def _check_single_quarter_pair(q1_q2_data: dict, group_id: str, group_type: str,
                              primary_symbol: str, other_symbols: list, timeframe_min: int,
                              q1_end_time_utc: pd.Timestamp, q2_end_time_utc: pd.Timestamp):
    """
    Decide whether the pair of consecutive quarter bars (Q1 -> Q2) triggers an alert.
    Returns (triggered: bool, message: str, q2_end_time_utc)
    """
    try:
        if q2_end_time_utc is not None:
            q2_idx = _quarter_index_from_boundary(q2_end_time_utc, timeframe=timeframe_min)
            q1_idx = (q2_idx - 1) % 4
        else:
            q1_idx = _quarter_index_from_boundary(q1_end_time_utc, timeframe=timeframe_min)
            q2_idx = (q1_idx + 1) % 4

        # skip wrap-around
        if q1_idx == 3 and q2_idx == 0:
            logger.info("_check_single_quarter_pair: skipping wrap-around transition Q4->Q1 for primary %s", primary_symbol)
            return False, None, None

        q1_label = f"Q{q1_idx + 1}"
        q2_label = f"Q{q2_idx + 1}"

        # compute broke_highs / broke_lows for group (use EPS)
        broke_highs = []
        broke_lows = []
        for sym, d in q1_q2_data.items():
            q1h = _safe_float(d['Q1'].get('high')) if hasattr(d['Q1'], 'get') else _safe_float(d['Q1']['high'])
            q2h = _safe_float(d['Q2'].get('high')) if hasattr(d['Q2'], 'get') else _safe_float(d['Q2']['high'])
            q1l = _safe_float(d['Q1'].get('low')) if hasattr(d['Q1'], 'get') else _safe_float(d['Q1']['low'])
            q2l = _safe_float(d['Q2'].get('low')) if hasattr(d['Q2'], 'get') else _safe_float(d['Q2']['low'])
            if q1h is None or q2h is None or q1l is None or q2l is None:
                continue
            if q2h > q1h + EPS:
                broke_highs.append(sym)
            if q2l < q1l - EPS:
                broke_lows.append(sym)

        # derive default "didn't break" lists globally (helpful to include, but we'll compute scenario-specific ones later)
        all_group_symbols = list(q1_q2_data.keys())
        global_didnt_break_highs = [s for s in all_group_symbols if s not in broke_highs]
        global_didnt_break_lows = [s for s in all_group_symbols if s not in broke_lows]

        # gather primary symbol Q values safely
        q1_high_primary = _safe_float(q1_q2_data[primary_symbol]['Q1'].get('high') if hasattr(q1_q2_data[primary_symbol]['Q1'],'get') else q1_q2_data[primary_symbol]['Q1']['high'])
        q2_high_primary = _safe_float(q1_q2_data[primary_symbol]['Q2'].get('high') if hasattr(q1_q2_data[primary_symbol]['Q2'],'get') else q1_q2_data[primary_symbol]['Q2']['high'])
        q1_low_primary = _safe_float(q1_q2_data[primary_symbol]['Q1'].get('low') if hasattr(q1_q2_data[primary_symbol]['Q1'],'get') else q1_q2_data[primary_symbol]['Q1']['low'])
        q2_low_primary = _safe_float(q1_q2_data[primary_symbol]['Q2'].get('low') if hasattr(q1_q2_data[primary_symbol]['Q2'],'get') else q1_q2_data[primary_symbol]['Q2']['low'])

        # if any of these are None, skip candidate
        if None in (q1_high_primary, q2_high_primary, q1_low_primary, q2_low_primary):
            logger.debug("_check_single_quarter_pair: missing primary numeric data for %s, skipping", primary_symbol)
            return False, None, None

        q1_time_str = _format_ts_for_target(q1_end_time_utc)
        q2_time_str = _format_ts_for_target(q2_end_time_utc)

        # --- HIGH-break branch ---
        if q2_high_primary > q1_high_primary + EPS:
            # For move_together: expect others to also have q2_high > q1_high; triggered if at least one other DID NOT break HIGH
            if group_type == "move_together":
                others_not = []
                for sym in other_symbols:
                    sym_q1_high = _safe_float(q1_q2_data[sym]['Q1'].get('high') if hasattr(q1_q2_data[sym]['Q1'],'get') else q1_q2_data[sym]['Q1']['high'])
                    sym_q2_high = _safe_float(q1_q2_data[sym]['Q2'].get('high') if hasattr(q1_q2_data[sym]['Q2'],'get') else q1_q2_data[sym]['Q2']['high'])
                    if sym_q1_high is None or sym_q2_high is None:
                        # if missing data, treat as "didn't break" conservatively
                        others_not.append(sym)
                        continue
                    if sym_q2_high <= sym_q1_high + EPS:
                        others_not.append(sym)
                if others_not:
                    # Build lists to pass to message: specific didn't-break-high (others_not) and global didn't-break-lows
                    msg = _build_pretty_alert_message(
                        group_id=group_id,
                        group_type=group_type,
                        timeframe_min=timeframe_min,
                        primary_symbol=primary_symbol,
                        q1_label=q1_label,
                        q2_label=q2_label,
                        q1_time_str=q1_time_str,
                        q2_time_str=q2_time_str,
                        action_verb="HIGH",
                        q1_val=q1_high_primary,
                        q2_val=q2_high_primary,
                        primary_direction_emoji="➡️",
                        didnt_break_high_list=others_not,
                        didnt_break_low_list=global_didnt_break_lows
                    )
                    logger.info("Triggered: %s", msg)
                    return True, msg, q2_end_time_utc

            # For reverse_moving: if primary (could be non-first) breaks HIGH, check that others held LOW (i.e., didn't break their lows)
            elif group_type == "reverse_moving":
                held_low = []
                for sym in other_symbols:
                    sym_q1_low = _safe_float(q1_q2_data[sym]['Q1'].get('low') if hasattr(q1_q2_data[sym]['Q1'],'get') else q1_q2_data[sym]['Q1']['low'])
                    sym_q2_low = _safe_float(q1_q2_data[sym]['Q2'].get('low') if hasattr(q1_q2_data[sym]['Q2'],'get') else q1_q2_data[sym]['Q2']['low'])
                    if sym_q1_low is None or sym_q2_low is None:
                        held_low.append(sym)
                        continue
                    if sym_q2_low >= sym_q1_low - EPS:
                        held_low.append(sym)
                if held_low:
                    # For message: show global didn't-break-highs and held_low as didn't-break-lows (relevant)
                    msg = _build_pretty_alert_message(
                        group_id=group_id,
                        group_type=group_type,
                        timeframe_min=timeframe_min,
                        primary_symbol=primary_symbol,
                        q1_label=q1_label,
                        q2_label=q2_label,
                        q1_time_str=q1_time_str,
                        q2_time_str=q2_time_str,
                        action_verb="HIGH",
                        q1_val=q1_high_primary,
                        q2_val=q2_high_primary,
                        primary_direction_emoji="➡️",
                        didnt_break_high_list=global_didnt_break_highs,
                        didnt_break_low_list=held_low
                    )
                    logger.info("Triggered: %s", msg)
                    return True, msg, q2_end_time_utc

        # --- LOW-break branch ---
        if q2_low_primary < q1_low_primary - EPS:
            # move_together: primary breaks low, others should also break low; trigger if some other did NOT break low.
            if group_type == "move_together":
                others_not = []
                for sym in other_symbols:
                    sym_q1_low = _safe_float(q1_q2_data[sym]['Q1'].get('low') if hasattr(q1_q2_data[sym]['Q1'],'get') else q1_q2_data[sym]['Q1']['low'])
                    sym_q2_low = _safe_float(q1_q2_data[sym]['Q2'].get('low') if hasattr(q1_q2_data[sym]['Q2'],'get') else q1_q2_data[sym]['Q2']['low'])
                    if sym_q1_low is None or sym_q2_low is None:
                        others_not.append(sym)
                        continue
                    if sym_q2_low >= sym_q1_low - EPS:
                        others_not.append(sym)
                if others_not:
                    msg = _build_pretty_alert_message(
                        group_id=group_id,
                        group_type=group_type,
                        timeframe_min=timeframe_min,
                        primary_symbol=primary_symbol,
                        q1_label=q1_label,
                        q2_label=q2_label,
                        q1_time_str=q1_time_str,
                        q2_time_str=q2_time_str,
                        action_verb="LOW",
                        q1_val=q1_low_primary,
                        q2_val=q2_low_primary,
                        primary_direction_emoji="⬇️",
                        didnt_break_high_list=global_didnt_break_highs,
                        didnt_break_low_list=others_not
                    )
                    logger.info("Triggered: %s", msg)
                    return True, msg, q2_end_time_utc

            # reverse_moving: primary breaks low, others should have held HIGH (i.e., didn't break highs)
            elif group_type == "reverse_moving":
                held_high = []
                for sym in other_symbols:
                    sym_q1_high = _safe_float(q1_q2_data[sym]['Q1'].get('high') if hasattr(q1_q2_data[sym]['Q1'],'get') else q1_q2_data[sym]['Q1']['high'])
                    sym_q2_high = _safe_float(q1_q2_data[sym]['Q2'].get('high') if hasattr(q1_q2_data[sym]['Q2'],'get') else q1_q2_data[sym]['Q2']['high'])
                    if sym_q1_high is None or sym_q2_high is None:
                        held_high.append(sym)
                        continue
                    if sym_q2_high <= sym_q1_high + EPS:
                        held_high.append(sym)
                if held_high:
                    msg = _build_pretty_alert_message(
                        group_id=group_id,
                        group_type=group_type,
                        timeframe_min=timeframe_min,
                        primary_symbol=primary_symbol,
                        q1_label=q1_label,
                        q2_label=q2_label,
                        q1_time_str=q1_time_str,
                        q2_time_str=q2_time_str,
                        action_verb="LOW",
                        q1_val=q1_low_primary,
                        q2_val=q2_low_primary,
                        primary_direction_emoji="⬇️",
                        didnt_break_high_list=held_high,
                        didnt_break_low_list=global_didnt_break_lows
                    )
                    logger.info("Triggered: %s", msg)
                    return True, msg, q2_end_time_utc

    except (KeyError, ValueError, TypeError) as e:
        logger.error(f"_check_single_quarter_pair: Error processing data for pair: {e}", exc_info=True)

    return False, None, None


# --------------------------
# Remaining code unchanged: caching, batching, processing coordinator, handlers...
# (I keep these functions intact to preserve behavior such as once-per-quarter gating)
# --------------------------

_SYMBOL_DATA_CACHE: dict = {}
_CACHE_TTL_SECONDS = 80
_BATCH_POINTER_FILE = "group_batch_pointer.json"
_BATCH_POINTER_KEY = "pointer"

def _load_batch_pointer() -> int:
    try:
        if os.path.exists(_BATCH_POINTER_FILE):
            with open(_BATCH_POINTER_FILE, 'r', encoding='utf-8') as f:
                j = json.load(f)
                return int(j.get(_BATCH_POINTER_KEY, 0))
    except Exception:
        logger.debug("Failed to read batch pointer, using 0", exc_info=True)
    return 0

def _save_batch_pointer(ptr: int) -> None:
    try:
        _atomic_write_json(_BATCH_POINTER_FILE, {_BATCH_POINTER_KEY: int(ptr)})
    except Exception:
        logger.debug("Failed to save batch pointer", exc_info=True)

def _cache_get(symbol: str):
    ent = _SYMBOL_DATA_CACHE.get(symbol)
    if not ent:
        return None
    if (datetime.now().timestamp() - ent.get("ts", 0)) > _CACHE_TTL_SECONDS:
        try:
            del _SYMBOL_DATA_CACHE[symbol]
        except Exception:
            pass
        return None
    return ent.get("df")

def _cache_set(symbol: str, df: pd.DataFrame):
    try:
        _SYMBOL_DATA_CACHE[symbol] = {"ts": datetime.now().timestamp(), "df": df}
    except Exception:
        logger.exception("_cache_set failed for %s", symbol)

def _all_groups_list() -> list:
    return list(GROUP_CHARACTERISTICS.keys())

def _total_unique_symbols() -> set:
    symset = set()
    for g in GROUP_CHARACTERISTICS.values():
        symset.update(g.get("symbols", []))
    return symset

def _select_group_batch(target_symbol_count: int, start_index: int) -> (list, int):
    groups = _all_groups_list()
    n = len(groups)
    if n == 0:
        return [], start_index
    selected = []
    seen = set()
    idx = start_index % n
    while True:
        gid = groups[idx]
        selected.append(gid)
        for s in GROUP_CHARACTERISTICS[gid].get("symbols", []):
            seen.add(s)
        if len(seen) >= target_symbol_count:
            idx = (idx + 1) % n
            break
        idx = (idx + 1) % n
        if idx % n == start_index % n:
            break
    new_pointer = idx % n
    return selected, new_pointer

async def _fetch_symbols_bulk(symbols: list[str]) -> dict:
    out = {}
    sem = asyncio.Semaphore(_SYMBOL_FETCH_CONCURRENCY)
    now_target = datetime.now(TIMEZONE_TARGET)

    async def _fetch_one(sym):
        async with sem:
            try:
                resolution = 1
                to_ts = int(now_target.astimezone(TIMEZONE_LITEFINANCE).timestamp())
                from_ts = int((now_target - timedelta(days=3)).astimezone(TIMEZONE_LITEFINANCE).timestamp())
                cached = _cache_get(sym)
                if cached is not None:
                    return sym, cached
                df = await _fetch_symbol_df(sym, resolution, from_ts, to_ts, _PER_SYMBOL_FETCH_TIMEOUT)
                if df is not None and not df.empty:
                    _cache_set(sym, df)
                return sym, df
            except Exception as e:
                logger.exception("_fetch_symbols_bulk: unexpected error for %s: %s", sym, e)
                return sym, None

    tasks = [asyncio.create_task(_fetch_one(sym)) for sym in symbols]
    done, pending = await asyncio.wait(tasks, timeout=_PER_ALERT_FETCH_TIMEOUT)
    for t in done:
        try:
            sym, df = t.result()
            out[sym] = df
        except Exception as e:
            logger.exception("_fetch_symbols_bulk: task result error: %s", e)
    for t in pending:
        t.cancel()
    await asyncio.gather(*pending, return_exceptions=True)
    return out

async def process_group_batch_and_run_alerts(batch_groups: list, alerts_map: dict, bot: Optional[object] = None):
    symbols_needed = []
    for g in batch_groups:
        gd = GROUP_CHARACTERISTICS.get(g)
        if not gd:
            continue
        for s in gd.get("symbols", []):
            if s not in symbols_needed:
                symbols_needed.append(s)

    if not symbols_needed:
        logger.debug("process_group_batch_and_run_alerts: no symbols for batch %s", batch_groups)
        return

    data_map_raw = await _fetch_symbols_bulk(symbols_needed)

    for alert_key, alert in list(alerts_map.items()):
        try:
            if not alert.get("active", False):
                continue
            group_id = alert.get("group_id")
            if group_id not in batch_groups:
                continue
            timeframe_min = alert.get("timeframe_min")
            user_id = alert.get("user_id")
            category = alert.get("category")
            group_info = find_group(category, group_id)
            if not group_info:
                continue
            symbols = group_info.get("symbols", [])
            group_type = group_info.get("type", "unknown")

            resampled_map = {}
            missing_symbols = []
            for sym in symbols:
                raw_df = data_map_raw.get(sym, None)
                if raw_df is None or (hasattr(raw_df, "empty") and raw_df.empty):
                    raw_df = _cache_get(sym)
                if raw_df is None or (hasattr(raw_df, "empty") and raw_df.empty):
                    missing_symbols.append(sym)
                    continue
                try:
                    res = await asyncio.to_thread(_resample_to_alert_timeframe, raw_df, timeframe_min)
                except Exception as e:
                    logger.exception("process_group_batch_and_run_alerts: resample error for %s: %s", sym, e)
                    missing_symbols.append(sym)
                    continue
                if res is None or (hasattr(res, 'empty') and res.empty):
                    missing_symbols.append(sym)
                else:
                    resampled_map[sym] = res

            if missing_symbols:
                logger.warning("process_group_batch_and_run_alerts: missing data for alert %s symbols: %s", alert_key, missing_symbols)
                continue

            primary_symbol = symbols[0]
            other_symbols = symbols[1:]
            triggered_results = _check_all_quarter_pairs(resampled_map, group_id, group_type, primary_symbol, other_symbols, timeframe_min)

            for (is_triggered, message, q2_end_time_utc) in triggered_results:
                if is_triggered and message:
                    quarter_key = None
                    try:
                        if q2_end_time_utc is not None:
                            q_idx = _quarter_index_from_boundary(q2_end_time_utc, timeframe=timeframe_min)
                            year = pd.to_datetime(q2_end_time_utc, utc=True).tz_convert(TIMEZONE_TARGET).year
                            quarter_key = f"{year}-tf{timeframe_min}-Q{q_idx+1}"
                        else:
                            now_local = datetime.now(TIMEZONE_TARGET)
                            q_idx = _quarter_index_from_boundary(pd.Timestamp(now_local.tzinfo and now_local), timeframe=timeframe_min)
                            year = now_local.year
                            quarter_key = f"{year}-tf{timeframe_min}-Q{q_idx+1}"
                    except Exception:
                        quarter_key = None

                    alerts_current = load_alerts()
                    current_entry = alerts_current.get(alert_key, {})
                    last_quarter = current_entry.get("last_trigger_quarter")
                    if last_quarter is not None and quarter_key is not None and last_quarter == quarter_key:
                        logger.info("Skipping trigger for %s because it already fired in same quarter %s", alert_key, quarter_key)
                        continue

                    if q2_end_time_utc is not None:
                        try:
                            sig = f"bar:{int(pd.to_datetime(q2_end_time_utc, utc=True).timestamp())}"
                        except Exception:
                            sig = f"msghash:{hash(message)}"
                    else:
                        sig = f"msghash:{hash(message)}"

                    last_sig = current_entry.get("last_trigger_signature")
                    if last_sig == sig:
                        logger.info("Skipping duplicate trigger for %s sig=%s", alert_key, sig)
                        continue

                    current_entry['last_trigger_signature'] = sig
                    if quarter_key is not None:
                        current_entry['last_trigger_quarter'] = quarter_key
                    current_entry['updated_at'] = datetime.now(timezone.utc).isoformat() + 'Z'
                    alerts_current[alert_key] = current_entry
                    save_alerts(alerts_current)

                    bg_name = f"trigger_handler::{alert_key}"
                    try:
                        _schedule_background_task(_handle_trigger_actions(alert_key, alert, message, symbols, timeframe_min, user_id, bot=bot), bg_name)
                    except Exception:
                        asyncio.create_task(_handle_trigger_actions(alert_key, alert, message, symbols, timeframe_min, user_id, bot=bot))

        except Exception as e:
            logger.exception("process_group_batch_and_run_alerts: error processing alert %s: %s", alert_key, e)

async def check_all_alerts_periodically_coordinator(bot: Optional[object] = None):
    try:
        logger.info("Periodic coordinator: selecting batch to process.")

        alerts = load_alerts() or {}
        active_alerts = {k: v for k, v in alerts.items() if v.get("active", False)}
        if not active_alerts:
            logger.debug("No active alerts to process.")
            return

        total_symbols = len(_total_unique_symbols())
        if total_symbols <= 0:
            total_symbols = 1
        pct = 0.10
        target_symbol_count = max(1, int(math.ceil(total_symbols * pct)))

        ptr = _load_batch_pointer()
        batch_groups, new_ptr = _select_group_batch(target_symbol_count, ptr)
        _save_batch_pointer(new_ptr)
        logger.info("Periodic coordinator: processing groups batch %s (target %d symbols)", batch_groups, target_symbol_count)

        await process_group_batch_and_run_alerts(batch_groups, alerts, bot=bot)

    except Exception as e:
        logger.exception("check_all_alerts_periodically_coordinator: unexpected error: %s", e)

async def _handle_trigger_actions(alert_key: str, alert_details: dict, message: str, symbols: list, timeframe_min: int, user_id: int, bot: Optional[object] = None):
    try:
        chart_data_list = await _generate_charts_for_symbols(symbols, timeframe_min)

        triggered_data = {
            "timestamp": datetime.now(timezone.utc).isoformat() + 'Z',
            "alert_key": alert_key,
            "alert_details": alert_details,
            "message": message,
            "charts": [
                {
                    "symbol": c.get('symbol'),
                    "timeframe": c.get('timeframe'),
                    "status": 'generated' if c.get('buffer') else c.get('error', 'error')
                } for c in chart_data_list
            ]
        }

        try:
            await asyncio.wait_for(asyncio.to_thread(save_triggered_alert_with_charts, triggered_data), timeout=_SAVE_IO_TIMEOUT)
        except asyncio.TimeoutError:
            logger.warning("_handle_trigger_actions: saving triggered alert timed out for %s", alert_key)
        except Exception as e:
            logger.exception("_handle_trigger_actions: save triggered alert error: %s", e)

        image_buffers = []
        for c in chart_data_list:
            buf = c.get('buffer')
            if buf is None:
                continue
            try:
                if hasattr(buf, 'read'):
                    try:
                        b = buf.read()
                    except Exception:
                        try:
                            buf.seek(0)
                            b = buf.read()
                        except Exception:
                            logger.warning("_handle_trigger_actions: unable to read buffer for chart %s", c.get('symbol'))
                            continue
                elif isinstance(buf, (bytes, bytearray)):
                    b = bytes(buf)
                else:
                    logger.warning("_handle_trigger_actions: chart buffer is unknown type for %s", c.get('symbol'))
                    continue

                if b:
                    image_buffers.append(b)
            except Exception:
                logger.exception("_handle_trigger_actions: error reading chart buffer for %s", c.get('symbol'))

        try:
            truncated = _truncate_message(message)
            await send_alert_notification(user_id, truncated, [], bot=bot)
        except Exception as e:
            logger.exception("_handle_trigger_actions: notifying user (text) failed: %s", e)
            try:
                await send_alert_notification(user_id, _truncate_message(str(message)), [], bot=bot)
            except Exception as e2:
                logger.exception("_handle_trigger_actions: notifying user (plain text fallback) failed: %s", e2)

        for b in image_buffers:
            try:
                await send_alert_notification(user_id, '', [b], bot=bot)
            except Exception as e:
                logger.exception("_handle_trigger_actions: sending chart failed: %s", e)

    except Exception as e:
        logger.exception("_handle_trigger_actions: unexpected error: %s", e)

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

def _generate_alert_key(user_id, category, group_id, timeframe_min):
    return f"{user_id}::{category}::{group_id}::{timeframe_min}"
