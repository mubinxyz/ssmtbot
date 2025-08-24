# services/alert_service.py
"""
Alert service updated to:
- Use get_ohlc() to compute previous-quarter HIGH/LOW.
- Use get_price() for current price checks (live detection).
- Poll ~20% of symbols every 15s (rotating batches) to reduce scraping load.
- Conservative fallbacks (use last resampled close if price is missing).
- Maintain live-only + one-time-per-quarter semantics.
"""
import asyncio
import json
import logging
import math
import os
import random
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd
import pytz

from utils.get_data import get_ohlc, get_price
from services.notify_user import send_alert_notification
from services.chart_service import generate_chart

logger = logging.getLogger(__name__)

# -------------------------
# Configuration
# -------------------------
ALERTS_STORE_FILE = "alerts_store.json"
TRIGGERED_ALERTS_WITH_CHARTS_FILE = "triggered_alerts_with_charts.json"

# Timezones
TIMEZONE_LITEFINANCE = pytz.FixedOffset(180)  # provider UTC+3
TIMEZONE_TARGET = pytz.timezone("Asia/Tehran")  # display (UTC+3:30)
TIMEZONE_UTC = pytz.utc

# Raw data resolutions
_SHORT_RAW_RESOLUTION = 1  # minutes for 5m/15m (we fetch 1m)
_LONG_RAW_RESOLUTION = 5   # minutes for 60m/240m (we fetch 5m)

SHORT_TFS = {5, 15}
LONG_TFS = {60, 240}

# Polling & caching for get_price (live price)
PRICE_POLL_INTERVAL_SECONDS = 15           # we run a poll every 15s for a batch
PRICE_BATCH_PCT = 0.20                    # poll 20% of symbols each poll
PRICE_CACHE_TTL_SECONDS = 75              # consider a cached price fresh if within 75s
PRICE_FETCH_CONCURRENCY = 3               # concurrency for get_price fetches
PRICE_FETCH_TIMEOUT = 8.0                 # seconds per get_price attempt
PRICE_FAILURES_BEFORE_COOLDOWN = 3
PRICE_COOLDOWN_BASE_SECONDS = 30          # base cooldown for flaky symbols (exponential backoff)

# Other timeouts
_PER_SYMBOL_FETCH_TIMEOUT = 20.0
_PER_ALERT_FETCH_TIMEOUT = 80.0
_CHART_GENERATION_TIMEOUT = 60.0
_SAVE_IO_TIMEOUT = 10.0

# Behavior
EPS = 1e-6
BAR_CLOSED_BUFFER_SECONDS = 1  # safety buffer when checking closedness of previous quarter bar
CACHE_TTL_SECONDS = 60

# -------------------------
# Groups
# -------------------------
GROUP_CHARACTERISTICS = {
    "dxy_eu_gu": {
        "label": "DXY / EURUSD / GBPUSD",
        "symbols": ["USDX", "EURUSD", "GBPUSD"],
        "type": "reverse_moving"
    },
    "dxy_aud_nzd": {
        "label": "DXY / AUDUSD / NZDUSD",
        "symbols": ["USDX", "AUDUSD", "NZDUSD"],
        "type": "reverse_moving"
    },
    "btc_eth_xrp": {
        "label": "BTCUSD / ETHUSD / XRPUSD",
        "symbols": ["BTCUSD", "ETHUSD", "XRPUSD"],
        "type": "move_together"
    },
    "spx_nq_ym": {
        "label": "S&P500 (SPX) / NASDAQ (NQ) / DOW (YM)",
        "symbols": ["SPX", "NQ", "YM"],
        "type": "move_together"
    },
    "dxy_xau_xag_aud": {
        "label": "DXY / XAU / XAG / AUD",
        "symbols": ["USDX", "XAUUSD", "XAGUSD", "AUDUSD"],
        "type": "reverse_moving"
    },
    # Energy bundle — oil first (reverse-moving vs USDCAD and DXY)
    "dxy_usdcad_owest_obrent": {
        "label": "WTI (West Texas) / BRENT / USDCAD / DXY",
        "symbols": ["WTI", "BRENT", "USDCAD", "USDX"],
        "type": "reverse_moving"
    },
}

def find_group(category: str, group_id: str):
    if category == "FOREX CURRENCIES" and group_id in GROUP_CHARACTERISTICS:
        return GROUP_CHARACTERISTICS[group_id]
    return None

# -------------------------
# Persistent storage helpers
# -------------------------
def load_alerts():
    if not os.path.exists(ALERTS_STORE_FILE):
        return {}
    try:
        with open(ALERTS_STORE_FILE, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        logger.exception("load_alerts: failed to read")
        return {}

def save_alerts(alerts: dict):
    tmp = f"{ALERTS_STORE_FILE}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(alerts, f, indent=2, ensure_ascii=False)
        os.replace(tmp, ALERTS_STORE_FILE)
    except Exception:
        logger.exception("save_alerts: failed to write")

def save_triggered_alert_with_charts(payload: dict):
    # append to a JSON list (atomic replace)
    data = []
    if os.path.exists(TRIGGERED_ALERTS_WITH_CHARTS_FILE):
        try:
            with open(TRIGGERED_ALERTS_WITH_CHARTS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f) or []
        except Exception:
            data = []
    data.append(payload)
    tmp = f"{TRIGGERED_ALERTS_WITH_CHARTS_FILE}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, TRIGGERED_ALERTS_WITH_CHARTS_FILE)
    except Exception:
        logger.exception("save_triggered_alert_with_charts failed")

# -------------------------
# Price polling / cache (get_price)
# -------------------------
_PRICE_CACHE = {}  # symbol -> {"price": float, "ts": epoch, "source": str}
_PRICE_FAILURES = {}  # symbol -> failure_count
_PRICE_COOLDOWN_UNTIL = {}  # symbol -> epoch ts

_price_poll_pointer = 0
_price_poll_lock = asyncio.Lock()

def _price_cache_get(symbol: str, max_age=PRICE_CACHE_TTL_SECONDS):
    ent = _PRICE_CACHE.get(symbol)
    if not ent:
        return None
    if time.time() - ent.get("ts", 0) > max_age:
        return None
    return ent

def _price_cache_set(symbol: str, price_obj: dict):
    _PRICE_CACHE[symbol] = {"price": float(price_obj.get("price")), "ts": time.time(), "source": price_obj.get("source")}

async def _fetch_price_for_symbol(symbol: str) -> Optional[dict]:
    """Call get_price in thread, with timeout and failure handling. Returns {'price':..., ...} or None"""
    now = time.time()
    cooldown = _PRICE_COOLDOWN_UNTIL.get(symbol)
    if cooldown and now < cooldown:
        logger.debug("Skipping price fetch for %s due to cooldown until %s", symbol, cooldown)
        return None

    try:
        coro = asyncio.to_thread(get_price, symbol)
        res = await asyncio.wait_for(coro, timeout=PRICE_FETCH_TIMEOUT)
        if res and res.get("price") is not None:
            # success: reset failures
            _PRICE_FAILURES.pop(symbol, None)
            _PRICE_COOLDOWN_UNTIL.pop(symbol, None)
            return res
        else:
            # treat as failure
            _PRICE_FAILURES[symbol] = _PRICE_FAILURES.get(symbol, 0) + 1
            failures = _PRICE_FAILURES[symbol]
            if failures >= PRICE_FAILURES_BEFORE_COOLDOWN:
                backoff = PRICE_COOLDOWN_BASE_SECONDS * (2 ** (failures - PRICE_FAILURES_BEFORE_COOLDOWN))
                _PRICE_COOLDOWN_UNTIL[symbol] = time.time() + backoff
                logger.warning("Price fetch for %s failed %d times. Putting on cooldown for %ds", symbol, failures, int(backoff))
    except asyncio.TimeoutError:
        _PRICE_FAILURES[symbol] = _PRICE_FAILURES.get(symbol, 0) + 1
        logger.warning("get_price timed out for %s", symbol)
    except Exception as e:
        _PRICE_FAILURES[symbol] = _PRICE_FAILURES.get(symbol, 0) + 1
        logger.warning("get_price error for %s: %s", symbol, e)
    return None

async def _price_poller_loop():
    """
    Background task. Every PRICE_POLL_INTERVAL_SECONDS it polls ~PRICE_BATCH_PCT of known symbols.
    Rotates through the symbol list so each symbol is polled in turn.
    """
    global _price_poll_pointer
    all_symbols = []
    for g in GROUP_CHARACTERISTICS.values():
        for s in g.get("symbols", []):
            if s not in all_symbols:
                all_symbols.append(s)

    if not all_symbols:
        logger.info("Price poller: no symbols configured")
        return

    total = len(all_symbols)
    batch_size = max(1, int(math.ceil(total * PRICE_BATCH_PCT)))

    logger.info("Price poller started: total_symbols=%d batch_size=%d interval=%ds", total, batch_size, PRICE_POLL_INTERVAL_SECONDS)

    sem = asyncio.Semaphore(PRICE_FETCH_CONCURRENCY)
    while True:
        try:
            # lock pointer update to be safe if poller restarts
            async with _price_poll_lock:
                start = _price_poll_pointer % total
                end = start + batch_size
                if end <= total:
                    batch = all_symbols[start:end]
                else:
                    # wrap
                    batch = all_symbols[start:total] + all_symbols[0:(end % total)]
                _price_poll_pointer = (start + batch_size) % total

            tasks = []
            async def _fetch_and_set(sym):
                async with sem:
                    res = await _fetch_price_for_symbol(sym)
                    if res:
                        try:
                            _price_cache_set(sym, res)
                        except Exception:
                            logger.exception("Failed to set price cache for %s", sym)

            for sym in batch:
                tasks.append(asyncio.create_task(_fetch_and_set(sym)))

            if tasks:
                done, pending = await asyncio.wait(tasks, timeout=PRICE_FETCH_TIMEOUT + 1)
                for p in pending:
                    p.cancel()
                await asyncio.gather(*pending, return_exceptions=True)

        except Exception:
            logger.exception("Price poller unexpected error")
        await asyncio.sleep(PRICE_POLL_INTERVAL_SECONDS)

# Ensure poller is scheduled once at startup by calling _schedule_background_task with this coroutine.

# -------------------------
# OHLC fetch & resample helpers (get_ohlc)
# -------------------------
def _convert_lf_time_to_target(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if 'datetime' not in df.columns:
        return pd.DataFrame()
    df2 = df.copy()
    df2['datetime'] = pd.to_datetime(df2['datetime'], utc=True)
    if df2['datetime'].dt.tz is None:
        df2['datetime'] = df2['datetime'].dt.tz_localize('UTC')
    df2['datetime'] = df2['datetime'].dt.tz_convert(TIMEZONE_LITEFINANCE).dt.tz_convert(TIMEZONE_TARGET)
    df2['timestamp'] = df2['datetime'].apply(lambda dt: int(dt.timestamp()))
    return df2

def _resample_to_alert_timeframe(raw_df: pd.DataFrame, alert_timeframe_min: int) -> pd.DataFrame:
    if raw_df is None or raw_df.empty:
        return pd.DataFrame()
    df = raw_df.copy()
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
    if df['datetime'].dt.tz is None:
        df['datetime'] = df['datetime'].dt.tz_localize('UTC')
    df = df.set_index('datetime')
    try:
        res = df.resample(f"{alert_timeframe_min}min", label='right', closed='right').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'timestamp': 'last'
        }).dropna().reset_index()
        if 'timestamp' not in res.columns or res['timestamp'].isna().all():
            res['timestamp'] = res['datetime'].dt.tz_convert(TIMEZONE_TARGET).apply(lambda dt: int(dt.timestamp()))
        return res
    except Exception:
        logger.exception("_resample_to_alert_timeframe failed")
        return pd.DataFrame()

# Batched OHLC fetch (per resolution)
async def _fetch_symbol_ohlc(symbol: str, resolution_min: int, from_ts: int, to_ts: int) -> Optional[pd.DataFrame]:
    try:
        coro = asyncio.to_thread(get_ohlc, symbol, resolution_min, from_ts, to_ts)
        df = await asyncio.wait_for(coro, timeout=_PER_SYMBOL_FETCH_TIMEOUT)
        if df is None or (hasattr(df, "empty") and df.empty):
            return None
        df_conv = _convert_lf_time_to_target(df)
        return df_conv
    except asyncio.TimeoutError:
        logger.warning("_fetch_symbol_ohlc timeout for %s @ res %d", symbol, resolution_min)
    except Exception:
        logger.exception("_fetch_symbol_ohlc error for %s @ res %d", symbol, resolution_min)
    return None

async def _fetch_symbols_bulk_for_resolution(symbols: list, resolution_min: int) -> dict:
    out = {}
    sem = asyncio.Semaphore(4)
    now_target = datetime.now(TIMEZONE_TARGET)
    to_ts = int(now_target.astimezone(TIMEZONE_LITEFINANCE).timestamp())
    from_ts = int((now_target - timedelta(days=3)).astimezone(TIMEZONE_LITEFINANCE).timestamp())

    async def _one(sym):
        async with sem:
            df = await _fetch_symbol_ohlc(sym, resolution_min, from_ts, to_ts)
            return sym, df

    tasks = [asyncio.create_task(_one(s)) for s in symbols]
    done, pending = await asyncio.wait(tasks, timeout=_PER_ALERT_FETCH_TIMEOUT)
    for t in done:
        try:
            sym, df = t.result()
            out[sym] = df
        except Exception:
            logger.exception("_fetch_symbols_bulk_for_resolution: task result error")
    for t in pending:
        t.cancel()
    await asyncio.gather(*pending, return_exceptions=True)
    return out

# -------------------------
# SSMT decision logic using current prices
# -------------------------
def _safe_float(v):
    try:
        return float(v)
    except Exception:
        return None

def _quarter_index_from_boundary(boundary_utc: pd.Timestamp, tz_name: str = "Asia/Tehran", timeframe: int = 15) -> int:
    if boundary_utc is None:
        return 0
    try:
        if getattr(boundary_utc, "tzinfo", None) is None:
            b = pd.Timestamp(boundary_utc).tz_localize("UTC")
        else:
            b = pd.Timestamp(boundary_utc).tz_convert("UTC")
        local = b.tz_convert(tz_name)
        h = local.hour; m = local.minute; mins = h*60 + m
        q1_start = 1*60 + 30; q2_start = 7*60 + 30; q3_start = 13*60 + 30; q4_start = 19*60 + 30
        if timeframe == 5:
            if q1_start <= mins < q2_start: quarter_start = q1_start
            elif q2_start <= mins < q3_start: quarter_start = q2_start
            elif q3_start <= mins < q4_start: quarter_start = q3_start
            else: quarter_start = q4_start
            segment = (mins - quarter_start) // 90
            return int(segment % 4)
        elif timeframe == 15:
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
    except Exception:
        return 0

def _build_pretty_alert_message(group_id: str, group_type: str, timeframe_min: int,
                               q1_label: str, q2_label: str,
                               primary_symbol: str, action_verb: str,
                               prev_q_val: float, current_price: float,
                               didnt_break_high_list: list, didnt_break_low_list: list) -> str:
    label = GROUP_CHARACTERISTICS.get(group_id, {}).get("label", group_id)
    title = f"🔔 *SSMT Alert — {label}*"
    meta = f"Group type: `{group_type}` | Timeframe: `{timeframe_min}m`"
    quarter_span = f"SSMT happened: {q1_label} → {q2_label}"
    action_line = f"{'➡️' if action_verb=='HIGH' else '⬇️'} *{primary_symbol} broke previous quarter {action_verb}* — previous `{_fmt(prev_q_val)}` → current `{_fmt(current_price)}`"
    prev_val_line = f"• Previous-quarter {action_verb}: `{_fmt(prev_q_val)}`"
    dh = ', '.join(didnt_break_high_list) if didnt_break_high_list else "None"
    dl = ', '.join(didnt_break_low_list) if didnt_break_low_list else "None"
    didnt_high_line = f"• Didn't break previous-quarter HIGH: {dh}"
    didnt_low_line = f"• Didn't break previous-quarter LOW: {dl}"
    parts = [title, meta, quarter_span, "――――――――――――――――――――――――――", action_line, prev_val_line, didnt_high_line, didnt_low_line]
    return "\n".join(parts)

def _fmt(val):
    try:
        s = f"{float(val):.5f}"
        if '.' in s:
            s = s.rstrip('0').rstrip('.')
        return s
    except Exception:
        return str(val)

def _evaluate_ssmt_using_current_prices(resampled_map: dict, price_cache_lookup, group_id: str, group_type: str, timeframe_min: int):
    """
    resampled_map: symbol -> resampled df (timeframe-level). Each df's last row is the last closed bar.
    price_cache_lookup: function(symbol)-> {'price':float,'ts':...} or None
    Returns list of tuples (message, q2_end_time_utc, quarter_key)
    """
    out = []
    # ensure all symbols have at least one closed bar (previous-quarter)
    symbols = list(resampled_map.keys())
    for s in symbols:
        df = resampled_map.get(s)
        if df is None or len(df) < 1:
            return out

    # For previous-quarter we use the last closed bar for each symbol (resampled_df.iloc[-1])
    q1_q2_data = {}
    for s, df in resampled_map.items():
        q1_bar = df.iloc[-1].to_dict()  # previous quarter bar
        q1_q2_data[s] = {'PREV': q1_bar}

    # Candidates: all symbols (treat first as reverse if group_type says so)
    candidate_list = symbols[:]  # primary candidate could be any symbol; loop all
    for candidate in candidate_list:
        others = [x for x in candidate_list if x != candidate]
        q1_bar = q1_q2_data[candidate]['PREV']
        try:
            q1_end_time_utc = pd.to_datetime(q1_bar['datetime'], utc=True)
        except Exception:
            q1_end_time_utc = None
        # quarter labels (previous quarter is q1; the SSMT happens in the next quarter)
        try:
            q1_idx = _quarter_index_from_boundary(q1_end_time_utc, timeframe=timeframe_min)
            q2_idx = (q1_idx + 1) % 4
            q1_label = f"Q{q1_idx+1}"
            q2_label = f"Q{q2_idx+1}"
            year = pd.to_datetime(q1_end_time_utc, utc=True).tz_convert(TIMEZONE_TARGET).year if q1_end_time_utc is not None else datetime.now(TIMEZONE_TARGET).year
            quarter_key = f"{year}-tf{timeframe_min}-Q{q2_idx+1}"
        except Exception:
            q1_label = "Q?"
            q2_label = "Q?"
            quarter_key = None

        # prev quarter high/low
        prev_high = _safe_float(q1_bar.get('high'))
        prev_low = _safe_float(q1_bar.get('low'))
        if prev_high is None or prev_low is None:
            continue

        # Get current prices for candidate and others (from price cache). If missing, fallback to last resampled close.
        def get_current_price(sym):
            pc = price_cache_lookup(sym)
            if pc and pc.get('price') is not None:
                return float(pc['price']), 'price_cache'
            # fallback: try resampled close
            df = resampled_map.get(sym)
            try:
                fallback_close = _safe_float(df.iloc[-1].get('close'))
                return fallback_close, 'last_resampled_close_fallback'
            except Exception:
                return None, 'no_price'

        primary_price, primary_src = get_current_price(candidate)
        if primary_price is None:
            # cannot evaluate candidate without primary price
            continue

        # Determine if primary "broke" prev quarter HIGH or LOW using current price
        broke_high_primary = primary_price > prev_high + EPS
        broke_low_primary = primary_price < prev_low - EPS

        # If primary didn't break in any direction, skip
        if not (broke_high_primary or broke_low_primary):
            continue

        # For other symbols compute whether they broke prev high/low using their current prices (same method)
        others_didnt_break_high = []
        others_didnt_break_low = []
        for s in others:
            s_price, s_src = get_current_price(s)
            if s_price is None:
                # conservative: treat as "didn't break" to allow alert to fire if appropriate
                others_didnt_break_high.append(s)
                others_didnt_break_low.append(s)
                continue
            # check high/low break relative to that symbol's previous-quarter (its own prev bar values)
            s_prev_high = _safe_float(q1_q2_data[s]['PREV'].get('high'))
            s_prev_low = _safe_float(q1_q2_data[s]['PREV'].get('low'))
            if s_prev_high is None or s_prev_low is None:
                others_didnt_break_high.append(s)
                others_didnt_break_low.append(s)
                continue
            # symbol broke its high?
            s_broke_high = s_price > s_prev_high + EPS
            s_broke_low = s_price < s_prev_low - EPS
            if not s_broke_high:
                others_didnt_break_high.append(s)
            if not s_broke_low:
                others_didnt_break_low.append(s)

        # Now apply group rules (using "broke" status computed from current price)
        # move_together: if primary broke HIGH and at least one other did NOT break HIGH -> alert
        # reverse_moving: primary could be reverse symbol or other; logic as you specified earlier.

        if broke_high_primary:
            # PRIMARY broke HIGH
            if group_type == "move_together":
                # others expected to break HIGH too; if at least one didn't -> alert
                if len(others_didnt_break_high) > 0:
                    msg = _build_pretty_alert_message(group_id, group_type, timeframe_min, q1_label, q2_label,
                                                      candidate, "HIGH", prev_high, primary_price,
                                                      others_didnt_break_high, others_didnt_break_low)
                    out.append((msg, q1_end_time_utc, quarter_key))
            elif group_type == "reverse_moving":
                # when someone (not necessarily first) breaks HIGH, others (incl first) should break LOW
                # alert if any other did NOT break LOW (i.e., held low)
                if len(others_didnt_break_low) > 0:
                    msg = _build_pretty_alert_message(group_id, group_type, timeframe_min, q1_label, q2_label,
                                                      candidate, "HIGH", prev_high, primary_price,
                                                      [s for s in q1_q2_data.keys() if s not in [candidate] and s not in [] and s not in []], # global didnt-break-highs not critical
                                                      others_didnt_break_low)
                    out.append((msg, q1_end_time_utc, quarter_key))

        if broke_low_primary:
            if group_type == "move_together":
                if len(others_didnt_break_low) > 0:
                    msg = _build_pretty_alert_message(group_id, group_type, timeframe_min, q1_label, q2_label,
                                                      candidate, "LOW", prev_low, primary_price,
                                                      [s for s in q1_q2_data.keys() if s not in []], # global didn't-break-highs cheap placeholder
                                                      others_didnt_break_low)
                    out.append((msg, q1_end_time_utc, quarter_key))
            elif group_type == "reverse_moving":
                if len(others_didnt_break_high) > 0:
                    msg = _build_pretty_alert_message(group_id, group_type, timeframe_min, q1_label, q2_label,
                                                      candidate, "LOW", prev_low, primary_price,
                                                      others_didnt_break_high,
                                                      [s for s in q1_q2_data.keys() if s not in []])
                    out.append((msg, q1_end_time_utc, quarter_key))

    return out

# -------------------------
# Coordinator & orchestration
# -------------------------
async def process_group_batch_and_run_alerts(batch_groups: list, alerts_map: dict, bot=None):
    """
    For each alert in batch_groups:
    - Fetch OHLC at appropriate raw resolution (1m for 5/15, 5m for 60/240)
    - Resample to timeframe to get previous-quarter bar (last closed bar)
    - Use live prices (from price cache) to decide "broke previous-quarter HIGH/LOW"
    - Persist last_triggered_quarter to enforce one-time-per-quarter
    """
    if not batch_groups:
        return

    # Collect which symbols are needed for short vs long raw fetch
    symbols_short = set()
    symbols_long = set()
    group_alerts_map = {}
    for alert_key, alert in alerts_map.items():
        if not alert.get("active"):
            continue
        gid = alert.get("group_id")
        if gid not in batch_groups:
            continue
        tf = alert.get("timeframe_min")
        group_alerts_map.setdefault(gid, []).append((alert_key, alert))
        syms = GROUP_CHARACTERISTICS.get(gid, {}).get("symbols", [])
        if tf in SHORT_TFS:
            symbols_short.update(syms)
        elif tf in LONG_TFS:
            symbols_long.update(syms)
        else:
            symbols_short.update(syms)

    # fetch raw data per resolution
    fetch_short = {}
    fetch_long = {}
    if symbols_short:
        fetch_short = await _fetch_symbols_bulk_for_resolution(list(symbols_short), _SHORT_RAW_RESOLUTION)
    if symbols_long:
        fetch_long = await _fetch_symbols_bulk_for_resolution(list(symbols_long), _LONG_RAW_RESOLUTION)

    # For each group's alerts, evaluate
    for gid, alerts_list in group_alerts_map.items():
        group_info = GROUP_CHARACTERISTICS.get(gid)
        if not group_info:
            continue
        syms = group_info.get("symbols", [])
        group_type = group_info.get("type", "unknown")

        for (alert_key, alert) in alerts_list:
            if not alert.get("active"):
                continue
            timeframe_min = alert.get("timeframe_min")
            user_id = alert.get("user_id")
            if timeframe_min in SHORT_TFS:
                raw_map = fetch_short
            elif timeframe_min in LONG_TFS:
                raw_map = fetch_long
            else:
                raw_map = fetch_short

            # Build resampled_map for this timeframe (symbol -> resampled df)
            resampled_map = {}
            missing = []
            for s in syms:
                raw_df = raw_map.get(s)
                if raw_df is None or raw_df.empty:
                    missing.append(s)
                    continue
                res = _resample_to_alert_timeframe(raw_df, timeframe_min)
                if res is None or res.empty:
                    missing.append(s)
                    continue
                # require at least one closed bar (previous-quarter)
                if len(res) < 1:
                    missing.append(s)
                    continue
                resampled_map[s] = res

            if missing:
                logger.debug("Skipping alert %s group %s timeframe %d due to missing symbols %s", alert_key, gid, timeframe_min, missing)
                continue

            # Evaluate SSMT using current prices obtained via price cache
            def price_lookup(sym):
                return _price_cache_get(sym, max_age=PRICE_CACHE_TTL_SECONDS)

            triggered_msgs = _evaluate_ssmt_using_current_prices(resampled_map, price_lookup, gid, group_type, timeframe_min)
            # triggered_msgs is list of (message, q1_end_time_utc, quarter_key)
            for (msg, q1_end_time_utc, quarter_key) in triggered_msgs:
                # ensure live-only: previous-quarter must be in the past and not current
                if q1_end_time_utc is not None:
                    q1_ts = int(pd.to_datetime(q1_end_time_utc, utc=True).timestamp())
                    now_ts = int(datetime.now(timezone.utc).timestamp())
                    # require prev-quarter bar to be fully closed well before now
                    if (q1_ts + BAR_CLOSED_BUFFER_SECONDS) >= now_ts:
                        logger.debug("Skipping trigger %s because prev-quarter bar %s is not safely in the past (now %s)", alert_key, q1_ts, now_ts)
                        continue

                # one-time-per-quarter enforcement
                alerts_store = load_alerts()
                entry = alerts_store.get(alert_key, {})
                last_quarter = entry.get("last_triggered_quarter")
                if last_quarter is not None and quarter_key is not None and last_quarter == quarter_key:
                    logger.debug("Skipping trigger %s because already fired in quarter %s", alert_key, quarter_key)
                    continue

                # avoid duplicate same-bar triggers
                last_bar_ts = entry.get("last_triggered_bar_ts")
                try:
                    q_ts_to_store = int(pd.to_datetime(q1_end_time_utc, utc=True).timestamp()) if q1_end_time_utc is not None else None
                except Exception:
                    q_ts_to_store = None
                if last_bar_ts is not None and q_ts_to_store is not None:
                    try:
                        if q_ts_to_store <= int(last_bar_ts):
                            logger.debug("Skipping trigger %s because bar %s already triggered", alert_key, last_bar_ts)
                            continue
                    except Exception:
                        pass

                # duplicate signature guard
                sig = f"price_break:{gid}:{timeframe_min}:{quarter_key}:{hash(msg)}"
                last_sig = entry.get("last_trigger_signature")
                if last_sig == sig:
                    logger.debug("Skipping duplicate signature for %s", alert_key)
                    continue

                # persist metadata
                entry['last_trigger_signature'] = sig
                if q_ts_to_store is not None:
                    entry['last_triggered_bar_ts'] = int(q_ts_to_store)
                if quarter_key is not None:
                    entry['last_triggered_quarter'] = quarter_key
                entry['updated_at'] = datetime.now(timezone.utc).isoformat() + "Z"
                alerts_store[alert_key] = entry
                save_alerts(alerts_store)

                # schedule notification in background
                try:
                    asyncio.create_task(_handle_trigger_actions(alert_key, alert, msg, syms, timeframe_min, user_id, bot=bot))
                except Exception:
                    logger.exception("Failed to schedule _handle_trigger_actions for %s", alert_key)

# -------------------------
# Trigger handler: charts & notify
# -------------------------
async def _generate_charts_for_symbols(symbols: list, timeframe_min: int):
    try:
        if asyncio.iscoroutinefunction(generate_chart):
            coro = generate_chart(symbols, timeframe=timeframe_min)
            charts = await asyncio.wait_for(coro, timeout=_CHART_GENERATION_TIMEOUT)
        else:
            charts = await asyncio.wait_for(asyncio.to_thread(generate_chart, symbols, timeframe_min), timeout=_CHART_GENERATION_TIMEOUT)
        result = []
        for i, buf in enumerate(charts):
            sym = symbols[i] if i < len(symbols) else "__combined__"
            result.append({"symbol": sym, "timeframe": timeframe_min, "buffer": buf})
        return result
    except Exception:
        logger.exception("_generate_charts_for_symbols failed")
        return []

async def _handle_trigger_actions(alert_key, alert_details, message, symbols, timeframe_min, user_id, bot=None):
    try:
        charts = await _generate_charts_for_symbols(symbols, timeframe_min)
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
            "alert_key": alert_key,
            "alert_details": alert_details,
            "message": message,
            "charts": [{"symbol": c.get("symbol"), "timeframe": c.get("timeframe"), "status": "generated" if c.get("buffer") else "error"} for c in charts]
        }
        try:
            await asyncio.to_thread(save_triggered_alert_with_charts, record)
        except Exception:
            logger.exception("_handle_trigger_actions: failed to save triggered record")

        # send text message first
        try:
            truncated = message if len(message) <= 3900 else message[:3897] + "..."
            await send_alert_notification(user_id, truncated, [], bot=bot)
        except Exception:
            logger.exception("_handle_trigger_actions: error sending text")

        # send images
        for c in charts:
            buf = c.get("buffer")
            if not buf:
                continue
            try:
                if hasattr(buf, "read"):
                    try:
                        b = buf.read()
                    except Exception:
                        buf.seek(0); b = buf.read()
                elif isinstance(buf, (bytes, bytearray)):
                    b = bytes(buf)
                else:
                    continue
                if b:
                    await send_alert_notification(user_id, "", [b], bot=bot)
            except Exception:
                logger.exception("_handle_trigger_actions: error sending chart")

    except Exception:
        logger.exception("_handle_trigger_actions unexpected error")

# -------------------------
# API helpers consumed by bot.py
# -------------------------
def _generate_alert_key(user_id, category, group_id, timeframe_min):
    return f"{user_id}::{category}::{group_id}::{timeframe_min}"

def set_ssmt_alert(user_id: int, group_id: str, timeframe_min: int, category: str = "FOREX CURRENCIES"):
    alerts = load_alerts()
    key = _generate_alert_key(user_id, category, group_id, timeframe_min)
    if key in alerts:
        alerts[key]["active"] = True
        alerts[key]["updated_at"] = datetime.now(timezone.utc).isoformat() + "Z"
    else:
        alerts[key] = {
            "user_id": user_id,
            "category": category,
            "group_id": group_id,
            "timeframe_min": timeframe_min,
            "active": True,
            "created_at": datetime.now(timezone.utc).isoformat() + "Z"
        }
    save_alerts(alerts)
    logger.info("Alert set/activated %s", key)

def deactivate_ssmt_alert(user_id: int, group_id: str, timeframe_min: int, category: str = "FOREX CURRENCIES") -> bool:
    alerts = load_alerts()
    key = _generate_alert_key(user_id, category, group_id, timeframe_min)
    if key in alerts and alerts[key].get("active"):
        alerts[key]["active"] = False
        alerts[key]["updated_at"] = datetime.now(timezone.utc).isoformat() + "Z"
        save_alerts(alerts)
        return True
    return False

def get_active_alerts():
    alerts = load_alerts()
    return {k: v for k, v in alerts.items() if v.get("active", False)}

async def check_all_alerts_periodically_coordinator(bot=None):
    """
    Call this periodically (recommended every 10-60s depending on load).
    This is the entrypoint used by bot.py.
    """
    try:
        alerts = load_alerts() or {}
        active_alerts = {k: v for k, v in alerts.items() if v.get("active", False)}
        if not active_alerts:
            logger.debug("No active alerts")
            return

        total_symbols = len({s for g in GROUP_CHARACTERISTICS.values() for s in g.get("symbols", [])}) or 1
        target_symbol_count = max(1, math.ceil(total_symbols * 0.10))

        # simple rotating pointer persisted in file
        pointer = 0
        pointer_file = "group_batch_pointer.json"
        try:
            if os.path.exists(pointer_file):
                with open(pointer_file, "r", encoding="utf-8") as f:
                    j = json.load(f)
                    pointer = int(j.get("pointer", 0))
        except Exception:
            pointer = 0

        # select groups batch
        groups = list(GROUP_CHARACTERISTICS.keys())
        n = len(groups)
        if n == 0:
            return
        selected = []
        seen = set()
        idx = pointer % n
        while True:
            gid = groups[idx]
            selected.append(gid)
            for s in GROUP_CHARACTERISTICS[gid].get("symbols", []):
                seen.add(s)
            if len(seen) >= target_symbol_count:
                idx = (idx + 1) % n
                break
            idx = (idx + 1) % n
            if idx % n == pointer % n:
                break
        new_ptr = idx % n
        try:
            with open(pointer_file + ".tmp", "w", encoding="utf-8") as f:
                json.dump({"pointer": new_ptr}, f)
            os.replace(pointer_file + ".tmp", pointer_file)
        except Exception:
            pass

        await process_group_batch_and_run_alerts(selected, active_alerts, bot=bot)

    except Exception:
        logger.exception("Coordinator failed")

# -------------------------
# Startup helper to schedule price poller
# -------------------------
def _schedule_background_task(coro, name: Optional[str] = None):
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
            logger.exception("_schedule_background_task: create asyncio task failed")
    def runner():
        try:
            asyncio.run(coro)
        except Exception:
            logger.exception("_schedule_background_task runner exception")
    import threading
    t = threading.Thread(target=runner, daemon=True, name=(name or "bg_task_thread"))
    t.start()
    return t

# Call the price poller at import/startup (safe to call multiple times; it will run in background)
try:
    _schedule_background_task(_price_poller_loop(), name="price_poller_loop")
except Exception:
    logger.exception("Failed to schedule price poller loop")
