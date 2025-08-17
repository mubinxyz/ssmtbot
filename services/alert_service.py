# services/alert_service.py
"""
Alert service — generalized group rule implementation supporting 3 or more symbols.
First symbol is considered the 'index' (reverse) and the rest are peers.

Changes in this revision:
- Aliases _trio_quarter_core -> _group_quarter_core (back-compat).
- Use strict comparisons to determine "new low" / "new high" direction to avoid
  ambiguous situations where both directions appear true because of equality.
"""
import asyncio
import json
import logging
from typing import List, Dict, Any, Optional, Callable, Tuple
from contextlib import suppress
import datetime as dt
import re

import pandas as pd
from dateutil import tz

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Persistence
PATH_ALERTS_JSON = "alerts_store.json"
_active_alerts: Dict[str, Dict[str, Any]] = {}

# Example categories and groups (keep as before)
CATEGORIES = [
    "FOREX CURRENCIES",
    "FUTURES ( STOCKS )",
    "CRYPTO ( USD )",
    "CRYPTO ( USDT )",
    "METALS",
]

GROUPS = {
    "FOREX CURRENCIES": [
        {"id": "dxy_eu_gu", "label": "DXY, EURUSD, GBPUSD", "symbols": ["DXY", "EURUSD", "GBPUSD"], "rule": "dxy_eu_gu_rule"},
        {"id": "dxy_chf_jpy", "label": "DXY, CHF, JPY", "symbols": ["DXY", "CHF", "JPY"], "rule": "dxy_xxx_rule"},
        {"id": "dxy_aud_nzd", "label": "DXY, AUD, NZD", "symbols": ["DXY", "AUD", "NZD"], "rule": "dxy_xxx_rule"},
    ],
    "FUTURES ( STOCKS )": [
        {"id": "spx_nq_ym", "label": "SPX / NQ / YM", "symbols": ["SPX", "NQ", "YM"], "rule": "index_trio_rule"},
        {"id": "es_nq_dow", "label": "ES / NQ / DOW", "symbols": ["ES", "NQ", "DOW"], "rule": "index_trio_rule"},
        {"id": "spx_dow_nq", "label": "SPX / DOW / NQ", "symbols": ["SPX", "DOW", "NQ"], "rule": "index_trio_rule"},
    ],
    "CRYPTO ( USD )": [
        {"id": "btc_eth_xrp", "label": "BTCUSD / ETHUSD / XRPUSD", "symbols": ["BTCUSD", "ETHUSD", "XRPUSD"], "rule": "crypto_trio_rule"},
        {"id": "btc_eth_total", "label": "BTCUSD / ETHUSD / TOTAL", "symbols": ["BTCUSD", "ETHUSD", "TOTAL"], "rule": "crypto_trio_rule"},
        {"id": "btc_xrp_doge", "label": "BTCUSD / XRPUSD / DOGEUSD", "symbols": ["BTCUSD", "XRPUSD", "DOGEUSD"], "rule": "crypto_trio_rule"},
    ],
    "CRYPTO ( USDT )": [
        {"id": "btc_eth_xrp_usdt", "label": "BTCUSDT / ETHUSDT / XRPUSDT", "symbols": ["BTCUSDT", "ETHUSDT", "XRPUSDT"], "rule": "crypto_trio_rule"},
        {"id": "btc_eth_total_usdt", "label": "BTCUSDT / ETHUSDT / TOTALUSDT", "symbols": ["BTCUSDT", "ETHUSDT", "TOTALUSDT"], "rule": "crypto_trio_rule"},
        {"id": "btc_xrp_doge_usdt", "label": "BTCUSDT / XRPUSDT / DOGEUSDT", "symbols": ["BTCUSDT", "XRPUSDT", "DOGEUSDT"], "rule": "crypto_trio_rule"},
    ],
    "METALS": [
        {"id": "dxy_xau_xag_aud", "label": "DXY / XAU / XAG / AUD", "symbols": ["DXY", "XAU", "XAG", "AUD"], "rule": "metals_rule"},
        {"id": "xau_xag_aud", "label": "XAU / XAG / AUD", "symbols": ["XAU", "XAG", "AUD"], "rule": "metals_rule"},
        {"id": "dxy_xau_aud", "label": "DXY / XAU / AUD", "symbols": ["DXY", "XAU", "AUD"], "rule": "metals_rule"},
    ]
}
# ---------------------------
# Persistence helpers
# ---------------------------
def _save_alerts_to_disk(path: str = PATH_ALERTS_JSON) -> None:
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(_active_alerts, f, indent=2, default=str)
    except Exception as e:
        logger.debug("Failed to persist alerts to disk: %s", e)


def _load_alerts_from_disk(path: str = PATH_ALERTS_JSON) -> None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
            _active_alerts.clear()
            for k, v in loaded.items():
                _active_alerts[k] = v
    except FileNotFoundError:
        return
    except Exception as e:
        logger.warning("Failed to load alerts from disk: %s", e)


with suppress(Exception):
    _load_alerts_from_disk()


# ---------------------------
# UI helpers (same as before)
# ---------------------------
def get_main_menu_keyboard() -> List[List[Dict[str, str]]]:
    rows = []
    for cat in CATEGORIES:
        rows.append([{"text": cat, "callback_data": f"cat::{cat}"}])
    return rows


def get_category_keyboard(category: str) -> List[List[Dict[str, str]]]:
    rows = []
    groups = GROUPS.get(category, [])
    for g in groups:
        rows.append([{"text": g["label"], "callback_data": f"group::{category}::{g['id']}"}])
    rows.append([{"text": "Back", "callback_data": "back::main"}])
    return rows


def get_timeframe_keyboard(category: str, group_id: str) -> List[List[Dict[str, str]]]:
    tf_list = [
        {"text": "15min ssmt", "callback_data": f"timeframe::{category}::{group_id}::15"},
        {"text": "1hour ssmt", "callback_data": f"timeframe::{category}::{group_id}::60"},
        {"text": "Back", "callback_data": f"back::group::{category}::{group_id}"}
    ]
    return [[x] for x in tf_list]


def get_group_action_keyboard(category: str, group_id: str, timeframe_min: int) -> List[List[Dict[str, str]]]:
    rows = [
        [{"text": f"Charts ({timeframe_min}min)","callback_data": f"charts::{category}::{group_id}::{timeframe_min}"}],
        [{"text": f"Activate alert ({timeframe_min}min)","callback_data": f"activate::{category}::{group_id}::{timeframe_min}"}],
        [{"text": f"Deactivate alert ({timeframe_min}min)","callback_data": f"deactivate::{category}::{group_id}::{timeframe_min}"}],
        [{"text": "Back", "callback_data": f"back::timeframe::{category}::{group_id}"}]
    ]
    return rows


# ---------------------------
# Backwards-compatible API helpers
# ---------------------------
def _make_alert_key(user_id: int, category: str, group_id: str, timeframe_min: int) -> str:
    return f"{user_id}::{category}::{group_id}::{timeframe_min}"


def _normalize_group_id_input(raw_gid: str) -> str:
    if not raw_gid:
        return ""
    gid = raw_gid.strip().lower()
    prefixes = ["forex_", "futures_", "future_", "crypto_", "crypto_usdt_", "crypto_usd_", "metals_", "metal_"]
    for p in prefixes:
        if gid.startswith(p):
            gid = gid[len(p):]
            break
    for p in ("group_", "g_"):
        if gid.startswith(p):
            gid = gid[len(p):]
            break
    return gid


def _find_group_category_by_group_id_flexible(group_input: str) -> Tuple[Optional[str], Optional[str]]:
    if not group_input:
        return None, None

    raw = str(group_input).strip()
    raw_lower = raw.lower()

    # exact id
    for cat, groups in GROUPS.items():
        for g in groups:
            if g.get("id", "").lower() == raw_lower:
                return cat, g["id"]

    # normalized
    normalized = _normalize_group_id_input(raw)
    if normalized and normalized != raw_lower:
        for cat, groups in GROUPS.items():
            for g in groups:
                if g.get("id", "").lower() == normalized:
                    return cat, g["id"]

    # label substring
    for cat, groups in GROUPS.items():
        for g in groups:
            if raw_lower in g.get("label", "").lower():
                return cat, g["id"]

    # id substring
    for cat, groups in GROUPS.items():
        for g in groups:
            if normalized in g.get("id", "").lower() or raw_lower in g.get("id", "").lower():
                return cat, g["id"]

    return None, None


def _parse_timeframe_to_int(tf: Any) -> Optional[int]:
    if tf is None:
        return None
    if isinstance(tf, int):
        return int(tf)
    if isinstance(tf, str):
        m = re.search(r"(\d+)", tf)
        if m:
            return int(m.group(1))
    try:
        return int(tf)
    except Exception:
        return None


def set_ssmt_alert(
    user_id: int,
    category: Optional[str] = None,
    group: Optional[str] = None,
    group_id: Optional[str] = None,
    timeframe: Optional[Any] = None,
    timeframe_min: Optional[int] = None,
    **kwargs
) -> Dict[str, Any]:
    gid_input = group_id if group_id is not None else group

    resolved_category = category
    resolved_gid = gid_input

    if resolved_gid is not None and resolved_category is None:
        cat, matched_gid = _find_group_category_by_group_id_flexible(resolved_gid)
        if cat is not None:
            resolved_category = cat
            resolved_gid = matched_gid
            logger.debug("Auto-resolved category '%s' and group_id '%s' for input '%s'", resolved_category, resolved_gid, gid_input)

    if resolved_gid is None:
        raise ValueError("Missing 'group' or 'group_id' parameter.")

    if resolved_category is None:
        raise ValueError(f"Could not resolve category for group id '{gid_input}'. Please provide 'category' explicitly.")

    tf_min = timeframe_min if timeframe_min is not None else _parse_timeframe_to_int(timeframe)
    if tf_min is None:
        raise ValueError("Could not parse timeframe; provide 'timeframe' or 'timeframe_min' (e.g. 15).")

    key = _make_alert_key(user_id, resolved_category, resolved_gid, tf_min)
    alert = {
        "user_id": int(user_id),
        "category": resolved_category,
        "group_id": resolved_gid,
        "timeframe_min": int(tf_min),
        "active": True,
        "created_at": dt.datetime.utcnow().isoformat() + "Z",
    }
    _active_alerts[key] = alert
    _save_alerts_to_disk()
    logger.info("Set alert %s for user %s", key, user_id)
    return alert


def deactivate_ssmt_alert(
    user_id: int,
    category: Optional[str] = None,
    group: Optional[str] = None,
    group_id: Optional[str] = None,
    timeframe: Optional[Any] = None,
    timeframe_min: Optional[int] = None,
    **kwargs
) -> bool:
    gid_input = group_id if group_id is not None else group

    resolved_category = category
    resolved_gid = gid_input

    if resolved_gid is not None and resolved_category is None:
        cat, matched_gid = _find_group_category_by_group_id_flexible(resolved_gid)
        if cat is not None:
            resolved_category = cat
            resolved_gid = matched_gid
            logger.debug("Auto-resolved category '%s' and group_id '%s' for input '%s'", resolved_category, resolved_gid, gid_input)

    if resolved_gid is None:
        logger.debug("deactivate_ssmt_alert missing group; nothing to do.")
        return False
    if resolved_category is None:
        logger.debug("deactivate_ssmt_alert could not resolve category for group %r", gid_input)
        return False

    tf_min = timeframe_min if timeframe_min is not None else _parse_timeframe_to_int(timeframe)
    if tf_min is None:
        logger.debug("deactivate_ssmt_alert: could not parse timeframe from %r/%r", timeframe, timeframe_min)
        return False

    key = _make_alert_key(user_id, resolved_category, resolved_gid, tf_min)
    if key in _active_alerts:
        _active_alerts[key]["active"] = False
        _save_alerts_to_disk()
        logger.info("Deactivated alert %s", key)
        return True

    logger.debug("Alert key not found: %s", key)
    return False


def get_active_alerts() -> List[Dict[str, Any]]:
    return [v for v in _active_alerts.values() if v.get("active")]


# ---------------------------
# Quarter helpers (UTC+3:30)
# ---------------------------
LOCAL_OFFSET_MINUTES = 3 * 60 + 30  # 3:30 offset in minutes


def _quarter_boundaries_for_date(date: dt.date, local_offset_minutes: int = LOCAL_OFFSET_MINUTES) -> List[dt.datetime]:
    local_tz = tz.tzoffset(None, local_offset_minutes * 60)
    utc_tz = tz.tzutc()
    local_times = [(1, 30), (7, 30), (13, 30), (19, 30)]

    out = []
    for hh, mm in local_times:
        local_dt = dt.datetime(year=date.year, month=date.month, day=date.day, hour=hh, minute=mm, tzinfo=local_tz)
        utc_dt = local_dt.astimezone(utc_tz).replace(tzinfo=None)
        out.append(utc_dt)
    nd = date + dt.timedelta(days=1)
    local_next = dt.datetime(year=nd.year, month=nd.month, day=nd.day, hour=1, minute=30, tzinfo=local_tz)
    utc_next = local_next.astimezone(utc_tz).replace(tzinfo=None)
    out.append(utc_next)
    return out


def _get_consecutive_quarters_for_timestamp(ts_utc: dt.datetime, local_offset_minutes: int = LOCAL_OFFSET_MINUTES) -> Tuple[Tuple[dt.datetime, dt.datetime, int], Tuple[dt.datetime, dt.datetime, int]]:
    utc_tz = tz.tzutc()
    if ts_utc.tzinfo is None:
        ts_utc_aware = ts_utc.replace(tzinfo=utc_tz)
    else:
        ts_utc_aware = ts_utc.astimezone(utc_tz)
    local_tz = tz.tzoffset(None, local_offset_minutes * 60)
    local_dt = ts_utc_aware.astimezone(local_tz)
    local_date = local_dt.date()

    b = _quarter_boundaries_for_date(local_date, local_offset_minutes)
    j = None
    for i in range(4):
        if (ts_utc >= b[i]) and (ts_utc < b[i + 1]):
            j = i
            break
    if j is None:
        if ts_utc < b[0]:
            prev_day = local_date - dt.timedelta(days=1)
            prev_b = _quarter_boundaries_for_date(prev_day, local_offset_minutes)
            prev_start, prev_end = prev_b[3], prev_b[4]
            cur_start, cur_end = b[0], b[1]
            return (prev_start, prev_end, 4), (cur_start, cur_end, 1)
        if ts_utc >= b[4]:
            next_day = local_date + dt.timedelta(days=1)
            next_b = _quarter_boundaries_for_date(next_day, local_offset_minutes)
            for i in range(4):
                if ts_utc >= next_b[i] and ts_utc < next_b[i+1]:
                    k = i
                    break
            if k == 0:
                return (b[3], b[4], 4), (next_b[0], next_b[1], 1)
            else:
                return (next_b[k-1], next_b[k], k), (next_b[k], next_b[k+1], k+1)

    if j == 0:
        prev_day = local_date - dt.timedelta(days=1)
        prev_b = _quarter_boundaries_for_date(prev_day, local_offset_minutes)
        prev_start, prev_end = prev_b[3], prev_b[4]
        prev_qnum = 4
        cur_start, cur_end = b[0], b[1]
        cur_qnum = 1
    else:
        prev_start, prev_end = b[j - 1], b[j]
        prev_qnum = j
        cur_start, cur_end = b[j], b[j + 1]
        cur_qnum = j + 1

    return (prev_start, prev_end, prev_qnum), (cur_start, cur_end, cur_qnum)


# ---------------------------
# Data extraction helper
# ---------------------------
def _get_interval_ohlc(df: pd.DataFrame, start: dt.datetime, end: dt.datetime) -> Dict[str, Optional[float]]:
    if df is None or df.empty:
        return {"open": None, "high": None, "low": None, "close": None}

    idx = df.index
    if getattr(idx, "tz", None) is None:
        mask = (idx >= start) & (idx < end)
    else:
        start_t = start.replace(tzinfo=tz.tzutc()).astimezone(idx.tz)
        end_t = end.replace(tzinfo=tz.tzutc()).astimezone(idx.tz)
        mask = (idx >= start_t) & (idx < end_t)

    sliced = df.loc[mask]
    if sliced.empty:
        return {"open": None, "high": None, "low": None, "close": None}

    cols_lower = {c.lower(): c for c in sliced.columns}
    try:
        open_col = cols_lower.get("open", list(sliced.columns)[0])
        high_col = cols_lower.get("high", list(sliced.columns)[1] if sliced.shape[1] > 1 else open_col)
        low_col = cols_lower.get("low", list(sliced.columns)[2] if sliced.shape[1] > 2 else high_col)
        close_col = cols_lower.get("close", list(sliced.columns)[-1])
        o = float(sliced[open_col].iloc[0])
        h = float(sliced[high_col].max())
        l = float(sliced[low_col].min())
        c = float(sliced[close_col].iloc[-1])
        return {"open": o, "high": h, "low": l, "close": c}
    except Exception:
        return {"open": None, "high": None, "low": None, "close": None}


# ---------------------------
# Group core extraction (works for any number of symbols >= 3)
# ---------------------------
def _group_quarter_core(symbols: List[str], market_data: Dict[str, pd.DataFrame]) -> Tuple[Optional[Dict[str, Dict[str, Dict[str, Optional[float]]]]], Optional[int], Optional[int], Optional[str]]:
    """
    Extract prev & cur OHLC intervals for ALL provided symbols.
    Returns (values, prev_qnum, cur_qnum, error_msg)
    values: {symbol: {"prev": {...}, "cur": {...}}}
    """
    if len(symbols) < 3:
        return None, None, None, "Requires at least 3 symbols (index + 2 peers)"

    for s in symbols:
        if s not in market_data or market_data[s] is None or market_data[s].empty:
            return None, None, None, f"Missing market data for {s}"

    try:
        combined_max = max([market_data[s].index.max() for s in symbols])
    except Exception:
        return None, None, None, "Failed to determine latest timestamp"

    if combined_max is None:
        return None, None, None, "No timestamps found in data"

    if hasattr(combined_max, "tzinfo") and combined_max.tzinfo is not None:
        combined_max = combined_max.astimezone(tz.tzutc()).replace(tzinfo=None)

    (prev_start, prev_end, prev_qnum), (cur_start, cur_end, cur_qnum) = _get_consecutive_quarters_for_timestamp(combined_max)

    values: Dict[str, Dict[str, Dict[str, Optional[float]]]] = {}
    for s in symbols:
        df = market_data[s]
        prev = _get_interval_ohlc(df, prev_start, prev_end)
        cur = _get_interval_ohlc(df, cur_start, cur_end)
        values[s] = {"prev": prev, "cur": cur}

    insufficient = [s for s, v in values.items() if any(x is None for x in (v["prev"]["high"], v["prev"]["low"], v["cur"]["high"], v["cur"]["low"]))]
    if insufficient:
        return None, None, None, f"Insufficient interval data for: {insufficient}"

    return values, prev_qnum, cur_qnum, None


# Backwards-compatibility alias for older tests/code
_trio_quarter_core = _group_quarter_core
_trio_quarter_core.__doc__ = "Alias -> _group_quarter_core (back-compat)."


def _fmt(v: Optional[float]) -> str:
    return f"{v:.6g}" if v is not None else "None"


# ---------------------------
# Generalized group rule (first symbol is index/reverse, remaining are peers)
# ---------------------------
def _rule_group_generalized(symbols: List[str], timeframe_min: int, market_data: Dict[str, pd.DataFrame]) -> Tuple[bool, str]:
    """
    Generalized rule for groups where the first symbol is the index (reverse) and the rest are peers.
    Accepts 3 or more symbols (A, B1, B2, ...).
    Patterns implemented (uses strict comparisons to establish direction):
      - A makes new LOW (cur_low < prev_low) but one or more peers fail to make new HIGH (cur_high <= prev_high) -> violation
      - Peers mismatch: some peers make new HIGH (cur_high > prev_high) while others do not -> violation
      - A makes new HIGH (cur_high > prev_high) but one or more peers fail to make new LOW (cur_low >= prev_low) -> violation
    Returns (violated, message)
    """
    if len(symbols) < 3:
        return False, "Group rule requires at least 3 symbols"

    values, prev_qnum, cur_qnum, err = _group_quarter_core(symbols, market_data)
    if err:
        return False, err

    A = symbols[0]
    peers = symbols[1:]

    try:
        A_prev_low = values[A]["prev"]["low"]
        A_cur_low = values[A]["cur"]["low"]
        A_prev_high = values[A]["prev"]["high"]
        A_cur_high = values[A]["cur"]["high"]
    except Exception:
        return False, "Failed to extract index values"

    # peers values
    peer_prev_high = {}
    peer_cur_high = {}
    peer_prev_low = {}
    peer_cur_low = {}
    for p in peers:
        peer_prev_high[p] = values[p]["prev"]["high"]
        peer_cur_high[p] = values[p]["cur"]["high"]
        peer_prev_low[p] = values[p]["prev"]["low"]
        peer_cur_low[p] = values[p]["cur"]["low"]

    # Determine direction using strict comparisons (avoid ambiguous both-true cases)
    A_new_low = (A_cur_low is not None and A_prev_low is not None and A_cur_low < A_prev_low)
    A_new_high = (A_cur_high is not None and A_prev_high is not None and A_cur_high > A_prev_high)

    # peers: strict definitions of making a new high/low
    peer_new_high = {p: (peer_cur_high[p] is not None and peer_prev_high[p] is not None and peer_cur_high[p] > peer_prev_high[p]) for p in peers}
    peer_new_low = {p: (peer_cur_low[p] is not None and peer_prev_low[p] is not None and peer_cur_low[p] < peer_prev_low[p]) for p in peers}

    violations: List[str] = []

    # Pattern 1: A new low but at least one peer did not make new high
    if A_new_low:
        failed_peers = [p for p, ok in peer_new_high.items() if not ok]
        if failed_peers:
            violations.append(f"{A} made new LOW (Q{prev_qnum}->{cur_qnum}) but peers failed to make new HIGH: {', '.join(failed_peers)}")

    # Pattern 2: peer high mismatch - any peer made new high while another didn't
    peers_with_high = [p for p, ok in peer_new_high.items() if ok]
    peers_without_high = [p for p, ok in peer_new_high.items() if not ok]
    if peers_with_high and peers_without_high:
        violations.append(f"Peer HIGH mismatch (Q{prev_qnum}->{cur_qnum}): made_high=[{', '.join(peers_with_high)}] missing_high=[{', '.join(peers_without_high)}]")

    # Pattern 3: A new high but at least one peer did not make new low
    if A_new_high:
        failed_low_peers = [p for p, ok in peer_new_low.items() if not ok]
        if failed_low_peers:
            violations.append(f"{A} made new HIGH (Q{prev_qnum}->{cur_qnum}) but peers failed to make new LOW: {', '.join(failed_low_peers)}")

    violated = len(violations) > 0

    # Build message: header, per-symbol numeric lines, then violations (if any)
    lines: List[str] = []
    lines.append(f"Group rule check ({', '.join(symbols)}) comparing Q{prev_qnum} -> Q{cur_qnum}")
    lines.append(f"{A}: prev_low={_fmt(A_prev_low)} cur_low={_fmt(A_cur_low)} prev_high={_fmt(A_prev_high)} cur_high={_fmt(A_cur_high)}")
    for p in peers:
        lines.append(f"{p}: prev_low={_fmt(peer_prev_low[p])} cur_low={_fmt(peer_cur_low[p])} prev_high={_fmt(peer_prev_high[p])} cur_high={_fmt(peer_cur_high[p])}")

    if violations:
        lines.append("VIOLATIONS:")
        lines.extend(violations)
    else:
        lines.append("OK — no violations detected according to the generalized group criteria.")

    return violated, "\n".join(lines)


# Backwards-compatible rename for earlier function names/tests
_trio_quarter_core = _group_quarter_core
_rule_trio_generalized = _rule_group_generalized
_rule_dxy_eu_gu_violation = _rule_group_generalized
_rule_dxy_xxx_alias = _rule_group_generalized


def _rule_index_trio_continuation(symbols: List[str], timeframe_min: int, market_data: Dict[str, pd.DataFrame]) -> Tuple[bool, str]:
    values, prev_qnum, cur_qnum, err = _group_quarter_core(symbols, market_data)
    if err:
        return False, err

    lines = [f"Index continuation check (comparing Q{prev_qnum} -> Q{cur_qnum})"]
    failures = []
    for s in symbols[:3]:
        prev_open = values[s]["prev"]["open"]
        prev_close = values[s]["prev"]["close"]
        cur_close = values[s]["cur"]["close"]
        if None in (prev_open, prev_close, cur_close):
            lines.append(f"{s}: insufficient data")
            failures.append(s)
            continue
        q1_trend = prev_close - prev_open
        if q1_trend > 0:
            ok = cur_close >= prev_close
            desc = "Qprev close <= cur close (bullish continuation)"
        elif q1_trend < 0:
            ok = cur_close <= prev_close
            desc = "Qprev close >= cur close (bearish continuation)"
        else:
            ok = True
            desc = "flat prev quarter -> no expectation"
        lines.append(f"{s}: prev_open={_fmt(prev_open)} prev_close={_fmt(prev_close)} cur_close={_fmt(cur_close)} -> { 'OK' if ok else 'FAIL'} ({desc})")
        if not ok:
            failures.append(s)

    return (len(failures) > 0), "\n".join(lines)


def _rule_crypto_trio_follow_total(symbols: List[str], timeframe_min: int, market_data: Dict[str, pd.DataFrame]) -> Tuple[bool, str]:
    if len(symbols) < 3:
        return False, "Crypto trio requires at least 3 symbols"

    values, prev_qnum, cur_qnum, err = _group_quarter_core(symbols, market_data)
    if err:
        return False, err

    total = symbols[2]
    total_prev = values[total]["prev"]["close"]
    total_cur = values[total]["cur"]["close"]
    if None in (total_prev, total_cur):
        return False, "Insufficient TOTAL data"
    total_sign = 1 if (total_cur - total_prev) > 0 else (-1 if (total_cur - total_prev) < 0 else 0)

    violated_any = False
    lines = [f"Crypto TOTAL change Q{prev_qnum}->{cur_qnum}: {_fmt(total_prev)} -> {_fmt(total_cur)}"]
    for s in symbols[:2]:
        prev_c = values[s]["prev"]["close"]
        cur_c = values[s]["cur"]["close"]
        if None in (prev_c, cur_c):
            lines.append(f"{s}: insufficient data")
            violated_any = True
            continue
        sign = 1 if (cur_c - prev_c) > 0 else (-1 if (cur_c - prev_c) < 0 else 0)
        ok = (total_sign == 0) or (sign == total_sign)
        lines.append(f"{s}: {_fmt(prev_c)} -> {_fmt(cur_c)} -> {'OK' if ok else 'FAIL'}")
        if not ok:
            violated_any = True

    return violated_any, "\n".join(lines)


def _rule_metals_inverse_to_dxy(symbols: List[str], timeframe_min: int, market_data: Dict[str, pd.DataFrame]) -> Tuple[bool, str]:
    if "DXY" not in symbols:
        return False, "Metals rule requires DXY in symbols."

    needed = ["DXY"]
    metals = [s for s in ("XAU", "XAG") if s in symbols]
    needed.extend(metals[:2])

    missing = [s for s in needed if s not in market_data or market_data[s] is None or market_data[s].empty]
    if missing:
        return False, f"Missing market data for {missing} — cannot evaluate rule."

    combined_max = max([market_data[s].index.max() for s in needed])
    if combined_max is None:
        return False, "No timestamps available"

    if hasattr(combined_max, "tzinfo") and combined_max.tzinfo is not None:
        combined_max = combined_max.astimezone(tz.tzutc()).replace(tzinfo=None)

    (prev_start, prev_end, prev_qnum), (cur_start, cur_end, cur_qnum) = _get_consecutive_quarters_for_timestamp(combined_max)

    dxy_prev = _get_interval_ohlc(market_data["DXY"], prev_start, prev_end)
    dxy_cur = _get_interval_ohlc(market_data["DXY"], cur_start, cur_end)
    if None in (dxy_prev["close"], dxy_cur["close"]):
        return False, "Insufficient DXY data."

    dxy_sign = 1 if (dxy_cur["close"] - dxy_prev["close"]) > 0 else (-1 if (dxy_cur["close"] - dxy_prev["close"]) < 0 else 0)

    lines = [f"DXY change Q{prev_qnum}->{cur_qnum}: {_fmt(dxy_prev['close'])} -> {_fmt(dxy_cur['close'])} (sign {dxy_sign})"]
    failures = []
    for metal in metals:
        m_prev = _get_interval_ohlc(market_data[metal], prev_start, prev_end)
        m_cur = _get_interval_ohlc(market_data[metal], cur_start, cur_end)
        if None in (m_prev["close"], m_cur["close"]):
            lines.append(f"{metal}: insufficient data")
            failures.append(metal)
            continue
        m_sign = 1 if (m_cur["close"] - m_prev["close"]) > 0 else (-1 if (m_cur["close"] - m_prev["close"]) < 0 else 0)
        ok = (dxy_sign == 0) or (m_sign == -dxy_sign)
        lines.append(f"{metal}: {_fmt(m_prev['close'])} -> {_fmt(m_cur['close'])} sign {m_sign} -> {'OK' if ok else 'FAIL'}")
        if not ok:
            failures.append(metal)

    violated = len(failures) > 0
    return violated, "\n".join(lines)


# ---------------------------
# Dispatch mapping
# ---------------------------
_RULE_DISPATCH = {
    "dxy_eu_gu_rule": _rule_group_generalized,
    "dxy_xxx_rule": _rule_group_generalized,
    "index_trio_rule": _rule_index_trio_continuation,
    "crypto_trio_rule": _rule_crypto_trio_follow_total,
    "metals_rule": _rule_metals_inverse_to_dxy,
}


# ---------------------------
# Evaluation & notification
# ---------------------------
async def evaluate_active_alerts(
    market_data: Dict[str, pd.DataFrame],
    send_alert_callback: Optional[Callable[[int, str, Optional[List[Any]]], Any]] = None,
    chart_service: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    fired = []
    active = get_active_alerts()
    if not active:
        return fired

    for alert in active:
        try:
            user_id = alert["user_id"]
            category = alert["category"]
            group_id = alert["group_id"]
            tf_min = int(alert["timeframe_min"])

            group_def = None
            for g in GROUPS.get(category, []):
                if g["id"] == group_id:
                    group_def = g
                    break
            if group_def is None:
                logger.debug("Unknown group %s in category %s", group_id, category)
                continue

            rule_name = group_def.get("rule")
            rule_fn = _RULE_DISPATCH.get(rule_name)
            if rule_fn is None:
                logger.debug("No rule function registered for %s (group %s)", rule_name, group_id)
                continue

            violated, message = rule_fn(group_def["symbols"], tf_min, market_data)
            if violated:
                charts = None
                if chart_service is not None:
                    try:
                        charts = await chart_service.generate_chart(group_def["symbols"], timeframe=tf_min)
                    except Exception as e:
                        logger.debug("Failed to generate charts for alert %s: %s", group_id, e)
                        charts = None

                if send_alert_callback is not None:
                    try:
                        maybe_awaitable = send_alert_callback(user_id, f"ALERT: {group_def['label']} - {tf_min}min\n\n{message}", charts)
                        if asyncio.iscoroutine(maybe_awaitable):
                            await maybe_awaitable
                    except Exception as e:
                        logger.exception("send_alert_callback failed for user %s: %s", user_id, e)

                fired.append({
                    "user_id": user_id,
                    "category": category,
                    "group_id": group_id,
                    "timeframe_min": tf_min,
                    "message": message,
                    "charts": charts
                })

        except Exception as e:
            logger.exception("Error evaluating alert %s: %s", alert, e)

    return fired


def find_group(category: str, group_id: str) -> Optional[Dict[str, Any]]:
    groups = GROUPS.get(category, [])
    for g in groups:
        if g["id"] == group_id:
            return g
    return None


async def example_send_alert_callback(user_id: int, message: str, charts: Optional[List[Any]] = None) -> None:
    logger.info("Would send to %s: %s (charts=%s)", user_id, message[:80], bool(charts))
