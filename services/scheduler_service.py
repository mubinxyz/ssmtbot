# services/scheduler_service.py
"""
Scheduler service for SSMT alerts.

- Fetches 1-minute OHLC using utils.get_data.get_ohlc (runs in background thread).
- Resamples 1-min data to each alert's timeframe (e.g. 15min).
- Evaluates rules via services.alert_service.
- When violated, optionally generates charts via chart_service.generate_chart(...) and
  calls send_alert_callback(user_id, message, charts).

Usage:
    asyncio.create_task(run_scheduler(send_alert_callback, chart_service=chart_service, interval_seconds=55))
"""
import asyncio
import datetime as dt
import logging
from typing import Callable, Dict, List, Optional

import pandas as pd

from utils.get_data import get_ohlc
from services import alert_service

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# How far back (in seconds) to fetch 1-minute data to be safe for quarter boundaries.
# 48 hours is large but safe; reduce if you want lower network usage.
FETCH_BACK_SECONDS = 48 * 3600


def _ensure_df_index_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure df has a proper datetime index. If it has a 'datetime' column, set that as index.
    Return df with a timezone-naive UTC index.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()

    # If there is a 'datetime' column, set it as index
    if "datetime" in df.columns:
        try:
            df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
            df = df.set_index("datetime")
        except Exception:
            pass

    # Convert index to datetime
    try:
        idx = pd.to_datetime(df.index)
    except Exception:
        df.index = pd.to_datetime(df.index, errors="coerce")
        idx = df.index

    # If tz-aware, convert to UTC and drop tzinfo; else assume naive means UTC.
    if getattr(idx, "tz", None) is not None:
        try:
            idx = idx.tz_convert("UTC").tz_localize(None)
        except Exception:
            idx = idx.tz_localize(None)
    df.index = idx
    return df


def _resample_to_tf(df: pd.DataFrame, tf_min: int) -> pd.DataFrame:
    """
    Resample a 1-minute dataframe to timeframe `tf_min` minutes using OHLC aggregation.
    Assumes df.index is datetime-like (UTC-naive).
    """
    if df is None or df.empty:
        return pd.DataFrame()

    df = _ensure_df_index_datetime(df)
    if df.empty:
        return pd.DataFrame()

    # prefer lower-case standard names
    cols_lower = {c.lower(): c for c in df.columns}
    rule = f"{int(tf_min)}min"

    # If explicit open/high/low/close exist, use them
    if {"open", "high", "low", "close"}.issubset(set(cols_lower.keys())):
        # rename to lowercase to aggregate consistently
        df2 = df.rename(columns={orig: orig.lower() for orig in df.columns})
        agg = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
        }
        res = df2.resample(rule).agg(agg).dropna()
        return res

    # fallback: assume first 4 columns are OHLC
    try:
        ohlc = df.iloc[:, :4].copy()
        ohlc.columns = ["open", "high", "low", "close"]
        res = ohlc.resample(rule).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
        }).dropna()
        return res
    except Exception:
        return pd.DataFrame()


async def _fetch_1min_for_symbol(symbol: str, to_date_ts: Optional[int] = None) -> pd.DataFrame:
    """
    Fetch 1-minute OHLC for symbol using utils.get_data.get_ohlc in a thread.
    Returns a DataFrame (possibly empty).
    """
    to_date_ts = to_date_ts or int(dt.datetime.utcnow().timestamp())
    from_date = to_date_ts - FETCH_BACK_SECONDS
    try:
        # get_ohlc is blocking (requests) so run in a thread
        df = await asyncio.to_thread(get_ohlc, symbol, 1, from_date, to_date_ts)
        if df is None:
            return pd.DataFrame()
        return _ensure_df_index_datetime(df)
    except Exception as e:
        logger.exception("Error fetching 1-min for %s: %s", symbol, e)
        return pd.DataFrame()


async def run_scheduler(
    send_alert_callback: Callable[[int, str, Optional[List[object]]], object],
    chart_service: Optional[object] = None,
    interval_seconds: int = 55
):
    """
    Main scheduler loop.

    send_alert_callback(user_id:int, message:str, charts:List[io.BytesIO]|None) - may be async or sync.
    chart_service: optional object exposing async generate_chart(symbols:List[str], timeframe:int)
    interval_seconds: scheduler frequency (defaults to 55 seconds).
    """
    logger.info("Starting scheduler: interval_seconds=%s", interval_seconds)

    while True:
        try:
            active_alerts = alert_service.get_active_alerts()
            if not active_alerts:
                logger.debug("No active alerts. Sleeping for %s seconds.", interval_seconds)
                await asyncio.sleep(interval_seconds)
                continue

            # Build a set of all symbols needed by active alerts and prepare a list of alerts to evaluate
            symbols_needed = set()
            alerts_to_process = []
            for a in active_alerts:
                user_id = int(a.get("user_id"))
                category = a.get("category")
                group_id = a.get("group_id") or a.get("group")
                tf_min = int(a.get("timeframe_min") or a.get("timeframe") or 15)

                # try to find group definition
                group_def = None
                if category and group_id:
                    group_def = alert_service.find_group(category, group_id)
                if group_def is None:
                    # try flexible resolver
                    cat, matched_gid = alert_service._find_group_category_by_group_id_flexible(group_id or "")
                    if cat and matched_gid:
                        group_def = alert_service.find_group(cat, matched_gid)
                        category = cat
                        group_id = matched_gid

                if group_def is None:
                    logger.debug("Skipping unknown group: category=%s id=%s", category, group_id)
                    continue

                alerts_to_process.append((user_id, category, group_def, tf_min))
                symbols_needed.update(group_def.get("symbols", []))

            if not alerts_to_process:
                logger.debug("No valid alerts to process. Sleeping...")
                await asyncio.sleep(interval_seconds)
                continue

            # fetch 1-minute data concurrently for needed symbols
            now_ts = int(dt.datetime.utcnow().timestamp())
            fetch_tasks = {s: asyncio.create_task(_fetch_1min_for_symbol(s, now_ts)) for s in symbols_needed}

            # wait for all fetches to complete (they run in background threads)
            await asyncio.gather(*fetch_tasks.values(), return_exceptions=True)

            # gather raw 1-min data
            raw_1min_data: Dict[str, pd.DataFrame] = {s: (fetch_tasks[s].result() if not fetch_tasks[s].cancelled() else pd.DataFrame()) for s in fetch_tasks}

            # Evaluate each alert individually, resampling to the alert's timeframe
            for (user_id, category, group_def, tf_min) in alerts_to_process:
                symbols = group_def.get("symbols", [])
                market_data_resampled: Dict[str, pd.DataFrame] = {}
                missing = False

                for s in symbols:
                    df1 = raw_1min_data.get(s)
                    if df1 is None or df1.empty:
                        logger.debug("Missing 1-min data for %s while evaluating alert %s/%s", s, category, group_def.get("id"))
                        missing = True
                        break
                    resampled = _resample_to_tf(df1, tf_min)
                    if resampled is None or resampled.empty:
                        logger.debug("Resampled data empty for %s tf=%s", s, tf_min)
                        missing = True
                        break
                    market_data_resampled[s] = resampled

                if missing:
                    continue

                # find the rule function and evaluate
                rule_name = group_def.get("rule")
                rule_fn = alert_service._RULE_DISPATCH.get(rule_name)
                if rule_fn is None:
                    logger.debug("No rule function for group %s", group_def.get("id"))
                    continue

                try:
                    violated, message = rule_fn(group_def.get("symbols", []), tf_min, market_data_resampled)
                except Exception as ex:
                    logger.exception("Rule %s raised while evaluating alert %s: %s", rule_name, group_def.get("id"), ex)
                    continue

                if violated:
                    logger.info("Alert violated for user=%s group=%s tf=%s", user_id, group_def.get("id"), tf_min)
                    charts = None
                    if chart_service is not None:
                        try:
                            # chart_service.generate_chart expected to be async
                            charts = await chart_service.generate_chart(group_def.get("symbols", []), timeframe=tf_min)
                        except Exception as e:
                            logger.debug("Chart generation failed for %s: %s", group_def.get("id"), e)
                            charts = None

                    # prepare message and send
                    title = f"🚨 ALERT: {group_def.get('label')} ({tf_min}min)"
                    payload = f"{title}\n\n{message}"

                    try:
                        maybe_awaitable = send_alert_callback(user_id, payload, charts)
                        if asyncio.iscoroutine(maybe_awaitable):
                            await maybe_awaitable
                    except Exception as e:
                        logger.exception("send_alert_callback failed for user %s: %s", user_id, e)

            # finished processing all alerts this cycle

        except Exception as e:
            logger.exception("Scheduler loop error: %s", e)

        # finally sleep until next iteration
        await asyncio.sleep(interval_seconds)
