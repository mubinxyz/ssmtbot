# services/chart_service.py

import io
import time
import logging
from contextlib import suppress
from typing import List, Optional
import datetime as dt
import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
from dateutil import tz as dateutil_tz
from utils.get_data import get_ohlc
from utils.normalize_data import normalize_symbol

# ===== Add these 2 lines at the top, before any matplotlib.pyplot import =====
import matplotlib
matplotlib.use("Agg")   # use non-GUI backend so Tkinter is never involved
# ==========================================================================

logger = logging.getLogger("services.chart_service")
logging.basicConfig(level=logging.INFO)

# ---------- CONFIG ----------
SOURCE_TZ = dateutil_tz.tzoffset(None, 3 * 3600)   # provider = UTC+3
LOCAL_TZ_NAME = "Asia/Tehran"
LOCAL_OFFSET_MINUTES = 210                         # fallback +3:30
NEAR_TOLERANCE_HOURS = 3                           # fallback tolerance used to decide "nearby" candle
QUARTER_COLORS = ["C2", "C3", "C4", "C5"]
TRUEOPEN_COLOR = "C6"
TRUEOPEN_STYLE = "--"
# ---------------------------

def _parse_datetimes_to_utc(series: pd.Series, source_tz=SOURCE_TZ) -> pd.Series:
    """
    Deterministic parsing:
      - epoch numbers -> detect seconds vs milliseconds then localize to source_tz then convert to UTC
      - naive datetimes -> localize to source_tz then convert to UTC
      - tz-aware datetimes -> convert to UTC
    Returns tz-aware UTC Series.
    """
    if pd.api.types.is_integer_dtype(series) or pd.api.types.is_float_dtype(series):
        # detect seconds vs milliseconds
        try:
            non_na = series.dropna()
            maxv = int(non_na.max()) if not non_na.empty else 0
        except Exception:
            maxv = 0
        unit = "ms" if maxv > 10**12 else "s"
        s = pd.to_datetime(series, unit=unit, errors="coerce")
        try:
            return s.dt.tz_localize(source_tz).dt.tz_convert("UTC")
        except Exception:
            fallback = dateutil_tz.tzoffset(None, LOCAL_OFFSET_MINUTES * 60)
            return s.dt.tz_localize(fallback).dt.tz_convert("UTC")

    s = pd.to_datetime(series, errors="coerce")
    if s.dt.tz is None:
        try:
            return s.dt.tz_localize(source_tz).dt.tz_convert("UTC")
        except Exception:
            fallback = dateutil_tz.tzoffset(None, LOCAL_OFFSET_MINUTES * 60)
            return s.dt.tz_localize(fallback).dt.tz_convert("UTC")
    else:
        return s.dt.tz_convert("UTC")


def _quarter_boundaries_utc_for_index(index: pd.DatetimeIndex,
                                      local_offset_minutes: int = LOCAL_OFFSET_MINUTES,
                                      tz_name: str = LOCAL_TZ_NAME) -> List[pd.Timestamp]:
    """
    Build 15m-quarter boundaries in local tz (Asia/Tehran = UTC+3:30) at:
       01:30, 07:30, 13:30, 19:30 and next-day 01:30
    Return tz-aware UTC pandas.Timestamp list.
    """
    if index is None or index.empty:
        return []

    if index.tz is None:
        idx_utc = index.tz_localize("UTC")
    else:
        idx_utc = index.tz_convert("UTC")

    try:
        idx_local = idx_utc.tz_convert(tz_name)
        local_tz = idx_local.tz
    except Exception:
        local_tz = dateutil_tz.tzoffset(None, local_offset_minutes * 60)
        idx_local = idx_utc.tz_convert(local_tz)

    start_local = (idx_local.min() - pd.Timedelta(days=1)).normalize()
    end_local = (idx_local.max() + pd.Timedelta(days=1)).normalize()
    days = pd.date_range(start=start_local, end=end_local, freq="D", tz=idx_local.tz)

    local_times = [(1, 30), (7, 30), (13, 30), (19, 30)]
    boundaries_utc = []

    for day in days:
        for hh, mm in local_times:
            try:
                local_ts = pd.Timestamp(year=day.year, month=day.month, day=day.day, hour=hh, minute=mm, tz=idx_local.tz)
                boundaries_utc.append(local_ts.tz_convert("UTC"))
            except Exception:
                local_dt = dt.datetime(year=day.year, month=day.month, day=day.day, hour=hh, minute=mm, tzinfo=local_tz)
                boundaries_utc.append(pd.Timestamp(local_dt.astimezone(dateutil_tz.tzutc())))
        # Also append next-day 01:30 to close the last quarter
        nd = day + pd.Timedelta(days=1)
        try:
            local_next = pd.Timestamp(year=nd.year, month=nd.month, day=nd.day, hour=1, minute=30, tz=idx_local.tz)
            boundaries_utc.append(local_next.tz_convert("UTC"))
        except Exception:
            local_dt_next = dt.datetime(year=nd.year, month=nd.month, day=nd.day, hour=1, minute=30, tzinfo=local_tz)
            boundaries_utc.append(pd.Timestamp(local_dt_next.astimezone(dateutil_tz.tzutc())))

    boundaries_utc = sorted(list(dict.fromkeys(boundaries_utc)))
    return boundaries_utc


def _five_minute_boundaries_utc_for_index(index: pd.DatetimeIndex,
                                          local_offset_minutes: int = LOCAL_OFFSET_MINUTES,
                                          tz_name: str = LOCAL_TZ_NAME) -> List[pd.Timestamp]:
    """
    Build 5m-quarter boundaries (90-minute steps) based on 15m quarters:
      For each 15m quarter (start -> next_start) generate:
        start + 0min, start + 90min, start + 180min, start + 270min, (the next_start is already included)
    Each returned timestamp is tz-aware UTC pandas.Timestamp.

    NOTE: although this function is named for 5m support, it returns 90-minute boundaries
    which represent the 5m-quarter divisions (each 90-minute block contains 18 x 5m candles).
    """
    if index is None or index.empty:
        return []

    # Ensure index is UTC-aware
    if index.tz is None:
        idx_utc = index.tz_localize("UTC")
    else:
        idx_utc = index.tz_convert("UTC")

    # get 15m quarter boundaries in UTC first
    quarter_boundaries = _quarter_boundaries_utc_for_index(index, local_offset_minutes=local_offset_minutes, tz_name=tz_name)
    if not quarter_boundaries:
        return []

    # convert to local tz for arithmetic
    try:
        local_tz = dateutil_tz.gettz(tz_name)
        local_quarters = [qb.tz_convert(tz_name) for qb in quarter_boundaries]
    except Exception:
        local_tz = dateutil_tz.tzoffset(None, local_offset_minutes * 60)
        local_quarters = []
        for qb in quarter_boundaries:
            try:
                local_quarters.append(pd.Timestamp(qb.tz_convert(local_tz)))
            except Exception:
                local_quarters.append(pd.Timestamp(qb).tz_convert(local_tz))

    five_min_boundaries_local = []

    # For each consecutive pair of 15m quarter boundaries, produce four 90-minute steps
    for i in range(len(local_quarters) - 1):
        start = local_quarters[i]
        end = local_quarters[i + 1]
        # duration of a quarter is expected to be 6 hours (360 minutes)
        # create boundaries at start + k*90 minutes for k=0..3
        for k in range(4):
            ts_local = start + pd.Timedelta(minutes=90 * k)
            # only include if within [start, end]
            if ts_local >= start and ts_local <= end:
                five_min_boundaries_local.append(ts_local)
        # we do NOT append end here because the next iteration's start equals end (except final)
    # append final quarter end (last element of local_quarters) so boundaries are closed
    five_min_boundaries_local.append(local_quarters[-1])

    # convert back to UTC and dedupe/sort
    boundaries_utc = []
    for ts in five_min_boundaries_local:
        try:
            boundaries_utc.append(ts.tz_convert("UTC"))
        except Exception:
            # fallback convert
            boundaries_utc.append(pd.Timestamp(ts).tz_localize(local_tz).tz_convert("UTC"))

    boundaries_utc = sorted(list(dict.fromkeys(boundaries_utc)))
    return boundaries_utc


def _hour_boundaries_utc_for_index(index: pd.DatetimeIndex,
                                   local_offset_minutes: int = LOCAL_OFFSET_MINUTES,
                                   tz_name: str = LOCAL_TZ_NAME) -> List[pd.Timestamp]:
    """
    Build hour boundaries in local tz (Asia/Tehran = UTC+3:30) for weekly quarters:
       MON = Q1, TUE = Q2, WED = Q3, THU = Q4
    Return tz-aware UTC pandas.Timestamp list.
    """
    if index is None or index.empty:
        return []

    if index.tz is None:
        idx_utc = index.tz_localize("UTC")
    else:
        idx_utc = index.tz_convert("UTC")

    try:
        idx_local = idx_utc.tz_convert(tz_name)
    except Exception:
        local_tz = dateutil_tz.tzoffset(None, local_offset_minutes * 60)
        idx_local = idx_utc.tz_convert(local_tz)

    # Expand range to ensure we capture all boundaries
    start_local = idx_local.min() - pd.Timedelta(days=7)
    end_local = idx_local.max() + pd.Timedelta(days=7)
    
    # Generate all dates in the range
    dates = pd.date_range(start=start_local.normalize(), end=end_local.normalize(), freq="D", tz=idx_local.tz)
    
    boundaries_utc = []
    # Only consider Monday through Thursday
    for date in dates:
        weekday = date.weekday()  # Monday=0, Sunday=6
        if weekday in [0, 1, 2, 3]:  # MON, TUE, WED, THU
            # Set time to 00:00 in local time
            try:
                local_boundary = date.normalize()
                boundaries_utc.append(local_boundary.tz_convert("UTC"))
            except Exception:
                local_dt = dt.datetime(year=date.year, month=date.month, day=date.day, tzinfo=date.tzinfo)
                boundaries_utc.append(pd.Timestamp(local_dt).tz_convert("UTC"))
    
    boundaries_utc = sorted(list(dict.fromkeys(boundaries_utc)))
    return boundaries_utc


def _day_boundaries_utc_for_index(index: pd.DatetimeIndex,
                                  local_offset_minutes: int = LOCAL_OFFSET_MINUTES,
                                  tz_name: str = LOCAL_TZ_NAME) -> List[pd.Timestamp]:
    """
    Build day boundaries in local tz (Asia/Tehran = UTC+3:30) for monthly quarters:
       WEEK1 = Q1, WEEK2 = Q2, WEEK3 = Q3, WEEK4 = Q4
    Return tz-aware UTC pandas.Timestamp list.
    """
    if index is None or index.empty:
        return []

    if index.tz is None:
        idx_utc = index.tz_localize("UTC")
    else:
        idx_utc = index.tz_convert("UTC")

    try:
        idx_local = idx_utc.tz_convert(tz_name)
    except Exception:
        local_tz = dateutil_tz.tzoffset(None, local_offset_minutes * 60)
        idx_local = idx_utc.tz_convert(local_tz)

    # Expand range to ensure we capture all boundaries
    start_local = idx_local.min() - pd.Timedelta(days=30)
    end_local = idx_local.max() + pd.Timedelta(days=30)
    
    # Generate week boundaries (start of each week, Monday)
    boundaries_utc = []
    current = start_local.normalize()
    
    while current <= end_local:
        # If this is a Monday, it's a week boundary
        if current.weekday() == 0:  # Monday
            try:
                local_boundary = current.normalize()
                boundaries_utc.append(local_boundary.tz_convert("UTC"))
            except Exception:
                local_dt = dt.datetime(year=current.year, month=current.month, day=current.day, tzinfo=current.tzinfo)
                boundaries_utc.append(pd.Timestamp(local_dt).tz_convert("UTC"))
        current += pd.Timedelta(days=1)
    
    boundaries_utc = sorted(list(dict.fromkeys(boundaries_utc)))
    return boundaries_utc


def _get_main_ax_from_mpf_axes(axes):
    if isinstance(axes, (list, tuple)):
        return axes[0]
    if isinstance(axes, dict):
        return axes.get("main", next(iter(axes.values())))
    return axes


def _quarter_index_from_boundary(boundary_utc: pd.Timestamp, tz_name: str = LOCAL_TZ_NAME, timeframe: int = 15) -> int:
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
                return 3

        elif timeframe == 240:
            week_of_month = (local.day - 1) // 7 + 1
            week_of_month = min(week_of_month, 4)
            return week_of_month - 1

        else:
            return 0
    except Exception:
        return 0


def _set_xaxis_labels_in_local_tz(ax, df_index: pd.DatetimeIndex,
                                  local_tz_offset_minutes: int = LOCAL_OFFSET_MINUTES,
                                  fmt: str = "%b %d %H:%M", tz_name: str = LOCAL_TZ_NAME):
    """
    Robustly set x-axis labels converting ticks to local tz (Asia/Tehran).
    Priority:
      1. Interpret tick as matplotlib date number -> convert -> local tz
      2. Fallback: if tick looks like an integer index, map to df_index (candle position)
    """
    try:
        local_tz = dateutil_tz.gettz(tz_name) or dateutil_tz.tzoffset(None, local_tz_offset_minutes * 60)
    except Exception:
        local_tz = dateutil_tz.tzoffset(None, local_tz_offset_minutes * 60)

    xticks = ax.get_xticks()
    labels = []
    for xt in xticks:
        label_dt = None
        try:
            dt_from_num = mdates.num2date(xt)
            if dt_from_num.tzinfo is None:
                dt_from_num = dt_from_num.replace(tzinfo=dateutil_tz.tzutc())
            label_dt = dt_from_num.astimezone(local_tz)
        except Exception:
            label_dt = None

        if label_dt is None:
            try:
                pos = int(round(float(xt)))
                if 0 <= pos < len(df_index):
                    ts = df_index[pos]
                    if getattr(ts, "tzinfo", None) is None:
                        dt_aware = pd.Timestamp(ts).tz_localize("UTC")
                    else:
                        dt_aware = pd.Timestamp(ts)
                    label_dt = dt_aware.tz_convert(local_tz)
            except Exception:
                label_dt = None

        labels.append(label_dt.strftime(fmt) if label_dt is not None else "")

    ax.set_xticks(xticks)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)


async def generate_chart(symbols: List[str], timeframe: int = 15) -> List[io.BytesIO]:
    chart_buffers: List[io.BytesIO] = []

    # Set time range based on timeframe
    to_date = int(time.time())
    if timeframe == 5:
        from_date = to_date - 1 * 24 * 60 * 60  # last 24 hours
    elif timeframe == 15:
        from_date = to_date - 3 * 24 * 60 * 60  # last 3 days
    elif timeframe == 60:
        from_date = to_date - 7 * 24 * 60 * 60  # last week
    elif timeframe == 240:
        from_date = to_date - 80 * 24 * 60 * 60  # last 80 days
    else:
        from_date = to_date - 3 * 24 * 60 * 60  # default to 3 days

    # choose tolerance per timeframe: much smaller for 5m because boundaries are 90min apart
    if timeframe == 5:
        tol = pd.Timedelta(minutes=45)   # accept nearest candle up to 45 minutes away
    elif timeframe == 15:
        tol = pd.Timedelta(minutes=60)
    else:
        tol = pd.Timedelta(hours=NEAR_TOLERANCE_HOURS)

    for symbol in symbols:
        fig = None
        try:
            norm_symbol = normalize_symbol(symbol)
            df: pd.DataFrame = get_ohlc(norm_symbol, timeframe=timeframe, from_date=from_date, to_date=to_date)

            if df is None or df.empty:
                logger.warning("[ChartService] No data for %s", norm_symbol)
                continue
            if "datetime" not in df.columns:
                logger.error("[ChartService] 'datetime' column missing for %s", norm_symbol)
                continue

            try:
                df["datetime"] = _parse_datetimes_to_utc(df["datetime"], source_tz=SOURCE_TZ)
            except Exception as e:
                logger.exception("[ChartService] Failed to parse datetimes for %s: %s", norm_symbol, e)
                continue

            df = df.dropna(subset=["datetime"])
            if df.empty:
                logger.warning("[ChartService] No valid timestamps for %s after parsing", norm_symbol)
                continue

            df.set_index("datetime", inplace=True)
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
            else:
                df.index = df.index.tz_convert("UTC")

            # Compute boundaries based on timeframe
            if timeframe == 5:
                boundaries = _five_minute_boundaries_utc_for_index(df.index, local_offset_minutes=LOCAL_OFFSET_MINUTES, tz_name=LOCAL_TZ_NAME)
            elif timeframe == 15:
                boundaries = _quarter_boundaries_utc_for_index(df.index, local_offset_minutes=LOCAL_OFFSET_MINUTES, tz_name=LOCAL_TZ_NAME)
            elif timeframe == 60:
                boundaries = _hour_boundaries_utc_for_index(df.index, local_offset_minutes=LOCAL_OFFSET_MINUTES, tz_name=LOCAL_TZ_NAME)
            elif timeframe == 240:
                boundaries = _day_boundaries_utc_for_index(df.index, local_offset_minutes=LOCAL_OFFSET_MINUTES, tz_name=LOCAL_TZ_NAME)
            else:
                boundaries = _quarter_boundaries_utc_for_index(df.index, local_offset_minutes=LOCAL_OFFSET_MINUTES, tz_name=LOCAL_TZ_NAME)

            filtered_boundaries = [b for b in boundaries if (b >= (df.index.min() - tol) and b <= (df.index.max() + tol))]

            fig, axes = mpf.plot(df, type="candle", style="yahoo", ylabel="Price", figsize=(18, 9), volume=False, returnfig=True)
            main_ax = _get_main_ax_from_mpf_axes(axes)

            rects = []
            for child in main_ax.get_children():
                if isinstance(child, mpatches.Rectangle):
                    try:
                        bbox = child.get_bbox()
                        if bbox.width > 0 and abs(bbox.height) > 0 and bbox.width < 1000:
                            rects.append(child)
                    except Exception:
                        continue
            used_x_centers: Optional[List[float]] = None
            if rects:
                rects_sorted = sorted(rects, key=lambda r: r.get_x())
                used_x_centers = [r.get_x() + r.get_width() / 2.0 for r in rects_sorted]

            x_coords_for_boundaries: List[float] = []
            good_boundaries: List[pd.Timestamp] = []
            for b in filtered_boundaries:
                try:
                    pos = df.index.get_indexer([pd.Timestamp(b)], method="nearest")[0]
                    if pos == -1:
                        logger.debug("[ChartService] boundary %s has no nearby candle (pos=-1), skipped", b)
                        continue
                    nearest_ts = df.index[pos]
                    delta = abs((nearest_ts - b).total_seconds())
                    if delta > tol.total_seconds():
                        logger.info("[ChartService] Skipping boundary %s: nearest candle at %s is %.1f seconds away (> tol %s)",
                                    b.isoformat(), nearest_ts.isoformat(), delta, tol)
                        continue
                    if used_x_centers and pos < len(used_x_centers):
                        x_coords_for_boundaries.append(used_x_centers[pos])
                    else:
                        x_coords_for_boundaries.append(float(pos))
                    good_boundaries.append(b)
                    logger.debug("[ChartService] Accepted boundary %s -> nearest candle %s (pos=%d, dist=%.2f s)",
                                 b.isoformat(), nearest_ts.isoformat(), pos, delta)
                except Exception as ex:
                    logger.debug("Failed to map boundary %r: %s", b, ex)

            for idx_b, b in enumerate(good_boundaries):
                try:
                    xcoord = x_coords_for_boundaries[idx_b]
                    main_ax.axvline(x=xcoord, linestyle="--", linewidth=0.8, alpha=0.8, color="C1")
                except Exception as ex:
                    logger.debug("Failed to draw vertical for %r: %s", b, ex)

            def _col_choice(df_obj, *candidates):
                for c in candidates:
                    if c in df_obj.columns:
                        return c
                for c in candidates:
                    lc = c.lower()
                    if lc in df_obj.columns:
                        return lc
                return None

            high_col = _col_choice(df, "High", "high")
            low_col = _col_choice(df, "Low", "low")
            open_col = _col_choice(df, "Open", "open")

            interval_highs = []
            interval_lows = []
            interval_high_ts = []
            interval_low_ts = []
            interval_open_ts = []

            for i in range(max(0, len(good_boundaries) - 1)):
                start_b = good_boundaries[i]
                end_b = good_boundaries[i + 1]
                try:
                    mask = (df.index >= pd.Timestamp(start_b)) & (df.index < pd.Timestamp(end_b))
                    segment = df.loc[mask]
                    if segment.empty:
                        interval_highs.append(None)
                        interval_lows.append(None)
                        interval_high_ts.append(None)
                        interval_low_ts.append(None)
                        interval_open_ts.append(None)
                    else:
                        h = segment[high_col].max() if high_col else None
                        l = segment[low_col].min() if low_col else None
                        interval_highs.append(h if pd.notna(h) else None)
                        interval_lows.append(l if pd.notna(l) else None)
                        try:
                            high_idx = segment[segment[high_col] == h].index
                            interval_high_ts.append(high_idx[0] if len(high_idx) else segment[high_col].idxmax())
                        except Exception:
                            try:
                                interval_high_ts.append(segment[high_col].idxmax())
                            except Exception:
                                interval_high_ts.append(None)
                        try:
                            low_idx = segment[segment[low_col] == l].index
                            interval_low_ts.append(low_idx[0] if len(low_idx) else segment[low_col].idxmin())
                        except Exception:
                            try:
                                interval_low_ts.append(segment[low_col].idxmin())
                            except Exception:
                                interval_low_ts.append(None)
                        try:
                            interval_open_ts.append(segment.index[0])
                        except Exception:
                            interval_open_ts.append(None)
                except Exception as ex:
                    logger.debug("Interval compute failed for %s: %s", norm_symbol, ex)
                    interval_highs.append(None)
                    interval_lows.append(None)
                    interval_high_ts.append(None)
                    interval_low_ts.append(None)
                    interval_open_ts.append(None)

            # Pre-compute quarter index for each good boundary based on timeframe
            boundary_quarter_idx = [_quarter_index_from_boundary(b, tz_name=LOCAL_TZ_NAME, timeframe=timeframe) for b in good_boundaries]

            for i in range(max(0, len(good_boundaries) - 2)):
                prev_h = interval_highs[i]
                prev_l = interval_lows[i]
                prev_h_ts = interval_high_ts[i]
                prev_l_ts = interval_low_ts[i]
                q_idx = boundary_quarter_idx[i] if i < len(boundary_quarter_idx) else (i % 4)
                q_color = QUARTER_COLORS[q_idx % 4]

                try:
                    xend = x_coords_for_boundaries[i + 2]
                except Exception:
                    try:
                        pos_end = df.index.get_indexer([pd.Timestamp(good_boundaries[i + 2])], method="nearest")[0]
                        xend = (used_x_centers[pos_end] if (used_x_centers and pos_end < len(used_x_centers)) else float(pos_end))
                    except Exception:
                        xend = None

                def _ts_to_x(ts_val):
                    if ts_val is None:
                        return None
                    try:
                        pos = df.index.get_indexer([pd.Timestamp(ts_val)], method="nearest")[0]
                        if pos == -1:
                            return None
                        return (used_x_centers[pos] if (used_x_centers and pos < len(used_x_centers)) else float(pos))
                    except Exception:
                        return None

                xstart_h = _ts_to_x(prev_h_ts)
                xstart_l = _ts_to_x(prev_l_ts)

                if xstart_h is None and prev_h is not None:
                    try:
                        pos_left = df.index.get_indexer([pd.Timestamp(good_boundaries[i])], method="nearest")[0]
                        xstart_h = (used_x_centers[pos_left] if (used_x_centers and pos_left < len(used_x_centers)) else float(pos_left))
                    except Exception:
                        xstart_h = None
                if xstart_l is None and prev_l is not None:
                    try:
                        pos_left = df.index.get_indexer([pd.Timestamp(good_boundaries[i])], method="nearest")[0]
                        xstart_l = (used_x_centers[pos_left] if (used_x_centers and pos_left < len(used_x_centers)) else float(pos_left))
                    except Exception:
                        xstart_l = None

                try:
                    if prev_h is not None and xstart_h is not None and xend is not None:
                        main_ax.hlines(y=float(prev_h), xmin=xstart_h, xmax=xend, linewidth=0.9, colors=q_color, linestyle="-", alpha=0.9, zorder=2)
                    if prev_l is not None and xstart_l is not None and xend is not None:
                        main_ax.hlines(y=float(prev_l), xmin=xstart_l, xmax=xend, linewidth=0.9, colors=q_color, linestyle="-", alpha=0.9, zorder=2)
                except Exception as ex:
                    logger.debug("Failed to draw prev-quarter HL for %s interval %d: %s", norm_symbol, i, ex)

            if open_col is None:
                logger.debug("[ChartService] Open column not found for %s; skipping true-open lines", norm_symbol)
            else:
                n_bounds = len(good_boundaries)
                # Modified logic: draw trueopen line even if quarters are incomplete
                # Find the last complete 4-quarter sequence or use the last available quarters
                if n_bounds >= 2:  # Need at least 2 boundaries to draw any line
                    # Determine how many complete 4-quarter sequences we have
                    complete_sequences = max(0, n_bounds - 4)
                    
                    # If we have at least one complete sequence, draw all of them
                    if complete_sequences > 0:
                        for start in range(0, complete_sequences + 1):
                            window_q = boundary_quarter_idx[start:start + 4]
                            if len(window_q) < 4:
                                continue
                            if window_q == [0, 1, 2, 3]:
                                open_ts = interval_open_ts[start + 1]
                                open_price = None
                                xstart = None
                                if open_ts is not None:
                                    try:
                                        pos_open = df.index.get_indexer([pd.Timestamp(open_ts)], method="nearest")[0]
                                        if pos_open != -1 and pos_open < len(df):
                                            open_price = df.iloc[pos_open][open_col]
                                            xstart = (used_x_centers[pos_open] if (used_x_centers and pos_open < len(used_x_centers)) else float(pos_open))
                                    except Exception:
                                        open_price = None
                                        xstart = None

                                if open_price is None or xstart is None:
                                    try:
                                        pos_left = df.index.get_indexer([pd.Timestamp(good_boundaries[start + 1])], method="nearest")[0]
                                        if pos_left != -1 and pos_left < len(df):
                                            open_price = df.iloc[pos_left][open_col]
                                            xstart = (used_x_centers[pos_left] if (used_x_centers and pos_left < len(used_x_centers)) else float(pos_left))
                                    except Exception:
                                        continue

                                try:
                                    xend = x_coords_for_boundaries[start + 4]
                                except Exception:
                                    try:
                                        pos_end = df.index.get_indexer([pd.Timestamp(good_boundaries[start + 4])], method="nearest")[0]
                                        xend = (used_x_centers[pos_end] if (used_x_centers and pos_end < len(used_x_centers)) else float(pos_end))
                                    except Exception:
                                        xend = None

                                if xend is None or xstart is None or open_price is None:
                                    continue

                                try:
                                    main_ax.hlines(y=float(open_price), xmin=xstart, xmax=xend, linewidth=1.0, colors=TRUEOPEN_COLOR, linestyle=TRUEOPEN_STYLE, alpha=0.95, zorder=2)
                                except Exception as ex:
                                    logger.debug("Failed to draw true-open for %s window starting at %d: %s", norm_symbol, start, ex)
                    
                    # Even if we don't have complete sequences, extend the last trueopen line to last-1 candle
                    # Find the most recent valid trueopen line
                    if n_bounds >= 5:  # Standard case with complete sequences
                        pass  # Already handled above
                    else:  # Incomplete case - extend the last available trueopen
                        # Find the last possible trueopen start (need at least 2 boundaries after start)
                        for start in range(max(0, n_bounds - 5), -1, -1):
                            if start + 2 < n_bounds:  # Need at least start+2 to have a meaningful line
                                # Get the open price from the start+1 boundary
                                open_ts = interval_open_ts[start + 1]
                                open_price = None
                                xstart = None
                                
                                if open_ts is not None:
                                    try:
                                        pos_open = df.index.get_indexer([pd.Timestamp(open_ts)], method="nearest")[0]
                                        if pos_open != -1 and pos_open < len(df):
                                            open_price = df.iloc[pos_open][open_col]
                                            xstart = (used_x_centers[pos_open] if (used_x_centers and pos_open < len(used_x_centers)) else float(pos_open))
                                    except Exception:
                                        open_price = None
                                        xstart = None

                                if open_price is None or xstart is None:
                                    try:
                                        pos_left = df.index.get_indexer([pd.Timestamp(good_boundaries[start + 1])], method="nearest")[0]
                                        if pos_left != -1 and pos_left < len(df):
                                            open_price = df.iloc[pos_left][open_col]
                                            xstart = (used_x_centers[pos_left] if (used_x_centers and pos_left < len(used_x_centers)) else float(pos_left))
                                    except Exception:
                                        continue

                                # Extend to the last - 1 candle
                                if len(df) >= 2 and xstart is not None and open_price is not None:
                                    # Use the x-coordinate of the second-to-last candle
                                    last_minus_one_pos = len(df) - 2
                                    xend = (used_x_centers[last_minus_one_pos] if (used_x_centers and last_minus_one_pos < len(used_x_centers)) else float(last_minus_one_pos))
                                    
                                    try:
                                        main_ax.hlines(y=float(open_price), xmin=xstart, xmax=xend, linewidth=1.0, colors=TRUEOPEN_COLOR, linestyle=TRUEOPEN_STYLE, alpha=0.95, zorder=2)
                                        logger.debug("Extended trueopen line for %s from boundary %d to last-1 candle", norm_symbol, start)
                                        break  # Only draw the most recent one
                                    except Exception as ex:
                                        logger.debug("Failed to draw extended true-open for %s: %s", norm_symbol, ex)
                                        break

            if x_coords_for_boundaries and len(x_coords_for_boundaries) >= 2:
                try:
                    y_min, y_max = main_ax.get_ylim()
                    y_for_label = y_max - (y_max - y_min) * 0.03
                except Exception:
                    y_for_label = None

                quarter_names = ["Q1", "Q2", "Q3", "Q4"]
                for i in range(len(x_coords_for_boundaries) - 1):
                    x_left = x_coords_for_boundaries[i]
                    x_right = x_coords_for_boundaries[i + 1]
                    x_label = (x_left + x_right) / 2.0
                    q_idx = boundary_quarter_idx[i] if i < len(boundary_quarter_idx) else (i % 4)
                    qname = quarter_names[q_idx % 4]
                    try:
                        if y_for_label is None:
                            main_ax.text(x_label, 0.98, qname, transform=main_ax.get_xaxis_transform(), ha="center", va="top", fontsize=9, bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.6, edgecolor="none"))
                        else:
                            main_ax.text(x_label, y_for_label, qname, ha="center", va="top", fontsize=9, bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.6, edgecolor="none"))
                    except Exception as ex:
                        logger.debug("Failed to draw quarter label %s at x=%r: %s", qname, x_label, ex)

            try:
                _set_xaxis_labels_in_local_tz(main_ax, df.index, local_tz_offset_minutes=LOCAL_OFFSET_MINUTES, fmt="%b %d %H:%M", tz_name=LOCAL_TZ_NAME)
            except Exception as ex:
                logger.debug("Failed to set x-axis labels in local tz for %s: %s", norm_symbol, ex)

            if fig is not None:
                fig.suptitle(f"{norm_symbol} - {timeframe}min", fontsize=12, color="gray", alpha=0.6)
                # Use subplots_adjust instead of tight_layout to avoid warnings
                try:
                    fig.subplots_adjust(top=0.94, bottom=0.1, left=0.05, right=0.95, hspace=0.2, wspace=0.2)
                except Exception:
                    pass
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
                buf.seek(0)
                chart_buffers.append(buf)

        except Exception as e:
            logger.exception("Failed to generate chart for %s: %s", symbol, e)

        finally:
            with suppress(Exception):
                if fig is not None:
                    plt.close(fig)

    return chart_buffers