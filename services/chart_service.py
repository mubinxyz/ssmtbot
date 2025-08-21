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

# Pillow for image stacking
from PIL import Image

logger = logging.getLogger("services.chart_service")
logging.basicConfig(level=logging.INFO)

# ---------- CONFIG ----------
SOURCE_TZ = dateutil_tz.tzoffset(None, 3 * 3600)   # provider = UTC+3
LOCAL_TZ_NAME = "Asia/Tehran"
LOCAL_OFFSET_MINUTES = 210                 # fallback +3:30
NEAR_TOLERANCE_HOURS = 3                   # fallback tolerance used to decide "nearby" candle
QUARTER_COLORS = ["C2", "C3", "C4", "C5"]
TRUEOPEN_COLOR = "C6"
TRUEOPEN_STYLE = "--"
# ---------------------------


def _parse_datetimes_to_utc(series: pd.Series, source_tz=SOURCE_TZ) -> pd.Series:
    if pd.api.types.is_integer_dtype(series) or pd.api.types.is_float_dtype(series):
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
    if index is None or index.empty:
        return []

    if index.tz is None:
        idx_utc = index.tz_localize("UTC")
    else:
        idx_utc = index.tz_convert("UTC")

    quarter_boundaries = _quarter_boundaries_utc_for_index(index, local_offset_minutes=local_offset_minutes, tz_name=tz_name)
    if not quarter_boundaries:
        return []

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

    for i in range(len(local_quarters) - 1):
        start = local_quarters[i]
        end = local_quarters[i + 1]
        for k in range(4):
            ts_local = start + pd.Timedelta(minutes=90 * k)
            if ts_local >= start and ts_local <= end:
                five_min_boundaries_local.append(ts_local)
    five_min_boundaries_local.append(local_quarters[-1])

    boundaries_utc = []
    for ts in five_min_boundaries_local:
        try:
            boundaries_utc.append(ts.tz_convert("UTC"))
        except Exception:
            boundaries_utc.append(pd.Timestamp(ts).tz_localize(local_tz).tz_convert("UTC"))

    boundaries_utc = sorted(list(dict.fromkeys(boundaries_utc)))
    return boundaries_utc


def _hour_boundaries_utc_for_index(index: pd.DatetimeIndex,
                                   local_offset_minutes: int = LOCAL_OFFSET_MINUTES,
                                   tz_name: str = LOCAL_TZ_NAME) -> List[pd.Timestamp]:
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

    start_local = idx_local.min() - pd.Timedelta(days=7)
    end_local = idx_local.max() + pd.Timedelta(days=7)
    dates = pd.date_range(start=start_local.normalize(), end=end_local.normalize(), freq="D", tz=idx_local.tz)
    boundaries_utc = []
    for date in dates:
        weekday = date.weekday()
        if weekday in [0, 1, 2, 3]:
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

    start_local = idx_local.min() - pd.Timedelta(days=30)
    end_local = idx_local.max() + pd.Timedelta(days=30)
    boundaries_utc = []
    current = start_local.normalize()
    while current <= end_local:
        if current.weekday() == 0:
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
    if boundary_utc is None:
        return 0

    try:
        if getattr(boundary_utc, "tzinfo", None) is None:
            b_utc = pd.Timestamp(boundary_utc).tz_localize("UTC")
        else:
            b_utc = pd.Timestamp(boundary_utc).tz_convert("UTC")
        local = b_utc.tz_convert(tz_name)

        if timeframe == 5:
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
            if weekday == 0:
                return 0
            elif weekday == 1:
                return 1
            elif weekday == 2:
                return 2
            elif weekday == 3:
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

    to_date = int(time.time())
    if timeframe == 5:
        from_date = to_date - 0.3 * 24 * 60 * 60
    elif timeframe == 15:
        from_date = to_date - 1.25 * 24 * 60 * 60
    elif timeframe == 60:
        from_date = to_date - 4.5 * 24 * 60 * 60
    elif timeframe == 240:
        from_date = to_date - 30 * 24 * 60 * 60
    else:
        from_date = to_date - 3 * 24 * 60 * 60

    if timeframe == 5:
        tol = pd.Timedelta(minutes=45)
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

            # Draw vertical quarter dividers (more obvious now)
            for idx_b, b in enumerate(good_boundaries):
                try:
                    xcoord = x_coords_for_boundaries[idx_b]
                    # make vertical divider more noticeable
                    main_ax.axvline(x=xcoord, linestyle="--", linewidth=1.2, alpha=0.95, color="C1", zorder=1)
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

            boundary_quarter_idx = [_quarter_index_from_boundary(b, tz_name=LOCAL_TZ_NAME, timeframe=timeframe) for b in good_boundaries]

            # Determine last-1 coordinate (used for true-open fallback)
            last_minus_one_pos = (len(df) - 2) if len(df) >= 2 else None
            last_minus_one_x = None
            if last_minus_one_pos is not None:
                if used_x_centers and last_minus_one_pos < len(used_x_centers):
                    last_minus_one_x = used_x_centers[last_minus_one_pos]
                else:
                    last_minus_one_x = float(last_minus_one_pos)

            # NEW: extend previous HLs to last candle + 10 (user requested +10)
            last_pos = (len(df) - 1) if len(df) >= 1 else None
            last_plus_ten_x = None
            if last_pos is not None:
                if used_x_centers and last_pos < len(used_x_centers):
                    last_plus_ten_x = used_x_centers[last_pos] + 10.0
                else:
                    last_plus_ten_x = float(last_pos) + 10.0

            # compute hl padding (so HLs don't exactly touch candle centers)
            try:
                if used_x_centers and len(used_x_centers) >= 2:
                    candle_spacing = used_x_centers[1] - used_x_centers[0]
                else:
                    candle_spacing = 1.0
                hl_pad = max(candle_spacing * 0.12, 0.2)
            except Exception:
                hl_pad = 0.3

            interval_count = len(interval_highs)

            for j in range(interval_count):
                prev_h = interval_highs[j]
                prev_l = interval_lows[j]
                prev_h_ts = interval_high_ts[j]
                prev_l_ts = interval_low_ts[j]
                q_idx = boundary_quarter_idx[j] if j < len(boundary_quarter_idx) else (j % 4)
                q_color = QUARTER_COLORS[q_idx % 4]

                xend = None
                try:
                    if (j + 2) < len(x_coords_for_boundaries):
                        xend = x_coords_for_boundaries[j + 2]
                    else:
                        try:
                            pos_end = df.index.get_indexer([pd.Timestamp(good_boundaries[j + 2])], method="nearest")[0]
                            if pos_end != -1:
                                xend = (used_x_centers[pos_end] if (used_x_centers and pos_end < len(used_x_centers)) else float(pos_end))
                        except Exception:
                            xend = None
                except Exception:
                    xend = None

                # fallback for HLs -> last candle + 10
                if xend is None:
                    xend = last_plus_ten_x

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
                        pos_left = df.index.get_indexer([pd.Timestamp(good_boundaries[j])], method="nearest")[0]
                        xstart_h = (used_x_centers[pos_left] if (used_x_centers and pos_left < len(used_x_centers)) else float(pos_left))
                    except Exception:
                        xstart_h = None
                if xstart_l is None and prev_l is not None:
                    try:
                        pos_left = df.index.get_indexer([pd.Timestamp(good_boundaries[j])], method="nearest")[0]
                        xstart_l = (used_x_centers[pos_left] if (used_x_centers and pos_left < len(used_x_centers)) else float(pos_left))
                    except Exception:
                        xstart_l = None

                try:
                    if prev_h is not None and xstart_h is not None and xend is not None:
                        xmin = xstart_h + hl_pad
                        xmax = xend - hl_pad if xend is not None else xend
                        if xmax is None or xmax <= xmin:
                            xmax = xend
                        main_ax.hlines(y=float(prev_h), xmin=xmin, xmax=xmax, linewidth=1.0, colors=q_color, linestyle="-", alpha=0.95, zorder=3)
                    if prev_l is not None and xstart_l is not None and xend is not None:
                        xmin = xstart_l + hl_pad
                        xmax = xend - hl_pad if xend is not None else xend
                        if xmax is None or xmax <= xmin:
                            xmax = xend
                        main_ax.hlines(y=float(prev_l), xmin=xmin, xmax=xmax, linewidth=1.0, colors=q_color, linestyle="-", alpha=0.95, zorder=3)
                except Exception as ex:
                    logger.debug("Failed to draw prev-quarter HL for %s interval %d: %s", norm_symbol, j, ex)

            # Draw true-open lines (unchanged behavior, fallback to last-1)
            if open_col is None:
                logger.debug("[ChartService] Open column not found for %s; skipping true-open lines", norm_symbol)
            else:
                n_bounds = len(good_boundaries)
                try:
                    for i in range(0, max(0, n_bounds - 1)):
                        try:
                            if i >= len(boundary_quarter_idx):
                                continue
                            if boundary_quarter_idx[i] != 1:
                                continue

                            open_ts = interval_open_ts[i] if i < len(interval_open_ts) else None
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

                            if (open_price is None or xstart is None) and i < len(good_boundaries):
                                try:
                                    pos_left = df.index.get_indexer([pd.Timestamp(good_boundaries[i])], method="nearest")[0]
                                    if pos_left != -1 and pos_left < len(df):
                                        open_price = df.iloc[pos_left][open_col]
                                        xstart = (used_x_centers[pos_left] if (used_x_centers and pos_left < len(used_x_centers)) else float(pos_left))
                                except Exception:
                                    pass

                            if open_price is None or xstart is None:
                                continue

                            xend = None
                            end_boundary_idx = i + 3
                            if end_boundary_idx < len(x_coords_for_boundaries):
                                xend = x_coords_for_boundaries[end_boundary_idx]
                            else:
                                if end_boundary_idx < len(good_boundaries):
                                    try:
                                        pos_end = df.index.get_indexer([pd.Timestamp(good_boundaries[end_boundary_idx])], method="nearest")[0]
                                        if pos_end != -1:
                                            xend = (used_x_centers[pos_end] if (used_x_centers and pos_end < len(used_x_centers)) else float(pos_end))
                                    except Exception:
                                        xend = None

                            if xend is None:
                                xend = last_minus_one_x

                            if xstart is not None and xend is not None:
                                try:
                                    main_ax.hlines(y=float(open_price), xmin=xstart, xmax=xend, linewidth=1.0,
                                                   colors=TRUEOPEN_COLOR, linestyle=TRUEOPEN_STYLE, alpha=0.95, zorder=2)
                                except Exception as ex:
                                    logger.debug("Failed to draw true-open for %s at interval %d: %s", norm_symbol, i, ex)
                        except Exception as ex:
                            logger.debug("True-open loop iteration failed for %s interval %d: %s", norm_symbol, i, ex)
                except Exception as ex:
                    logger.debug("True-open drawing logic failed for %s: %s", norm_symbol, ex)

                # extend most recent incomplete true-open only if it's Q2
                try:
                    if len(interval_open_ts) > 0 and last_minus_one_x is not None:
                        candidate_interval_idx = len(interval_open_ts) - 1
                        q_candidate = None
                        if candidate_interval_idx < len(boundary_quarter_idx):
                            q_candidate = boundary_quarter_idx[candidate_interval_idx]
                        else:
                            try:
                                candidate_ts = interval_open_ts[candidate_interval_idx]
                                if candidate_ts is not None:
                                    q_candidate = _quarter_index_from_boundary(candidate_ts, tz_name=LOCAL_TZ_NAME, timeframe=timeframe)
                            except Exception:
                                q_candidate = None

                        if q_candidate == 1:
                            open_ts = interval_open_ts[candidate_interval_idx]
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

                            if (open_price is None or xstart is None) and candidate_interval_idx < len(good_boundaries):
                                try:
                                    pos_left = df.index.get_indexer([pd.Timestamp(good_boundaries[candidate_interval_idx])], method="nearest")[0]
                                    if pos_left != -1 and pos_left < len(df):
                                        open_price = df.iloc[pos_left][open_col]
                                        xstart = (used_x_centers[pos_left] if (used_x_centers and pos_left < len(used_x_centers)) else float(pos_left))
                                except Exception:
                                    pass

                            if open_price is not None and xstart is not None:
                                try:
                                    main_ax.hlines(y=float(open_price), xmin=xstart, xmax=last_minus_one_x, linewidth=1.0,
                                                   colors=TRUEOPEN_COLOR, linestyle=TRUEOPEN_STYLE, alpha=0.95, zorder=2)
                                    logger.debug("Extended trueopen line for %s from candidate interval %d to last-1 candle", norm_symbol, candidate_interval_idx)
                                except Exception as ex:
                                    logger.debug("Failed to draw extended true-open for %s: %s", norm_symbol, ex)
                except Exception as ex:
                    logger.debug("True-open extension logic failed for %s: %s", norm_symbol, ex)

            # Presentation tweaks: vertical padding (reduced), grid, annotate recent HLs, quarter labels, legend
            try:
                y_min, y_max = main_ax.get_ylim()
                # reduce padding from 6% to 3% to avoid too much top margin
                margin = (y_max - y_min) * 0.03
                main_ax.set_ylim(y_min - margin, y_max + margin)
            except Exception:
                pass

            main_ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.30)

            # annotate most recent interval HLs (if any)
            try:
                if interval_count > 0:
                    k = interval_count - 1
                    label_x = last_plus_ten_x if last_plus_ten_x is not None else (len(df) - 1)
                    if interval_highs[k] is not None:
                        main_ax.text(label_x + max(0.6, hl_pad * 2), float(interval_highs[k]),
                                     f"{float(interval_highs[k]):.4f}",
                                     va="center", fontsize=9,
                                     bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, edgecolor="none"))
                    if interval_lows[k] is not None:
                        main_ax.text(label_x + max(0.6, hl_pad * 2), float(interval_lows[k]),
                                     f"{float(interval_lows[k]):.4f}",
                                     va="center", fontsize=9,
                                     bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, edgecolor="none"))
            except Exception:
                pass

            # draw quarter labels with boxed background and stronger visibility
            try:
                if x_coords_for_boundaries and len(x_coords_for_boundaries) >= 2:
                    quarter_names = ["Q1", "Q2", "Q3", "Q4"]
                    for i in range(len(x_coords_for_boundaries) - 1):
                        x_left = x_coords_for_boundaries[i]
                        x_right = x_coords_for_boundaries[i + 1]
                        x_label = (x_left + x_right) / 2.0
                        q_idx = boundary_quarter_idx[i] if i < len(boundary_quarter_idx) else (i % 4)
                        qname = quarter_names[q_idx % 4]
                        box_edge = QUARTER_COLORS[q_idx % 4]
                        try:
                            y_for_label = None
                            try:
                                y_min2, y_max2 = main_ax.get_ylim()
                                y_for_label = y_max2 - (y_max2 - y_min2) * 0.03
                            except Exception:
                                y_for_label = None
                            if y_for_label is None:
                                main_ax.text(x_label, 0.98, qname, transform=main_ax.get_xaxis_transform(),
                                             ha="center", va="top", fontsize=10,
                                             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor=box_edge))
                            else:
                                main_ax.text(x_label, y_for_label, qname, ha="center", va="top", fontsize=10,
                                             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor=box_edge))
                        except Exception as ex:
                            logger.debug("Failed to draw quarter label %s at x=%r: %s", qname, x_label, ex)
            except Exception:
                pass

            # small legend
            try:
                legend_items = [
                    mpatches.Patch(color=QUARTER_COLORS[0], label="Quarter HL (Q1)"),
                    mpatches.Patch(color=QUARTER_COLORS[1], label="Quarter HL (Q2)"),
                    mpatches.Patch(color=TRUEOPEN_COLOR, label="True Open")
                ]
                main_ax.legend(handles=legend_items, loc="upper left", fontsize=9, framealpha=0.85)
            except Exception:
                pass

            # set title centered over the plotting axes (so it visually centers above the candles)
            try:
                main_ax.set_title(f"{norm_symbol} - {timeframe}min", fontsize=12, color="gray", pad=14)
            except Exception:
                try:
                    # fallback to figure-level if axes-level fails
                    fig.suptitle(f"{norm_symbol} - {timeframe}min", fontsize=12, color="gray", alpha=0.6)
                except Exception:
                    pass

            if fig is not None:
                # tighten subplot top so title is closer to plot and reduce empty space
                try:
                    fig.subplots_adjust(top=0.95, bottom=0.08, left=0.05, right=0.95, hspace=0.2, wspace=0.2)
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

    # Stack generated images vertically and append combined image as last buffer
    try:
        if chart_buffers and len(chart_buffers) > 0:
            pil_imgs = []
            for b in chart_buffers:
                try:
                    b.seek(0)
                    pil_imgs.append(Image.open(io.BytesIO(b.getvalue())).convert("RGBA"))
                except Exception:
                    continue

            if pil_imgs:
                pad = 8  # pixels between images
                max_w = max(img.width for img in pil_imgs)
                total_h = sum(img.height for img in pil_imgs) + pad * (len(pil_imgs) - 1)

                stacked = Image.new("RGBA", (max_w, total_h), (255, 255, 255, 255))
                y = 0
                for img in pil_imgs:
                    x = (max_w - img.width) // 2
                    stacked.paste(img, (x, y), img if img.mode == "RGBA" else None)
                    y += img.height + pad

                combined_buf = io.BytesIO()
                stacked.save(combined_buf, format="PNG")
                combined_buf.seek(0)
                chart_buffers.append(combined_buf)
    except Exception as ex:
        logger.debug("Failed to create stacked combined image: %s", ex)

    return chart_buffers
