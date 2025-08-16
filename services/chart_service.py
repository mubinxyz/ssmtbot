# services/chart_service.py

import io
import time
import logging
from contextlib import suppress
from typing import List, Set
import datetime as dt

import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
from dateutil import tz

from utils.get_data import get_ohlc
from utils.normalize_data import normalize_symbol


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _to_utc_naive_datetime(value) -> dt.datetime:
    """
    Convert many timestamp representations to a UTC-naive Python datetime.
    If naive, assume UTC.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        raise ValueError("Invalid timestamp value")

    if isinstance(value, dt.datetime):
        d = value
        if d.tzinfo is None:
            d = d.replace(tzinfo=tz.tzutc())
        return d.astimezone(tz.tzutc()).replace(tzinfo=None)

    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        raise ValueError(f"Cannot convert to timestamp: {value!r}")

    if ts.tzinfo is None:
        py = ts.to_pydatetime().replace(tzinfo=tz.tzutc())
    else:
        py = ts.to_pydatetime()
    return py.astimezone(tz.tzutc()).replace(tzinfo=None)


def _quarter_boundaries_utc_naive_for_index(index: pd.DatetimeIndex,
                                           local_offset_minutes: int = 210) -> List[dt.datetime]:
    """
    For each date in the index range produce quarter boundary datetimes in local timezone (UTC+3:30),
    convert them to UTC, and return UTC-naive python datetimes suitable for mapping to DataFrame rows.
    Boundaries per day: 01:30, 07:30, 13:30, 19:30 and next-day 01:30.
    """
    if index.empty:
        return []

    local_tz = tz.tzoffset(None, local_offset_minutes * 60)
    utc_tz = tz.tzutc()

    start_date = index.min().normalize()
    end_date = index.max().normalize()
    days = pd.date_range(start=start_date, end=end_date, freq="D")

    boundaries: Set[dt.datetime] = set()
    local_times = [(1, 30), (7, 30), (13, 30), (19, 30)]

    for day in days:
        for hh, mm in local_times:
            local_dt = dt.datetime(year=day.year, month=day.month, day=day.day, hour=hh, minute=mm, tzinfo=local_tz)
            utc_dt = local_dt.astimezone(utc_tz)
            boundaries.add(utc_dt.replace(tzinfo=None))

        nd = day + pd.Timedelta(days=1)
        local_dt_next = dt.datetime(year=nd.year, month=nd.month, day=nd.day, hour=1, minute=30, tzinfo=local_tz)
        utc_next = local_dt_next.astimezone(utc_tz)
        boundaries.add(utc_next.replace(tzinfo=None))

    return sorted(boundaries)


def _get_main_ax_from_mpf_axes(axes):
    """
    mplfinance returns axes either as list/tuple or dict-like. Return the main candlestick axis.
    """
    if isinstance(axes, (list, tuple)):
        return axes[0]
    if isinstance(axes, dict):
        if "main" in axes:
            return axes["main"]
        return next(iter(axes.values()))
    return axes


def _set_xaxis_labels_in_local_tz(ax, df_index: pd.DatetimeIndex, local_tz_offset_minutes: int = 210, fmt: str = "%b %d %H:%M"):
    """
    Replace x-axis tick labels on `ax` to show datetime converted to UTC+3:30 (or any offset).
    Handles ticks that are row-index numeric positions (common mplfinance internal mapping)
    or matplotlib date numbers (if mplfinance used real dates).
    - df_index: the DataFrame index (UTC-naive datetimes assumed).
    - fmt: strftime format for labels.
    """
    local_tz = tz.tzoffset(None, local_tz_offset_minutes * 60)
    xticks = ax.get_xticks()
    labels = []

    for xt in xticks:
        label_dt = None
        # Try to interpret xt as row-number index (mplfinance common case)
        try:
            pos = int(round(float(xt)))
            if 0 <= pos < len(df_index):
                dt_utc = df_index[pos]
                if dt_utc.tzinfo is None:
                    dt_aware = dt_utc.replace(tzinfo=tz.tzutc())
                else:
                    dt_aware = dt_utc
                label_dt = dt_aware.astimezone(local_tz)
        except Exception:
            label_dt = None

        # If that failed, try to interpret xt as matplotlib datenums
        if label_dt is None:
            try:
                dt_from_num = mdates.num2date(xt)
                if dt_from_num.tzinfo is None:
                    dt_from_num = dt_from_num.replace(tzinfo=tz.tzutc())
                dt_local = dt_from_num.astimezone(local_tz)
                label_dt = dt_local
            except Exception:
                label_dt = None

        if label_dt is None:
            labels.append("")
        else:
            labels.append(label_dt.strftime(fmt))

    ax.set_xticks(xticks)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)


async def generate_chart(symbols: List[str], timeframe: int = 15) -> List[io.BytesIO]:
    """
    Generate candlestick charts for a list of symbols with quarter vertical lines
    (quarters defined as 01:30, 07:30, 13:30, 19:30 local -> UTC+3:30), display x-axis labels
    in UTC+3:30 and add quarter labels (Q1..Q4) between the vertical lines.
    """
    chart_buffers: List[io.BytesIO] = []
    to_date = int(time.time())
    from_date = to_date - 3 * 24 * 60 * 60  # last 3 days

    for symbol in symbols:
        fig = None
        try:
            norm_symbol = normalize_symbol(symbol)
            df: pd.DataFrame = get_ohlc(
                norm_symbol,
                timeframe=timeframe,
                from_date=from_date,
                to_date=to_date
            )

            if df is None or df.empty:
                logger.warning("[ChartService] No data for %s", norm_symbol)
                continue

            if "datetime" not in df.columns:
                logger.error("[ChartService] 'datetime' column missing for %s", norm_symbol)
                continue

            # Normalize datetime column to UTC-naive python datetimes
            if pd.api.types.is_integer_dtype(df["datetime"]):
                df["datetime"] = pd.to_datetime(df["datetime"], unit="s", utc=True).apply(
                    lambda ts: ts.to_pydatetime().astimezone(tz.tzutc()).replace(tzinfo=None))
            else:
                df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce").apply(_to_utc_naive_datetime)

            df = df.dropna(subset=["datetime"])
            if df.empty:
                logger.warning("[ChartService] All datetimes invalid after conversion for %s", norm_symbol)
                continue

            # set index as UTC-naive datetimes (mplfinance can accept naive datetimes)
            df.set_index("datetime", inplace=True)

            # compute quarter boundaries (UTC-naive python datetimes)
            boundaries = _quarter_boundaries_utc_naive_for_index(df.index, local_offset_minutes=210)

            # filter boundaries to within data range (+/- one timeframe tolerance)
            filtered_boundaries = [b for b in boundaries if (b >= (df.index.min() - pd.Timedelta(minutes=timeframe))
                                                              and b <= (df.index.max() + pd.Timedelta(minutes=timeframe)))]
            if not filtered_boundaries:
                logger.debug("[ChartService] No quarter boundaries in range for %s", norm_symbol)

            # Map boundaries to nearest row positions (numeric positions mplfinance often uses)
            v_indices: List[float] = []
            for b in filtered_boundaries:
                try:
                    pos = df.index.get_indexer([pd.Timestamp(b)], method="nearest")[0]
                    if pos == -1:
                        continue
                    v_indices.append(float(pos))
                except Exception as ex:
                    logger.debug("Failed to map boundary %r to row index: %s", b, ex)

            vlines_payload = dict(v=v_indices, linewidths=0.8, colors='C1', linestyle='--', alpha=0.8) if v_indices else None

            # Try mpf.plot with vlines first
            used_x_centers = None  # if we detect exact candle centers, populate this list
            try:
                fig, axes = mpf.plot(
                    df,
                    type="candle",
                    style="yahoo",
                    ylabel="Price",
                    figsize=(18, 9),
                    volume=False,
                    vlines=vlines_payload,
                    returnfig=True
                )
            except Exception as e_v:
                # Fallback — plot normally and draw lines at exact candle centers
                logger.info("[ChartService] vlines approach failed for %s: %s. Falling back to exact-candle-centers approach.", norm_symbol, e_v)
                fig, axes = mpf.plot(
                    df,
                    type="candle",
                    style="yahoo",
                    ylabel="Price",
                    figsize=(18, 9),
                    volume=False,
                    returnfig=True
                )

                main_ax = _get_main_ax_from_mpf_axes(axes)

                # Collect Rectangle artists that represent candle bodies.
                rects = []
                for child in main_ax.get_children():
                    if isinstance(child, mpatches.Rectangle):
                        try:
                            bbox = child.get_bbox()
                            w = float(bbox.width)
                            h = float(bbox.height)
                        except Exception:
                            continue
                        # heuristics to identify candle bodies vs large background rect
                        if w > 0 and abs(h) > 0 and bbox.width < 1000:
                            rects.append(child)

                x_centers = []
                if rects:
                    rects_sorted = sorted(rects, key=lambda r: r.get_x())
                    x_centers = [r.get_x() + r.get_width() / 2.0 for r in rects_sorted]
                    used_x_centers = x_centers  # store for label placement

                # Draw vertical lines: prefer exact candle center if available else numeric row pos
                for b in filtered_boundaries:
                    try:
                        pos = df.index.get_indexer([pd.Timestamp(b)], method="nearest")[0]
                        if pos == -1:
                            continue
                        if x_centers and pos < len(x_centers):
                            xcoord = x_centers[pos]
                            main_ax.axvline(x=xcoord, linestyle="--", linewidth=0.8, alpha=0.8)
                        else:
                            main_ax.axvline(x=float(pos), linestyle="--", linewidth=0.8, alpha=0.8)
                    except Exception as inner_ex:
                        logger.debug("Failed to draw exact-candle vline for %r: %s", b, inner_ex)

            # If mpf.plot succeeded with vlines, axes is available and main_ax can be extracted
            main_ax = _get_main_ax_from_mpf_axes(axes)

            # Build x coordinates for each filtered_boundary to place labels (use exact centers if detected)
            x_coords_for_boundaries: List[float] = []
            # If we have used_x_centers from fallback, use them; otherwise fallback to numeric positions
            for b in filtered_boundaries:
                try:
                    pos = df.index.get_indexer([pd.Timestamp(b)], method="nearest")[0]
                    if pos == -1:
                        continue
                    if used_x_centers and pos < len(used_x_centers):
                        x_coords_for_boundaries.append(used_x_centers[pos])
                    else:
                        # numeric row position
                        x_coords_for_boundaries.append(float(pos))
                except Exception as ex:
                    logger.debug("Failed to compute x coord for boundary %r: %s", b, ex)

            # Draw quarter labels centered between consecutive boundary x coordinates.
            # Labels repeat Q1..Q4 for each group of 5 boundaries (4 intervals).
            if x_coords_for_boundaries and len(x_coords_for_boundaries) >= 2:
                # compute axis y coordinate to place labels (near top)
                try:
                    y_min, y_max = main_ax.get_ylim()
                    y_for_label = y_max - (y_max - y_min) * 0.03  # 3% from top
                except Exception:
                    y_for_label = None

                quarter_names = ["Q1", "Q2", "Q3", "Q4"]
                label_idx = 0
                for i in range(len(x_coords_for_boundaries) - 1):
                    x_left = x_coords_for_boundaries[i]
                    x_right = x_coords_for_boundaries[i + 1]
                    x_label = (x_left + x_right) / 2.0

                    qname = quarter_names[label_idx % 4]
                    label_idx += 1

                    try:
                        if y_for_label is None:
                            # If we couldn't compute y, place at axis top in axis coordinates
                            main_ax.text(x_label, 0.98, qname,
                                         transform=main_ax.get_xaxis_transform(),
                                         ha="center", va="top", fontsize=9,
                                         bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.6, edgecolor="none"))
                        else:
                            main_ax.text(x_label, y_for_label, qname,
                                         ha="center", va="top", fontsize=9,
                                         bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.6, edgecolor="none"))
                    except Exception as ex:
                        logger.debug("Failed to draw quarter label %s at x=%r: %s", qname, x_label, ex)

            # Replace the x-axis labels with UTC+3:30 converted timestamps aligned to df rows
            try:
                _set_xaxis_labels_in_local_tz(main_ax, df.index, local_tz_offset_minutes=210, fmt="%b %d %H:%M")
            except Exception as ex:
                logger.debug("Failed to set x-axis labels in local tz for %s: %s", norm_symbol, ex)

            # Custom title styling
            if fig is not None:
                fig.suptitle(f"{norm_symbol} - {timeframe}min", fontsize=12, color="gray", alpha=0.6)

                # improve layout so labels don't get cut off
                fig.tight_layout(rect=[0, 0, 1, 0.96])

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
