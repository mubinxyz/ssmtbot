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
    Generate candlestick charts for a list of symbols with:
      - SSMT quarter vertical lines (01:30,07:30,13:30,19:30 local -> UTC+3:30)
      - Quarter labels (Q1..Q4)
      - HORIZONTAL LINES: previous-quarter HIGH and LOW starting at the actual candle
        where the high/low happened (within the previous quarter) and ending at the
        end of the next quarter interval.
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

            # ---- ALWAYS plot WITHOUT passing vlines to mplfinance ----
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

            # Collect Rectangle artists that represent candle bodies to compute exact centers (if possible)
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
            used_x_centers = None
            if rects:
                rects_sorted = sorted(rects, key=lambda r: r.get_x())
                x_centers = [r.get_x() + r.get_width() / 2.0 for r in rects_sorted]
                used_x_centers = x_centers

            # Map boundaries -> nearest row positions (numeric positions mplfinance often uses)
            boundary_row_positions: List[float] = []
            for b in filtered_boundaries:
                try:
                    pos = df.index.get_indexer([pd.Timestamp(b)], method="nearest")[0]
                    if pos == -1:
                        continue
                    boundary_row_positions.append(float(pos))
                except Exception as ex:
                    logger.debug("Failed to map boundary %r to row index: %s", b, ex)

            # --- draw vertical lines at boundaries (manual; uses row positions / centers) ---
            for idx, b in enumerate(filtered_boundaries):
                try:
                    pos = df.index.get_indexer([pd.Timestamp(b)], method="nearest")[0]
                    if pos == -1:
                        continue
                    if used_x_centers and pos < len(used_x_centers):
                        xcoord = used_x_centers[pos]
                        main_ax.axvline(x=xcoord, linestyle="--", linewidth=0.8, alpha=0.8, color="C1")
                    else:
                        main_ax.axvline(x=float(pos), linestyle="--", linewidth=0.8, alpha=0.8, color="C1")
                except Exception as inner_ex:
                    logger.debug("Failed to draw vline for %r: %s", b, inner_ex)

            # Build x coordinates for each filtered_boundary to place labels and to be used for horizontal spans
            x_coords_for_boundaries: List[float] = []
            for b in filtered_boundaries:
                try:
                    pos = df.index.get_indexer([pd.Timestamp(b)], method="nearest")[0]
                    if pos == -1:
                        continue
                    if used_x_centers and pos < len(used_x_centers):
                        x_coords_for_boundaries.append(used_x_centers[pos])
                    else:
                        x_coords_for_boundaries.append(float(pos))
                except Exception as ex:
                    logger.debug("Failed to compute x coord for boundary %r: %s", b, ex)

            # -----------------------------
            # Compute interval HIGH/LOW for each quarter interval
            # interval i is between filtered_boundaries[i] and filtered_boundaries[i+1]
            # then draw HIGH/LOW of interval i starting at the actual candle where the high/low happened
            # and ending at the end (right boundary) of the next interval (i+1 -> boundary i+2).
            # -----------------------------
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
            if high_col is None or low_col is None:
                logger.debug("[ChartService] High/Low columns not found for %s; skipping horizontal SSMT lines", norm_symbol)
            else:
                interval_highs = []
                interval_lows = []
                interval_high_ts = []  # actual timestamp where high occurred in interval
                interval_low_ts = []   # actual timestamp where low occurred in interval

                # compute highs/lows and the exact timestamp of those highs/lows for each interval
                for i in range(len(filtered_boundaries) - 1):
                    start_b = filtered_boundaries[i]
                    end_b = filtered_boundaries[i + 1]
                    try:
                        mask = (df.index >= pd.Timestamp(start_b)) & (df.index < pd.Timestamp(end_b))
                        segment = df.loc[mask]
                        if segment.empty:
                            interval_highs.append(None)
                            interval_lows.append(None)
                            interval_high_ts.append(None)
                            interval_low_ts.append(None)
                        else:
                            h = segment[high_col].max()
                            l = segment[low_col].min()
                            interval_highs.append(h if pd.notna(h) else None)
                            interval_lows.append(l if pd.notna(l) else None)

                            # timestamp(s) where high / low occur — choose first occurrence
                            try:
                                high_idx = segment[segment[high_col] == h].index
                                if len(high_idx):
                                    interval_high_ts.append(high_idx[0])
                                else:
                                    interval_high_ts.append(segment.index[segment[high_col].argmax()])
                            except Exception:
                                # fallback: idxmax
                                try:
                                    interval_high_ts.append(segment[high_col].idxmax())
                                except Exception:
                                    interval_high_ts.append(None)

                            try:
                                low_idx = segment[segment[low_col] == l].index
                                if len(low_idx):
                                    interval_low_ts.append(low_idx[0])
                                else:
                                    interval_low_ts.append(segment.index[segment[low_col].argmin()])
                            except Exception:
                                try:
                                    interval_low_ts.append(segment[low_col].idxmin())
                                except Exception:
                                    interval_low_ts.append(None)
                    except Exception as ex:
                        logger.debug("Failed to compute high/low for interval %r-%r: %s", start_b, end_b, ex)
                        interval_highs.append(None)
                        interval_lows.append(None)
                        interval_high_ts.append(None)
                        interval_low_ts.append(None)

                # now draw each previous interval's high/low starting from the actual candle timestamp
                # across the next interval (i -> i+1 requires boundaries i..i+2)
                for i in range(len(filtered_boundaries) - 2):
                    prev_h = interval_highs[i]
                    prev_l = interval_lows[i]
                    prev_h_ts = interval_high_ts[i]
                    prev_l_ts = interval_low_ts[i]
                    if prev_h is None and prev_l is None:
                        continue

                    # determine x_end = right boundary of next interval (boundary i+2)
                    try:
                        xend = x_coords_for_boundaries[i + 2]
                    except Exception:
                        # fallback: numeric pos of boundary i+2
                        try:
                            pos_end = df.index.get_indexer([pd.Timestamp(filtered_boundaries[i + 2])], method="nearest")[0]
                            xend = float(pos_end)
                        except Exception:
                            continue

                    # determine x_start: actual candle center for the prev-quarter high/low timestamp
                    def _timestamp_to_x(ts_val):
                        if ts_val is None:
                            return None
                        try:
                            # map timestamp to nearest row position
                            pos = df.index.get_indexer([pd.Timestamp(ts_val)], method="nearest")[0]
                            if pos == -1:
                                return None
                            if used_x_centers and pos < len(used_x_centers):
                                return used_x_centers[pos]
                            return float(pos)
                        except Exception:
                            return None

                    xstart_h = _timestamp_to_x(prev_h_ts)
                    xstart_l = _timestamp_to_x(prev_l_ts)

                    # if no valid start coordinate, try to use the left boundary of prev interval as fallback
                    if xstart_h is None and prev_h is not None:
                        try:
                            pos_left = df.index.get_indexer([pd.Timestamp(filtered_boundaries[i])], method="nearest")[0]
                            xstart_h = used_x_centers[pos_left] if (used_x_centers and pos_left < len(used_x_centers)) else float(pos_left)
                        except Exception:
                            xstart_h = None

                    if xstart_l is None and prev_l is not None:
                        try:
                            pos_left = df.index.get_indexer([pd.Timestamp(filtered_boundaries[i])], method="nearest")[0]
                            xstart_l = used_x_centers[pos_left] if (used_x_centers and pos_left < len(used_x_centers)) else float(pos_left)
                        except Exception:
                            xstart_l = None

                    # draw horizontal lines from the actual high/low candle to the end of next interval
                    try:
                        if prev_h is not None and xstart_h is not None:
                            main_ax.hlines(y=float(prev_h), xmin=xstart_h, xmax=xend,
                                           linewidth=0.9, colors="C2", linestyle="-", alpha=0.9, zorder=2)
                        if prev_l is not None and xstart_l is not None:
                            main_ax.hlines(y=float(prev_l), xmin=xstart_l, xmax=xend,
                                           linewidth=0.9, colors="C3", linestyle="-", alpha=0.9, zorder=2)
                    except Exception as ex:
                        logger.debug("Failed to draw horizontal prev-quarter lines for %s interval %d: %s", norm_symbol, i, ex)

            # Draw quarter labels centered between consecutive boundary x coordinates.
            if x_coords_for_boundaries and len(x_coords_for_boundaries) >= 2:
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

            # Custom title styling and save
            if fig is not None:
                fig.suptitle(f"{norm_symbol} - {timeframe}min", fontsize=12, color="gray", alpha=0.6)

                # improve layout so labels don't get cut off
                with suppress(Exception):
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
