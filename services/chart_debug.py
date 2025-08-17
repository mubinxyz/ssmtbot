# services/chart_debug.py  -- diagnostic helper (temporary)
import logging
import time
import pandas as pd
from dateutil import tz as dateutil_tz
from utils.get_data import get_ohlc
from utils.normalize_data import normalize_symbol

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("chart_debug")

# Replace these so we can test a single symbol quickly
TEST_SYMBOLS = ["XAU","EURUSD","DXY"]   # change to the symbol you want to test
TIMEFRAME = 15

# Config — same assumptions as production
SOURCE_TZ = dateutil_tz.tzoffset(None, 3 * 3600)  # LiteFinance = UTC+3
LOCAL_TZ_NAME = "Asia/Tehran"  # +3:30

def make_quarter_boundaries_utc(idx: pd.DatetimeIndex):
    """Create quarter boundaries in local tz (01:30,07:30,13:30,19:30,next-day 01:30) and return UTC-aware list"""
    local_offset = 210
    try:
        idx_local = idx.tz_convert(LOCAL_TZ_NAME)
        local_tz = idx_local.tz
    except Exception:
        local_tz = dateutil_tz.tzoffset(None, local_offset * 60)
        idx_local = idx.tz_convert(local_tz)
    start_local = idx_local.min().normalize()
    end_local = idx_local.max().normalize()
    days = pd.date_range(start=start_local, end=end_local, freq="D", tz=idx_local.tz)
    local_times = [(1,30),(7,30),(13,30),(19,30)]
    boundaries = []
    for d in days:
        for hh,mm in local_times:
            ts_local = pd.Timestamp(year=d.year, month=d.month, day=d.day, hour=hh, minute=mm, tz=d.tz)
            boundaries.append(ts_local.tz_convert("UTC"))
        nd = d + pd.Timedelta(days=1)
        ts_local = pd.Timestamp(year=nd.year, month=nd.month, day=nd.day, hour=1, minute=30, tz=d.tz)
        boundaries.append(ts_local.tz_convert("UTC"))
    return sorted(boundaries)

def debug_symbol(sym):
    norm = normalize_symbol(sym)
    to_date = int(time.time())
    from_date = to_date - 3*24*60*60
    df = get_ohlc(norm, timeframe=TIMEFRAME, from_date=from_date, to_date=to_date)
    logger.info("===== Debug for symbol: %s (normalized: %s) =====", sym, norm)
    if df is None:
        logger.info("get_ohlc returned None for %s", norm)
        return
    logger.info("raw df.columns: %s", df.columns.tolist())
    if "datetime" in df.columns:
        logger.info("raw datetime sample (first 30 rows):\n%s", df["datetime"].head(30).to_list())
        logger.info("dtype: %s; pandas dtype: %s", type(df["datetime"].iloc[0]) if len(df) else None, df["datetime"].dtype)
    else:
        logger.warning("No 'datetime' column in returned DF")

    # Try deterministic parse same as we do in production:
    try:
        # numeric epoch -> interpret as source tz then convert to UTC
        if pd.api.types.is_integer_dtype(df["datetime"]) or pd.api.types.is_float_dtype(df["datetime"]):
            s = pd.to_datetime(df["datetime"], unit="s", errors="coerce").dt.tz_localize(SOURCE_TZ).dt.tz_convert("UTC")
            logger.info("Interpreted datetimes as epoch seconds localized to SOURCE_TZ -> sample (UTC tz-aware):\n%s", s.head(10).to_list())
            idx = pd.DatetimeIndex(s)
        else:
            s = pd.to_datetime(df["datetime"], errors="coerce")
            logger.info("Parsed with pd.to_datetime -> sample dtype tzinfo: %s", "tz-aware" if s.dt.tz is not None else "naive")
            if s.dt.tz is None:
                # treat naive as SOURCE_TZ then convert to UTC
                s = s.dt.tz_localize(SOURCE_TZ).dt.tz_convert("UTC")
                logger.info("Interpreted naive as SOURCE_TZ -> sample (UTC):\n%s", s.head(10).to_list())
            else:
                s = s.dt.tz_convert("UTC")
                logger.info("Converted tz-aware to UTC -> sample:\n%s", s.head(10).to_list())
            idx = pd.DatetimeIndex(s)
    except Exception as e:
        logger.exception("Failed to parse datetimes: %s", e)
        return

    # Print index properties
    logger.info("Index tz: %s", idx.tz)
    logger.info("Index sample (first 10): %s", idx[:10].tolist())

    # Compute quarter boundaries in local tz, converted to UTC
    boundaries = make_quarter_boundaries_utc(idx)
    logger.info("Computed quarter boundaries (UTC):")
    for b in boundaries:
        try:
            # also print representation in local tz
            local = b.tz_convert(LOCAL_TZ_NAME)
        except Exception:
            local = b
        logger.info(" UTC=%s   LOCAL=%s", b.isoformat(), getattr(local, "isoformat", lambda: str(local))())

    # Map boundaries to nearest candles
    logger.info("Mapping boundaries to nearest candle indices/timestamps:")
    for b in boundaries:
        try:
            pos = idx.get_indexer([b], method="nearest")[0]
            if pos == -1:
                logger.info(" boundary %s -> no close candle", b)
            else:
                logger.info(" boundary %s -> nearest index pos %s ts=%s (value at pos open?)", b, pos, idx[pos])
                # print OHLC at that pos
                try:
                    row = df.iloc[pos].to_dict()
                    logger.info("   row @ pos=%s : %s", pos, {k: row[k] for k in ("Open","High","Low","Close") if k in row})
                except Exception:
                    pass
        except Exception as e:
            logger.debug("mapping failure: %s", e)

for sym in TEST_SYMBOLS:
    debug_symbol(sym)
