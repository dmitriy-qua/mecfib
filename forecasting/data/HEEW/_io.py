"""
HEEW - flat CSV readers for energy and (auxiliary) NWS weather.

The HEEW raw layout is much simpler than KSRP's per-yearly-Excel archive:
all energy records for a community live in a single CSV with
``Year, Month, Day, Hour, Weekday, Electricity, PV, Cooling, Heat,
Emission, Total Energy`` columns. The companion ``Total_weather.csv`` carries
the National Weather Service readings (Imperial units) at the same hourly
grid. Both are read here.

ASCII-only (no em-dashes) to remain safe under workspace mount limits.
"""

import os

import numpy as np
import pandas as pd


def _build_datetime(df: pd.DataFrame) -> pd.DatetimeIndex:
    """Construct a naive hourly DatetimeIndex from Year/Month/Day/Hour."""
    return pd.to_datetime(dict(
        year   = pd.to_numeric(df['Year'],  errors='coerce').astype('Int64'),
        month  = pd.to_numeric(df['Month'], errors='coerce').astype('Int64'),
        day    = pd.to_numeric(df['Day'],   errors='coerce').astype('Int64'),
        hour   = pd.to_numeric(df['Hour'],  errors='coerce').astype('Int64'),
    ), errors='coerce')


def load_energy_csv(path: str) -> pd.DataFrame:
    """Read a HEEW community / building energy CSV.

    Expected columns (case-insensitive substring match is intentionally NOT
    applied — the HEEW raw files use a fixed schema):
        Year, Month, Day, Hour, Weekday,
        Electricity, PV, Cooling, Heat, Emission, Total Energy

    Returns a frame indexed on the local-time hourly grid with the energy
    columns preserved in their raw units (Electricity kW, PV kW, Cooling Ton,
    Heat mmBTU/hr, Emission Ton, Total Energy mmBTU/hr). Unit conversion is
    handled in ``prepare.py``.
    """
    if not os.path.isfile(path):
        print(f'  [warn] energy file not found: {path}')
        return pd.DataFrame()
    print(f'  [energy] reading {os.path.basename(path)}')
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    ts = _build_datetime(df)
    df['date'] = ts
    df = df.dropna(subset=['date']).set_index('date').sort_index()
    df = df[~df.index.duplicated(keep='first')]
    print(f'    rows = {len(df):,}  range = {df.index[0]}  ->  {df.index[-1]}')
    return df


def load_nws_weather_csv(path: str) -> pd.DataFrame:
    """Read the HEEW Total_weather.csv (National Weather Service, Imperial).

    Expected columns:
        Year, Month, Day, Hour, DOW,
        Temperature (F), Dew Point (F), Humidity (%), Wind Speed (mph),
        Wind Gust (mph), Pressure (in Hg), Precip (in)

    The returned frame is indexed on the local-time hourly grid with
    Imperial-unit columns preserved. Conversion to metric / canonical
    ``local_*`` names happens in ``prepare.py``.
    """
    if not os.path.isfile(path):
        print(f'  [warn] NWS weather file not found: {path}')
        return pd.DataFrame()
    print(f'  [nws] reading {os.path.basename(path)}')
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    ts = _build_datetime(df)
    df['date'] = ts
    df = df.dropna(subset=['date']).set_index('date').sort_index()
    df = df[~df.index.duplicated(keep='first')]
    print(f'    rows = {len(df):,}')
    return df


def reindex_hourly(df: pd.DataFrame, idx: pd.DatetimeIndex,
                   ffill_limit: int = 3) -> pd.DataFrame:
    """Reindex a frame to ``idx`` with short forward/backward fill."""
    if df is None or df.empty:
        return pd.DataFrame(index=idx)
    return df.reindex(idx).ffill(limit=ffill_limit).bfill(limit=ffill_limit)


# ---------------------------------------------------------------------------
# Trailing padding — see KSRP_Campus/_io.py for the rationale (cowork mount
# fills tail-truncated bytes with NULs, which Python rejects as 'source code
# string cannot contain null bytes').
# ---------------------------------------------------------------------------

_COWORK_MOUNT_PADDING = (
    'X' * 200
)
