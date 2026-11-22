"""
HEEW - open-meteo and AQI loaders.

The HEEW raw archive ships a single ``weather.csv`` from open-meteo that
concatenates *two* blocks separated by a blank line:

    block 1: hourly readings (2014-01-01T00:00 ... 2022-12-31T23:00)
    block 2: daily summaries  (2014-01-01      ... 2022-12-31)

We only consume the hourly block. The daily block has fewer columns and
would otherwise confuse pandas; the regex date filter below drops it
naturally (daily rows lack the 'T' delimiter), but we keep an explicit
slice-on-blank-line cut so column-mismatch warnings stay quiet.

Open-meteo timezone handling
----------------------------
The published HEEW weather.csv has ``timezone = America/Los_Angeles`` and
``utc_offset_seconds = -25200`` (i.e. -7 h fixed). The label is misleading
(Los Angeles is UTC-8 in winter), but the timestamps themselves are at a
fixed UTC-7 clock that matches Phoenix MST year-round. We therefore trust
the offset over the label and treat the values as already aligned with
``SITE_TIMEZONE = America/Phoenix``.

Split out of prepare.py to keep each file under the cowork workspace's
single-write byte budget.
"""

import os
from io import StringIO

import numpy as np
import pandas as pd

from forecasting.data.HEEW.constants import (
    SITE_TIMEZONE, DATASET_START, DATASET_END,
)


# Raw open-meteo header -> clean column name. Aligned with KSRP / HouseZero
# so downstream feature lists stay portable across datasets.
OM_WEATHER_COLS = {
    'temperature_2m (°C)':              'temperature_2m',
    'relative_humidity_2m (%)':         'relative_humidity_2m',
    'dew_point_2m (°C)':                'dew_point_2m',
    'apparent_temperature (°C)':        'apparent_temperature',
    'rain (mm)':                        'rain_precipitation',
    'snowfall (cm)':                    'snowfall_precipitation',
    'pressure_msl (hPa)':               'pressure_msl',
    'surface_pressure (hPa)':           'surface_pressure',
    'cloud_cover (%)':                  'cloud_cover',
    'et0_fao_evapotranspiration (mm)':  'et0_fao_evapotranspiration',
    'vapour_pressure_deficit (kPa)':    'vapour_pressure_deficit',
    'wind_speed_10m (km/h)':            'wind_speed_10m',
    'wind_speed_100m (km/h)':           'wind_speed_100m',
    'wind_direction_10m (°)':           'wind_direction_10m',
    'wind_direction_100m (°)':          'wind_direction_100m',
    'wind_gusts_10m (km/h)':            'wind_gusts_10m',
    'boundary_layer_height (m)':        'boundary_layer_height',
    'sunshine_duration (s)':            'sunshine_duration',
    'shortwave_radiation (W/m²)':       'shortwave_radiation',
    'direct_radiation (W/m²)':          'direct_radiation',
    'diffuse_radiation (W/m²)':         'diffuse_radiation',
    'global_tilted_irradiance (W/m²)':  'global_tilted_irradiance',
    'terrestrial_radiation (W/m²)':     'terrestrial_radiation',
    'direct_normal_irradiance (W/m²)':  'direct_normal_irradiance',
    'is_day ()':                        'is_day',
}


def _read_hourly_block(path: str) -> pd.DataFrame:
    """Return only the hourly portion of a HEEW open-meteo CSV.

    The file begins with a metadata header, a blank line, the hourly data
    header, the hourly rows, another blank line, then the daily data
    header and daily rows. We scan once to find the first 'time' header
    line (start of hourly block) and the next blank line after it (end).
    """
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    header_idx = None
    utc_offset = None
    tz_name = None
    for i, ln in enumerate(lines):
        cells = [c.strip().strip('﻿') for c in ln.split(',')]
        first = cells[0].lower() if cells else ''
        if first == 'latitude' and 'utc_offset_seconds' in [c.lower() for c in cells]:
            cols = [c.lower() for c in cells]
            try:
                off_col = cols.index('utc_offset_seconds')
                tz_col  = cols.index('timezone')
            except ValueError:
                continue
            for j in range(i + 1, min(i + 4, len(lines))):
                vals = [v.strip() for v in lines[j].split(',')]
                if len(vals) >= max(off_col, tz_col) + 1 and vals[0]:
                    try:
                        utc_offset = int(vals[off_col])
                    except (ValueError, IndexError):
                        utc_offset = None
                    tz_name = vals[tz_col] or None
                    break
        if first == 'time' and header_idx is None:
            header_idx = i
            break

    if header_idx is None:
        print(f'  [warn] cannot find hourly data header in {path}')
        return pd.DataFrame()

    # Find the end of the hourly block — first blank line after header_idx
    # (HEEW's open-meteo CSV uses a blank line as the daily-block separator).
    end_idx = len(lines)
    for k in range(header_idx + 1, len(lines)):
        if not lines[k].strip():
            end_idx = k
            break

    print(f'  [weather] {os.path.basename(path):30s}  tz={tz_name}  '
          f'utc_offset={utc_offset}s  hourly_rows={end_idx - header_idx - 1:,}')

    wraw = pd.read_csv(StringIO(''.join(lines[header_idx:end_idx])))
    wraw.columns = wraw.columns.str.strip().str.lstrip('﻿')
    wraw = wraw.rename(columns={'time': 'date'})

    # Keep only true ISO-hourly rows. Belt-and-braces: even if the daily
    # block leaked past the blank-line cut, daily rows lack the 'T' in the
    # timestamp and are filtered out here.
    ts_re = r'^\d{4}-\d{2}-\d{2}T'
    wraw = wraw[wraw['date'].astype(str).str.match(ts_re)]
    naive = pd.to_datetime(wraw['date'], format='ISO8601', errors='coerce')

    # HEEW open-meteo is published with a fixed UTC-7 clock (utc_offset_seconds
    # = -25200) and a misleading 'America/Los_Angeles' label. SITE_TIMEZONE is
    # 'America/Phoenix' (also UTC-7 fixed, no DST), so the data is already
    # aligned with the site clock; no re-anchoring required.
    if utc_offset == -25200:
        wraw['date'] = naive
    elif tz_name and tz_name == SITE_TIMEZONE:
        wraw['date'] = naive
    elif tz_name:
        try:
            anchored = naive.dt.tz_localize(
                tz_name, ambiguous='infer', nonexistent='shift_forward')
            wraw['date'] = anchored.dt.tz_convert(SITE_TIMEZONE).dt.tz_localize(None)
            print(f'    re-anchored {tz_name} -> {SITE_TIMEZONE}')
        except Exception:
            wraw['date'] = naive
    else:
        wraw['date'] = naive

    wraw = wraw.dropna(subset=['date']).set_index('date').sort_index()
    wraw = wraw[~wraw.index.duplicated(keep='first')]
    return wraw


def load_open_meteo(weather_csvs, idx):
    """Parse one or more open-meteo CSV files and return a frame reindexed to ``idx``."""
    if not weather_csvs:
        print('  [weather] no open-meteo CSV(s) -- canonical block will be zero-filled.')
        return pd.DataFrame(index=idx)

    frames = []
    for path in weather_csvs:
        if not os.path.isfile(path):
            print(f'  [warn] open-meteo file missing: {path}')
            continue
        wraw = _read_hourly_block(path)
        if wraw is not None and not wraw.empty:
            frames.append(wraw)

    if not frames:
        return pd.DataFrame(index=idx)
    combined = pd.concat(frames).sort_index()
    combined = combined[~combined.index.duplicated(keep='first')]

    out = pd.DataFrame(index=idx)
    for raw_col, clean_col in OM_WEATHER_COLS.items():
        if raw_col in combined.columns:
            out[clean_col] = pd.to_numeric(combined[raw_col], errors='coerce').reindex(idx)
        else:
            out[clean_col] = 0.0
    return out


def load_aqi(aqi_csv, idx):
    """Daily AQI -> hourly via ffill. Columns: pm25 pm10 o3 no2 so2 co.

    The HEEW raw AQI file contains some dates outside the dataset window
    (e.g. 2026 entries); we slice to ``[DATASET_START, DATASET_END]`` before
    reindexing. Missing channels (e.g. so2 / co rarely reported in the
    Phoenix metropolitan network) are zero-filled rather than dropped to
    keep the feature schema stable.
    """
    cols = ('pm25', 'pm10', 'o3', 'no2', 'so2', 'co')
    if not aqi_csv or not os.path.isfile(aqi_csv):
        out = pd.DataFrame(index=idx)
        for c in cols:
            out[c] = 0.0
        return out
    aqi_raw = pd.read_csv(aqi_csv, encoding='latin-1')
    aqi_raw.columns = aqi_raw.columns.str.strip()
    aqi_raw['date'] = pd.to_datetime(aqi_raw['date'].astype(str).str.strip(),
                                     format='%Y/%m/%d', errors='coerce')
    aqi_raw = aqi_raw.dropna(subset=['date']).sort_values('date').set_index('date')
    aqi_raw = aqi_raw.loc[DATASET_START:DATASET_END]
    hourly = aqi_raw.reindex(idx, method=None).ffill(limit=24).bfill(limit=24)
    out = pd.DataFrame(index=idx)
    for c in cols:
        if c in hourly.columns:
            out[c] = pd.to_numeric(hourly[c], errors='coerce').fillna(0).values
        else:
            out[c] = 0.0
    return out


def default_weather_paths(raw_dir: str) -> list:
    """Discover the open-meteo CSV inside ``raw_dir``."""
    cand = []
    for nm in ('weather.csv', 'weather_2014-2022.csv', 'open_meteo.csv'):
        p = os.path.join(raw_dir, nm)
        if os.path.isfile(p):
            cand.append(p)
    return cand


# ---------------------------------------------------------------------------
# Trailing padding -- cowork mount safety. See KSRP_Campus/_weather.py.
# ---------------------------------------------------------------------------

_COWORK_MOUNT_PADDING = (
    'X' * 200
)
