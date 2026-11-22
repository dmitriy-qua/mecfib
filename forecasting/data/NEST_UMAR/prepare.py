"""
Data preparation pipeline for NEST UMAR forecasting.

Reads the four yearly UMAR wide-format CSVs (2019-07-01 -> 2023-07-01,
minute resolution), aggregates to an hourly grid, joins open-meteo weather
and AQI, derives the four consumption targets (thermal_heating_kw,
thermal_cooling_kw, dhw_kw, electric_demand_kw) and writes the final
processed CSV ready for model training.

UMAR is a residential apartment at the NEST building (EMPA Duebendorf, CH).
There is no on-site PV / solar-thermal generation, so this pipeline produces
consumption-only outputs. Per-target columns expected by
forecasting.consumption.constants.LSTM_COLUMNS are produced (with sensible
fallbacks/zeros where UMAR has no equivalent sensor, e.g. indoor RH/CO2).

Usage
-----
    python forecasting/data/NEST_UMAR/prepare.py
    python forecasting/data/NEST_UMAR/prepare.py --raw_dir raw --out processed/data_hourly.csv
"""

import argparse
import glob
import os
import sys
import warnings

import numpy as np
import pandas as pd

# Allow running as a script from any CWD
_DIR       = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_DIR)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from forecasting.constants import (
    TARGET_THERMAL, TARGET_THERMAL_HEATING, TARGET_THERMAL_COOLING,
    TARGET_DHW, TARGET_ELECTRIC,
)
from forecasting.utils import get_season, is_heating_season

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Site / dataset constants (Zurich, NEST UMAR)
# ---------------------------------------------------------------------------
# UMAR's timestamps are CET WITHOUT DST (fixed UTC+1, "winter time year-round").
# This is the convention used at NEST/EMPA so the minute-level series stays
# continuous through DST transitions (verified empirically: 1440 rows/day all
# year, and a +1h lag-correlation peak between the UMAR roof irradiation
# sensor and open-meteo's irradiance after re-anchoring weather from
# Europe/Berlin to UTC+1).  Etc/GMT-1 in IANA == UTC+1 (sign is inverted).
SITE_TIMEZONE         = 'Etc/GMT-1'         # UMAR clock: UTC+1 fixed
SITE_TIMEZONE_DISPLAY = 'CET (UTC+1, no DST)'
WEATHER_NATIVE_TZ     = 'Europe/Berlin'     # open-meteo emits Berlin local
DATASET_START = '2019-07-01'
DATASET_END   = '2023-06-30'

# Raw filename pattern (yearly wide CSVs in raw_dir)
_RAW_GLOB = 'umar_*_wide_descId.csv'

# UMAR columns -- physical layout of the apartment
_ROOMS         = ['272', '273', '274', '275', '276']
_TEMP_COLS     = [f'temp_{r}'  for r in _ROOMS]
_SETP_COLS     = [f'setp_{r}'  for r in _ROOMS]
_WINDOW_COLS   = ['window_272', 'window_273_1', 'window_273_2', 'window_274']
_VALVE_COLS    = ['y1_272', 'y1_273', 'y2_273', 'y3_273', 'y1_274', 'y1_275', 'y1_276']

# Power columns (kW) -- direct minute-level measurements
_POWER_HEATING = 'heating_power'
_POWER_COOLING = 'cooling_power'
_POWER_DHW     = 'dhw_power'
_POWER_TOTAL   = 'total_active_power'
_VOL_DHW       = 'dhw_volume'

# Outdoor sensors on the building roof
_OUTDOOR_TEMP  = 'temp_amb'
_OUTDOOR_IRRAD = 'irrad'

# Open-meteo weather columns to keep (raw -> clean name).
# Mirrors HouseZero so the consumption LSTM/GBDT feature schema lines up.
_WEATHER_COLS = {
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
    'direct_radiation (W/m²)':          'direct_radiation',
    'diffuse_radiation (W/m²)':         'diffuse_radiation',
    'global_tilted_irradiance (W/m²)':  'global_tilted_irradiance',
    'terrestrial_radiation (W/m²)':     'terrestrial_radiation',
    'direct_normal_irradiance (W/m²)':  'direct_normal_irradiance',
    'is_day ()':                        'is_day',
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_umar_raw(raw_dir: str) -> pd.DataFrame:
    """Concatenate every yearly UMAR file in raw_dir into one minute-level frame."""
    pattern = os.path.join(raw_dir, _RAW_GLOB)
    paths   = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f'No UMAR files matched {pattern}')

    frames = []
    for p in paths:
        print(f'  [umar] loading {os.path.basename(p)} ...')
        df = pd.read_csv(p, encoding='latin-1')
        # The first column is an unnamed row index ("", "1", "2", ...). Drop it.
        first = df.columns[0]
        if first == '' or first.startswith('Unnamed'):
            df = df.drop(columns=first)
        df['time'] = pd.to_datetime(df['time'])
        df = df.set_index('time').sort_index()
        frames.append(df)

    combined = pd.concat(frames).sort_index()
    combined = combined[~combined.index.duplicated(keep='first')]
    return combined


def _aggregate_hourly(minute_df: pd.DataFrame, hourly_idx: pd.DatetimeIndex) -> pd.DataFrame:
    """Aggregate minute-resolution data to the hourly grid.

    Power & temperatures use the mean (continuous signals). Window/valve flags
    (0/1) also use the mean, which equals the fraction of the hour the
    window/valve was open -- a useful feature in the [0, 1] range.
    """
    rules: dict[str, str] = {}

    for col in [
        _POWER_HEATING, _POWER_COOLING, _POWER_DHW, _POWER_TOTAL,
        _VOL_DHW, _OUTDOOR_TEMP, _OUTDOOR_IRRAD,
        *_TEMP_COLS, *_SETP_COLS, *_WINDOW_COLS, *_VALVE_COLS,
    ]:
        if col in minute_df.columns:
            rules[col] = 'mean'

    print(f'  [agg] aggregating {len(minute_df):,} minute rows -> hourly ...')
    hourly = minute_df[list(rules)].resample('h').agg(rules)

    # Align to the canonical hourly index (fills any gaps with NaN).
    hourly = hourly.reindex(hourly_idx)
    # Short-gap fill: ffill/bfill up to ~3 h, matches HouseZero conventions.
    hourly = hourly.ffill(limit=3).bfill(limit=3)
    return hourly


def _make_hourly_index(start: str, end: str) -> pd.DatetimeIndex:
    return pd.date_range(start=start, end=end + ' 23:00', freq='h', tz=None)


def _load_open_meteo_weather(weather_csv: str, idx: pd.DatetimeIndex) -> pd.DataFrame:
    """Parse the open-meteo CSV (metadata block + hourly + daily block) and
    return a frame reindexed to ``idx`` with cleaned column names.

    Open-meteo emits timestamps in the timezone passed to its API. For UMAR
    that is Europe/Berlin (DST-aware), but UMAR's own clock is CET fixed
    (UTC+1, no DST). Both clocks therefore differ by 1 hour during DST, so
    weather must be re-anchored to UMAR's clock to keep features and targets
    aligned.

    Conversion rules:
      * tz declared and matches SITE_TIMEZONE  -> already on UMAR clock; just strip tz.
      * tz declared (e.g. Europe/Berlin)       -> localize to that tz (DST aware),
                                                  convert to SITE_TIMEZONE, strip tz.
      * utc_offset == 0                        -> data is UTC; convert to SITE_TIMEZONE.
      * utc_offset declared, no tz             -> re-anchor via Etc/GMT+/-N fallback.
      * nothing declared                       -> assume already on UMAR clock (warn).
    """
    from io import StringIO

    with open(weather_csv, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    header_idx = None
    utc_offset = None
    tz_name    = None
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
        if first == 'time':
            header_idx = i
            break

    if header_idx is None:
        raise ValueError(f'Cannot find data header in {weather_csv}')
    print(f'  [weather] open-meteo metadata: tz={tz_name}  utc_offset={utc_offset}s')
    print(f'  [weather] target clock = {SITE_TIMEZONE}  ({SITE_TIMEZONE_DISPLAY})')

    weather_raw = pd.read_csv(StringIO(''.join(lines[header_idx:])))
    weather_raw.columns = weather_raw.columns.str.strip().str.lstrip('﻿')
    weather_raw = weather_raw.rename(columns={'time': 'date'})

    # Drop any stray repeated-header rows (daily block has its own 'time' header).
    ts_re = r'^\d{4}-\d{2}-\d{2}T'
    weather_raw = weather_raw[weather_raw['date'].astype(str).str.match(ts_re)]

    naive = pd.to_datetime(weather_raw['date'], format='ISO8601')

    if tz_name and tz_name == SITE_TIMEZONE:
        # Already on UMAR's clock; nothing to do.
        weather_raw['date'] = naive
    elif tz_name:
        # Open-meteo emitted timestamps in `tz_name` local time (e.g. Europe/Berlin
        # with DST). Re-anchor to UMAR's fixed CET clock so summer/winter rows
        # line up with the building telemetry.
        try:
            anchored = naive.dt.tz_localize(
                tz_name, ambiguous='infer', nonexistent='shift_forward',
            )
        except Exception:
            # Fall back to naive utc_offset re-anchoring (no DST awareness).
            anchored = naive.dt.tz_localize(f'Etc/GMT{-utc_offset // 3600:+d}') \
                if utc_offset is not None else naive.dt.tz_localize('UTC')
        weather_raw['date'] = anchored.dt.tz_convert(SITE_TIMEZONE).dt.tz_localize(None)
        print(f'  [weather] converted {tz_name} -> {SITE_TIMEZONE}')
    elif utc_offset == 0:
        weather_raw['date'] = (
            naive.dt.tz_localize('UTC')
                 .dt.tz_convert(SITE_TIMEZONE)
                 .dt.tz_localize(None)
        )
        print(f'  [weather] converted UTC -> {SITE_TIMEZONE}')
    elif utc_offset is not None:
        weather_raw['date'] = (
            naive.dt.tz_localize(f'Etc/GMT{-utc_offset // 3600:+d}')
                 .dt.tz_convert(SITE_TIMEZONE)
                 .dt.tz_localize(None)
        )
        print(f'  [weather] converted Etc/GMT{-utc_offset // 3600:+d} -> {SITE_TIMEZONE}')
    else:
        print('  [weather] no tz/utc_offset metadata - assuming timestamps are on UMAR clock')
        weather_raw['date'] = naive

    weather_raw = weather_raw.set_index('date')
    # DST "fall back" creates two UTC hours that map to the same local time;
    # keep the first occurrence so the index is unique before reindex().
    weather_raw = weather_raw[~weather_raw.index.duplicated(keep='first')]

    out = pd.DataFrame(index=idx)
    for raw_col, clean_col in _WEATHER_COLS.items():
        if raw_col in weather_raw.columns:
            out[clean_col] = pd.to_numeric(weather_raw[raw_col], errors='coerce').reindex(idx)
        else:
            out[clean_col] = 0.0
    return out


def _load_aqi(aqi_csv: str, idx: pd.DatetimeIndex) -> pd.DataFrame:
    """Daily AQI -> hourly via ffill, columns: pm25 pm10 o3 no2 so2 co."""
    aqi_raw = pd.read_csv(aqi_csv, encoding='latin-1')
    aqi_raw.columns = aqi_raw.columns.str.strip()
    aqi_raw['date'] = pd.to_datetime(aqi_raw['date'].astype(str).str.strip(),
                                     format='%Y/%m/%d', errors='coerce')
    aqi_raw = aqi_raw.dropna(subset=['date']).sort_values('date').set_index('date')
    aqi_raw = aqi_raw.loc[DATASET_START:DATASET_END]
    hourly = aqi_raw.reindex(idx, method=None).ffill(limit=24).bfill(limit=24)

    out = pd.DataFrame(index=idx)
    for col in ['pm25', 'pm10', 'o3', 'no2', 'so2', 'co']:
        if col in hourly.columns:
            out[col] = pd.to_numeric(hourly[col], errors='coerce').fillna(0).values
        else:
            out[col] = 0.0
    return out


# ---------------------------------------------------------------------------
# Main preparation function
# ---------------------------------------------------------------------------

def prepare_dataset(
    raw_dir:     str,
    weather_csv: str,
    aqi_csv:     str,
    output_path: str | None = None,
) -> pd.DataFrame:

    print('[prepare] Building hourly index ...')
    idx = _make_hourly_index(DATASET_START, DATASET_END)

    # --- 1. Load minute-level UMAR files ------------------------------------
    print('[prepare] Loading UMAR yearly files ...')
    minute_df = _load_umar_raw(raw_dir)
    print(f'  Loaded {len(minute_df):,} minute rows  '
          f'({minute_df.index[0]} -> {minute_df.index[-1]})')

    # --- 2. Aggregate to hourly ---------------------------------------------
    hourly = _aggregate_hourly(minute_df, idx)

    # ========================================================================
    # BUILD OUTPUT DATAFRAME
    # ========================================================================
    out = pd.DataFrame(index=idx)

    # ---- TARGETS -----------------------------------------------------------
    out[TARGET_THERMAL_HEATING] = hourly[_POWER_HEATING].clip(lower=0)
    out[TARGET_THERMAL_COOLING] = hourly[_POWER_COOLING].clip(lower=0)
    out[TARGET_THERMAL]         = (out[TARGET_THERMAL_HEATING] + out[TARGET_THERMAL_COOLING]).clip(lower=0)
    out[TARGET_DHW]             = hourly[_POWER_DHW].clip(lower=0)
    out[TARGET_ELECTRIC]        = hourly[_POWER_TOTAL].clip(lower=0)

    # ---- Calendar features -------------------------------------------------
    out['hour']        = idx.hour
    out['month']       = idx.month
    out['season']      = idx.month.map(get_season)
    out['day_of_week'] = idx.dayofweek
    bdays = pd.bdate_range(start=DATASET_START, end=DATASET_END + ' 23:00', freq='h')
    out['is_business'] = idx.isin(bdays).astype(int)
    out['is_heating_season'] = pd.Series(
        [int(is_heating_season(ts.date())) for ts in idx], index=idx
    )

    # ---- Cyclic encodings --------------------------------------------------
    out['hour_sin']  = np.sin(2 * np.pi * idx.hour      / 24)
    out['hour_cos']  = np.cos(2 * np.pi * idx.hour      / 24)
    out['month_sin'] = np.sin(2 * np.pi * idx.month     / 12)
    out['month_cos'] = np.cos(2 * np.pi * idx.month     / 12)
    out['dow_sin']   = np.sin(2 * np.pi * idx.dayofweek / 7)
    out['dow_cos']   = np.cos(2 * np.pi * idx.dayofweek / 7)

    # ---- Indoor aggregates --------------------------------------------------
    temp_present = [c for c in _TEMP_COLS if c in hourly.columns]
    setp_present = [c for c in _SETP_COLS if c in hourly.columns]
    out['zone_temp_mean_c']     = hourly[temp_present].mean(axis=1) if temp_present else np.nan
    out['zone_setpoint_mean_c'] = hourly[setp_present].mean(axis=1) if setp_present else np.nan

    # Per-room temperature columns kept (useful for room-level analysis)
    for c in temp_present:
        out[c] = hourly[c]
    for c in setp_present:
        out[c] = hourly[c]

    # ---- Window / valve aggregates (fraction of the hour open, in [0, 1]) ---
    win_present = [c for c in _WINDOW_COLS if c in hourly.columns]
    val_present = [c for c in _VALVE_COLS  if c in hourly.columns]
    out['window_open_frac'] = hourly[win_present].mean(axis=1).clip(0, 1) if win_present else 0.0
    out['valve_open_frac']  = hourly[val_present].mean(axis=1).clip(0, 1) if val_present else 0.0

    # Per-room window / valve channels kept alongside the aggregates so the
    # prepared CSV preserves every raw UMAR sensor (mirrors the temp/setp
    # treatment above). Values are in [0, 1] = fraction of the hour open.
    for c in win_present:
        out[c] = hourly[c].clip(0, 1)
    for c in val_present:
        out[c] = hourly[c].clip(0, 1)

    # ---- DHW volume flow ----------------------------------------------------
    if _VOL_DHW in hourly.columns:
        out['dhw_volume_m3h'] = hourly[_VOL_DHW].clip(lower=0)

    # ---- Local outdoor (UMAR's own roof sensors) ----------------------------
    if _OUTDOOR_TEMP in hourly.columns:
        out['local_air_temp_c'] = hourly[_OUTDOOR_TEMP]
    if _OUTDOOR_IRRAD in hourly.columns:
        out['local_solar_rad_wm2'] = hourly[_OUTDOOR_IRRAD].clip(lower=0)

    # ========================================================================
    # JOIN OPEN-METEO WEATHER
    # ========================================================================
    print('[prepare] Joining open-meteo weather ...')
    weather = _load_open_meteo_weather(weather_csv, idx)
    for col in weather.columns:
        out[col] = weather[col]

    # Solar-obstruction proxy: cloud cover during the day, 100 % at night.
    # Mirrors HouseZero so consumption feature whitelists line up.
    is_day_series = out.get('is_day', pd.Series(0, index=idx)).fillna(0)
    out['solar_obstruction_pct'] = (
        out['cloud_cover'].fillna(0) * is_day_series
        + 100 * (1 - is_day_series)
    ).clip(0, 100)

    # ========================================================================
    # JOIN AQI
    # ========================================================================
    print('[prepare] Joining AQI data ...')
    aqi = _load_aqi(aqi_csv, idx)
    for col in aqi.columns:
        out[col] = aqi[col]

    # ========================================================================
    # GAP FILLING & VALIDATION
    # ========================================================================
    print('[prepare] Filling remaining gaps ...')
    numeric_cols = out.select_dtypes(include=[np.number]).columns.tolist()
    out[numeric_cols] = out[numeric_cols].ffill(limit=3).bfill(limit=3)

    before = len(out)
    out = out.dropna(subset=[
        TARGET_THERMAL_HEATING, TARGET_THERMAL_COOLING,
        TARGET_DHW, TARGET_ELECTRIC,
    ])
    dropped = before - len(out)
    if dropped:
        print(f'  [info] Dropped {dropped} rows with missing target values.')

    # Sanity: thermal + DHW should be a sizeable fraction of total electric.
    # NOTE: heating_power / cooling_power are thermal output; total_active_power
    # is electric input, so a ratio above 1.0 reflects heat-pump COP, not error.
    if _POWER_TOTAL in hourly.columns:
        total = out[TARGET_ELECTRIC].clip(lower=0).sum()
        thd   = (out[TARGET_THERMAL] + out[TARGET_DHW]).sum()
        if total > 0:
            print(f'  [sanity] (thermal + dhw) / total_electric = {thd / total:.3f}')

    # Sanity: cross-correlate UMAR's roof irradiance with open-meteo's tilted
    # irradiance at lag {-1, 0, +1} h. Zero lag MUST win after the timezone
    # conversion -- if -1 or +1 wins, weather is misaligned with UMAR.
    if 'local_solar_rad_wm2' in out.columns and 'global_tilted_irradiance' in out.columns:
        day_mask = out.get('is_day', pd.Series(1, index=out.index)) > 0.5
        a = out.loc[day_mask, 'local_solar_rad_wm2']
        b = out.loc[day_mask, 'global_tilted_irradiance']
        cors = {}
        for lag in (-1, 0, 1):
            shifted = b.shift(lag)
            valid   = a.notna() & shifted.notna()
            if valid.sum() > 100:
                cors[lag] = float(np.corrcoef(a[valid], shifted[valid])[0, 1])
        best = max(cors, key=cors.get) if cors else None
        cors_str = '  '.join(f'lag{l:+d}={v:.3f}' for l, v in sorted(cors.items()))
        print(f'  [sanity] roof-irrad vs open-meteo GTI:  {cors_str}')
        if best is not None and best != 0:
            print(f'  [WARN] best correlation at lag {best:+d}h -- weather may be misaligned!')
        else:
            print(f'  [sanity] weather clock aligned with UMAR (best lag = 0)')

    print(f'[prepare] Final dataset: {len(out)} rows x {len(out.columns)} columns.')
    print(f'  Date range: {out.index[0]}  to  {out.index[-1]}')

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        out.index.name = 'date'
        out.to_csv(output_path)
        print(f'[prepare] Saved to {output_path}')

    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description='Prepare NEST UMAR dataset for forecasting.')
    base = _DIR
    p.add_argument('--raw_dir',     default=os.path.join(base, 'raw'))
    p.add_argument('--weather_csv', default=os.path.join(base, 'raw', 'weather.csv'))
    p.add_argument('--aqi_csv',     default=os.path.join(base, 'raw', 'aqi.csv'))
    p.add_argument('--out',         default=os.path.join(base, 'processed', 'data_hourly.csv'))
    return p.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    prepare_dataset(
        raw_dir=args.raw_dir,
        weather_csv=args.weather_csv,
        aqi_csv=args.aqi_csv,
        output_path=args.out,
    )
