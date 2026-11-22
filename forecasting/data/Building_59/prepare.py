"""
Data preparation pipeline for LBNL Building 59 forecasting.

Reads the raw Building 59 CSVs (electrical end-use meters, on-site weather
station, zone CO2, zone heating/cooling setpoints, camera occupant counts,
WiFi device counts), aggregates everything to a common hourly grid, joins
open-meteo reanalysis weather and AQI, derives the consumption targets, and
writes the final processed CSV ready for model training.

Building 59 is an all-electric office building (Berkeley, CA). It meters
electricity by END USE (plug loads, lighting, HVAC) split North/South, but
has no heat meter, so thermal output cannot be recovered. We map the HVAC
electric input to the framework's "thermal" target and the plug+lighting
electricity to the "electric" target. There is no domestic-hot-water meter,
so dhw_kw is identically zero. See constants.py for the full mapping and the
important caveat that thermal_* columns are HVAC ELECTRIC kW, not thermal kW.

Usage
-----
    python forecasting/data/Building_59/prepare.py
    python forecasting/data/Building_59/prepare.py --raw_dir raw --out processed/data_hourly.csv
"""

import argparse
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
from forecasting.data.Building_59.constants import (
    SITE_TIMEZONE, DATASET_START, DATASET_END,
    HVAC_USES, PLUG_USES, LIGHT_USES,
    F_TO_C_OFFSET, F_TO_C_SCALE,
)

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Raw-file column maps
# ---------------------------------------------------------------------------
# Electrical end-use channels (kW). The raw file ships a trailing unnamed
# column; we select only the named channels below.
_ELE_CHANNELS = list(PLUG_USES) + list(LIGHT_USES) + list(HVAC_USES)

# On-site weather station (15-min). Raw -> clean name. Units already SI:
# air temp / dew point in degC, RH in %, solar radiation in W/m^2.
_SITE_WEATHER_COLS = {
    'air_temp_set_1':               'local_air_temp_c',
    'dew_point_temperature_set_1d': 'local_dew_point_c',
    'relative_humidity_set_1':      'local_rh_pct',
    'solar_radiation_set_1':        'local_solar_rad_wm2',
}

# open-meteo columns to keep (raw -> clean name). Mirrors HouseZero / UMAR so
# the consumption LSTM/GBDT feature schema lines up across datasets.
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
    'shortwave_radiation (W/m²)':       'shortwave_radiation',
    'direct_radiation (W/m²)':          'direct_radiation',
    'diffuse_radiation (W/m²)':         'diffuse_radiation',
    'global_tilted_irradiance (W/m²)':  'global_tilted_irradiance',
    'terrestrial_radiation (W/m²)':     'terrestrial_radiation',
    'direct_normal_irradiance (W/m²)':  'direct_normal_irradiance',
    'is_day ()':                        'is_day',
}


# ---------------------------------------------------------------------------
# Generic CSV readers
# ---------------------------------------------------------------------------

def _read_timeseries(path: str, date_col: str = 'date') -> pd.DataFrame:
    """Read a raw Building 59 CSV, parse the timestamp column, dedupe, sort.

    Building telemetry is logged in building-local (Pacific) wall-clock time.
    We keep timestamps naive (no tz) -- they define the canonical clock that
    open-meteo weather is later re-aligned to. Daylight-saving "fall back"
    can produce a duplicated local hour; we keep the first occurrence.
    """
    df = pd.read_csv(path, encoding='latin-1')
    # Drop trailing unnamed/blank columns produced by stray commas.
    df = df.loc[:, [c for c in df.columns if c and not str(c).startswith('Unnamed')]]
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()
    df = df[~df.index.duplicated(keep='first')]
    return df


def _make_hourly_index(start: str, end: str) -> pd.DatetimeIndex:
    return pd.date_range(start=start, end=end + ' 23:00', freq='h', tz=None)


def _resample_mean(df: pd.DataFrame, cols, idx: pd.DatetimeIndex,
                   ffill: int = 3) -> pd.DataFrame:
    """Hourly mean of the given columns, reindexed to ``idx`` with short fill."""
    present = [c for c in cols if c in df.columns]
    if not present:
        return pd.DataFrame(index=idx)
    hourly = df[present].apply(pd.to_numeric, errors='coerce').resample('h').mean()
    hourly = hourly.reindex(idx)
    if ffill:
        hourly = hourly.ffill(limit=ffill).bfill(limit=ffill)
    return hourly


# ---------------------------------------------------------------------------
# Electrical end-use loader
# ---------------------------------------------------------------------------

def _load_electric(raw_dir: str, idx: pd.DatetimeIndex) -> pd.DataFrame:
    path = os.path.join(raw_dir, 'ele.csv')
    print(f'  [ele] loading {os.path.basename(path)} ...')
    df = _read_timeseries(path)
    hourly = _resample_mean(df, _ELE_CHANNELS, idx)
    # Clip tiny negative meter noise to zero.
    for c in hourly.columns:
        hourly[c] = hourly[c].clip(lower=0)
    return hourly


# ---------------------------------------------------------------------------
# On-site weather loader (Berkeley Lab campus station, 15-min)
# ---------------------------------------------------------------------------

def _load_site_weather(raw_dir: str, idx: pd.DatetimeIndex) -> pd.DataFrame:
    path = os.path.join(raw_dir, 'site_weather.csv')
    print(f'  [site_weather] loading {os.path.basename(path)} ...')
    df = _read_timeseries(path)
    df = df.rename(columns=_SITE_WEATHER_COLS)
    cols = list(_SITE_WEATHER_COLS.values())
    hourly = _resample_mean(df, cols, idx)
    if 'local_solar_rad_wm2' in hourly.columns:
        hourly['local_solar_rad_wm2'] = hourly['local_solar_rad_wm2'].clip(lower=0)
    return hourly


# ---------------------------------------------------------------------------
# Zone setpoint / CO2 loaders
# ---------------------------------------------------------------------------

def _load_zone_setpoints(raw_dir: str, idx: pd.DatetimeIndex) -> pd.DataFrame:
    """Mean cooling / heating setpoint across all zones (degF -> degC)."""
    out = pd.DataFrame(index=idx)
    for fname, clean in [('zone_temp_sp_c.csv', 'zone_cooling_sp_c'),
                         ('zone_temp_sp_h.csv', 'zone_heating_sp_c')]:
        path = os.path.join(raw_dir, fname)
        if not os.path.exists(path):
            continue
        print(f'  [zone_sp] loading {fname} ...')
        df = _read_timeseries(path)
        sp_cols = [c for c in df.columns if c.endswith('_cooling_sp') or c.endswith('_heating_sp')]
        if not sp_cols:
            continue
        mean_f = df[sp_cols].apply(pd.to_numeric, errors='coerce').mean(axis=1)
        mean_c = (mean_f - F_TO_C_OFFSET) * F_TO_C_SCALE
        hourly = mean_c.resample('h').mean().reindex(idx).ffill(limit=6).bfill(limit=6)
        out[clean] = hourly
    return out


def _load_zone_co2(raw_dir: str, idx: pd.DatetimeIndex) -> pd.DataFrame:
    path = os.path.join(raw_dir, 'zone_co2.csv')
    out = pd.DataFrame(index=idx)
    if not os.path.exists(path):
        return out
    print(f'  [zone_co2] loading {os.path.basename(path)} ...')
    df = _read_timeseries(path)
    co2_cols = [c for c in df.columns if c.endswith('_co2')]
    if co2_cols:
        mean_co2 = df[co2_cols].apply(pd.to_numeric, errors='coerce').mean(axis=1)
        out['zone_co2_mean_ppm'] = mean_co2.resample('h').mean().reindex(idx) \
            .ffill(limit=3).bfill(limit=3)
    return out


# ---------------------------------------------------------------------------
# Occupancy loaders (camera counts + WiFi device counts)
# ---------------------------------------------------------------------------

def _load_occupancy(raw_dir: str, idx: pd.DatetimeIndex) -> pd.DataFrame:
    """Camera occupant counts (occ.csv) and WiFi device counts (wifi.csv).

    Camera counts end 2019-02-21; WiFi runs the full record. Both are good
    occupancy proxies. Missing intervals are treated as "no data" -> 0 occupant
    signal rather than propagated NaN, so the columns stay model-usable.
    """
    out = pd.DataFrame(index=idx)

    occ_path = os.path.join(raw_dir, 'occ.csv')
    if os.path.exists(occ_path):
        print('  [occ] loading occ.csv ...')
        occ = _read_timeseries(occ_path)
        occ_cols = [c for c in occ.columns if c.startswith('occ_')]
        if occ_cols:
            total = occ[occ_cols].apply(pd.to_numeric, errors='coerce').sum(axis=1)
            out['occupant_count'] = total.resample('h').mean().reindex(idx).fillna(0)

    wifi_path = os.path.join(raw_dir, 'wifi.csv')
    if os.path.exists(wifi_path):
        print('  [wifi] loading wifi.csv ...')
        wifi = _read_timeseries(wifi_path)
        wifi_cols = [c for c in wifi.columns if c.startswith('wifi_')]
        if wifi_cols:
            total = wifi[wifi_cols].apply(pd.to_numeric, errors='coerce').sum(axis=1)
            out['wifi_device_count'] = total.resample('h').mean().reindex(idx) \
                .ffill(limit=3).fillna(0)
    return out


# ---------------------------------------------------------------------------
# open-meteo weather loader (+ auto timezone alignment)
# ---------------------------------------------------------------------------

def _load_open_meteo_weather(weather_csv: str, idx: pd.DatetimeIndex) -> pd.DataFrame:
    """Parse the open-meteo CSV and return a frame reindexed to ``idx``.

    The Building 59 open-meteo export has NO metadata block (it starts directly
    with the 'time' header), so the timezone is unknown a priori. We read the
    naive timestamps as-is here; ``_auto_align_weather`` shifts them onto the
    building clock afterwards using the on-site pyranometer as ground truth.
    """
    weather_raw = pd.read_csv(weather_csv)
    weather_raw.columns = weather_raw.columns.str.strip().str.lstrip('﻿')
    time_col = 'time' if 'time' in weather_raw.columns else weather_raw.columns[0]
    weather_raw = weather_raw.rename(columns={time_col: 'date'})

    ts_re = r'^\d{4}-\d{2}-\d{2}T'
    weather_raw = weather_raw[weather_raw['date'].astype(str).str.match(ts_re)]
    weather_raw['date'] = pd.to_datetime(weather_raw['date'], format='ISO8601')
    weather_raw = weather_raw.set_index('date').sort_index()
    weather_raw = weather_raw[~weather_raw.index.duplicated(keep='first')]

    # Resample to a clean hourly grid spanning the dataset (no reindex to idx
    # yet -- alignment shift happens first, on the native grid).
    out = pd.DataFrame(index=weather_raw.index)
    for raw_col, clean_col in _WEATHER_COLS.items():
        if raw_col in weather_raw.columns:
            out[clean_col] = pd.to_numeric(weather_raw[raw_col], errors='coerce')
        else:
            out[clean_col] = np.nan
    return out


def _auto_align_weather(weather: pd.DataFrame, site: pd.DataFrame,
                        idx: pd.DatetimeIndex) -> pd.DataFrame:
    """Shift open-meteo weather onto the building clock.

    open-meteo's timestamps may be UTC or site-local depending on export
    settings, and this file carries no metadata to disambiguate. We find the
    integer-hour shift in [-12, +12] h that maximises the daytime correlation
    between open-meteo shortwave radiation and the on-site pyranometer
    (local_solar_rad_wm2), then apply it. Solar radiation has a sharp,
    unambiguous diurnal peak, so the best shift pins the timezone offset.
    """
    ref = site.get('local_solar_rad_wm2')
    cand = weather.get('shortwave_radiation')
    best_shift = 0
    if ref is not None and cand is not None:
        ref_h = pd.to_numeric(ref, errors='coerce')
        best_corr = -np.inf
        cors = {}
        for shift in range(-12, 13):
            shifted = cand.shift(shift).reindex(idx)
            r = ref_h.reindex(idx)
            mask = r.notna() & shifted.notna() & (r > 5)   # daytime only
            if mask.sum() > 500:
                c = float(np.corrcoef(r[mask], shifted[mask])[0, 1])
                cors[shift] = c
                if c > best_corr:
                    best_corr, best_shift = c, shift
        if cors:
            print(f'  [weather] best solar-alignment shift = {best_shift:+d} h '
                  f'(corr={best_corr:.3f})')
    else:
        print('  [weather] no on-site solar reference -- assuming weather is '
              'already on the building clock (shift 0)')

    shifted = weather.shift(best_shift)
    return shifted.reindex(idx)


# ---------------------------------------------------------------------------
# AQI loader
# ---------------------------------------------------------------------------

def _load_aqi(aqi_csv: str, idx: pd.DatetimeIndex) -> pd.DataFrame:
    """Daily AQI -> hourly via ffill. Columns: pm25 pm10 o3 no2 so2 co.

    NOTE: the supplied aqi.csv covers 2022-2024, which does NOT overlap the
    2018-2020 dataset window, so after filtering to [DATASET_START, DATASET_END]
    the frame is empty and every AQI column fills to 0. The columns are still
    emitted to keep the schema identical to the other datasets.
    """
    out = pd.DataFrame(index=idx)
    cols = ['pm25', 'pm10', 'o3', 'no2', 'so2', 'co']
    if not os.path.exists(aqi_csv):
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
    for c in cols:
        if c in hourly.columns:
            out[c] = pd.to_numeric(hourly[c], errors='coerce').fillna(0).values
        else:
            out[c] = 0.0
    if aqi_raw.empty:
        print('  [aqi] WARNING: aqi.csv does not overlap 2018-2020 -> all zeros')
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

    # --- 1. Load building telemetry -----------------------------------------
    print('[prepare] Loading building telemetry ...')
    ele      = _load_electric(raw_dir, idx)
    site_wx  = _load_site_weather(raw_dir, idx)
    zone_sp  = _load_zone_setpoints(raw_dir, idx)
    zone_co2 = _load_zone_co2(raw_dir, idx)
    occ      = _load_occupancy(raw_dir, idx)

    # ========================================================================
    # BUILD OUTPUT DATAFRAME
    # ========================================================================
    out = pd.DataFrame(index=idx)

    # ---- TARGETS -----------------------------------------------------------
    # HVAC electric input -> "thermal" target (electric kW, NOT thermal kW).
    hvac_cols  = [c for c in HVAC_USES  if c in ele.columns]
    plug_cols  = [c for c in PLUG_USES  if c in ele.columns]
    light_cols = [c for c in LIGHT_USES if c in ele.columns]

    hvac_total = ele[hvac_cols].sum(axis=1) if hvac_cols else pd.Series(0.0, index=idx)
    elec_total = ele[plug_cols + light_cols].sum(axis=1) if (plug_cols + light_cols) \
        else pd.Series(0.0, index=idx)

    # Calendar heating/cooling mask (Oct-15 -> Apr-15 per forecasting.utils).
    heat_mask = pd.Series([is_heating_season(ts.date()) for ts in idx], index=idx)

    out[TARGET_THERMAL]         = hvac_total.clip(lower=0)
    out[TARGET_THERMAL_HEATING] = hvac_total.where(heat_mask, 0.0).clip(lower=0)
    out[TARGET_THERMAL_COOLING] = hvac_total.where(~heat_mask, 0.0).clip(lower=0)
    out[TARGET_DHW]             = 0.0                       # no DHW meter
    out[TARGET_ELECTRIC]        = elec_total.clip(lower=0)

    # ---- End-use breakdown channels kept (analysis-friendly) ---------------
    for c in hvac_cols + plug_cols + light_cols:
        out[f'{c.lower()}_kw'] = ele[c]

    # ---- Calendar features -------------------------------------------------
    out['hour']        = idx.hour
    out['month']       = idx.month
    out['season']      = idx.month.map(get_season)
    out['day_of_week'] = idx.dayofweek
    bdays = pd.bdate_range(start=DATASET_START, end=DATASET_END + ' 23:00', freq='h')
    out['is_business'] = idx.isin(bdays).astype(int)
    out['is_heating_season'] = heat_mask.astype(int)

    # ---- Cyclic encodings --------------------------------------------------
    out['hour_sin']  = np.sin(2 * np.pi * idx.hour      / 24)
    out['hour_cos']  = np.cos(2 * np.pi * idx.hour      / 24)
    out['month_sin'] = np.sin(2 * np.pi * idx.month     / 12)
    out['month_cos'] = np.cos(2 * np.pi * idx.month     / 12)
    out['dow_sin']   = np.sin(2 * np.pi * idx.dayofweek / 7)
    out['dow_cos']   = np.cos(2 * np.pi * idx.dayofweek / 7)

    # ---- Indoor environment (zone aggregates) ------------------------------
    for col in zone_sp.columns:
        out[col] = zone_sp[col]
    for col in zone_co2.columns:
        out[col] = zone_co2[col]

    # ---- Occupancy proxies -------------------------------------------------
    for col in occ.columns:
        out[col] = occ[col]

    # ---- On-site weather station -------------------------------------------
    for col in site_wx.columns:
        out[col] = site_wx[col]

    # ========================================================================
    # JOIN OPEN-METEO WEATHER (auto-aligned to building clock)
    # ========================================================================
    print('[prepare] Joining open-meteo weather ...')
    weather_native = _load_open_meteo_weather(weather_csv, idx)
    weather = _auto_align_weather(weather_native, site_wx, idx)
    for col in weather.columns:
        out[col] = weather[col]

    # Solar-obstruction proxy: cloud cover during the day, 100% at night.
    # Mirrors HouseZero / UMAR so consumption feature whitelists line up.
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

    # Sanity: HVAC should be a sizeable fraction of total building electricity.
    tot_elec = out[TARGET_ELECTRIC].sum()
    tot_hvac = out[TARGET_THERMAL].sum()
    if (tot_elec + tot_hvac) > 0:
        frac = tot_hvac / (tot_elec + tot_hvac)
        print(f'  [sanity] HVAC / (HVAC + plug+light) electricity = {frac:.3f}')

    # Sanity: cross-correlate on-site pyranometer with open-meteo shortwave at
    # lag {-1, 0, +1} h after alignment. Zero lag MUST win.
    if 'local_solar_rad_wm2' in out.columns and 'shortwave_radiation' in out.columns:
        day_mask = out.get('is_day', pd.Series(1, index=out.index)) > 0.5
        a = out.loc[day_mask, 'local_solar_rad_wm2']
        b = out.loc[day_mask, 'shortwave_radiation']
        cors = {}
        for lag in (-1, 0, 1):
            shifted = b.shift(lag)
            valid   = a.notna() & shifted.notna()
            if valid.sum() > 100:
                cors[lag] = float(np.corrcoef(a[valid], shifted[valid])[0, 1])
        if cors:
            best = max(cors, key=cors.get)
            cors_str = '  '.join(f'lag{l:+d}={v:.3f}' for l, v in sorted(cors.items()))
            print(f'  [sanity] site-solar vs open-meteo SW:  {cors_str}')
            if best != 0:
                print(f'  [WARN] best correlation at lag {best:+d}h -- weather may be misaligned!')
            else:
                print('  [sanity] weather clock aligned with building (best lag = 0)')

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
    p = argparse.ArgumentParser(description='Prepare LBNL Building 59 dataset for forecasting.')
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
