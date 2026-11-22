"""
Data preparation pipeline for the KSRP Campus 20-year dataset.

Heavy lifting is split into sibling modules:
  _io.py         folder / Excel I/O + fuzzy column matchers (incl. the
                 per-yearly-file ``load_folder_canonical`` matcher)
  _weather.py    open-meteo + AQI loaders
  _buildings.py  per-building water-loop + load aggregator
  _corr.py       publication-style correlation heatmaps

Per-yearly-file canonical matching (via ``load_folder_canonical``) is the
right behaviour when a unit-label drifts between years (KSRP 2008 publishes
``Cooling load (GJ)`` while every other year publishes ``Cooling load (kW)``,
yet both store kW-magnitude values). The matcher claims whichever column is
present in each year without double-counting.

Usage
-----
    python forecasting/data/KSRP_Campus/prepare.py
    python forecasting/data/KSRP_Campus/prepare.py --raw_dir raw --out processed/data_hourly.csv

If you see ``ImportError: cannot import name 'LOAD_RULES'`` or
``source code string cannot contain null bytes``, delete any stale
``__pycache__`` next to this file:
    rm -rf forecasting/data/KSRP_Campus/__pycache__
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd

_DIR       = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_DIR)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from forecasting.constants import (
    TARGET_THERMAL, TARGET_THERMAL_HEATING, TARGET_THERMAL_COOLING,
    TARGET_DHW, TARGET_ELECTRIC, TARGET_PV,
)
from forecasting.utils import get_season, is_heating_season
from forecasting.data.KSRP_Campus.constants import (
    DATASET_START, DATASET_END, OUTAGE_WINDOWS, PV_OP_START, PV_OP_END,
    YEAR_CALIBRATION,
)
from forecasting.data.KSRP_Campus._io import (
    load_folder_canonical, to_hourly,
)
from forecasting.data.KSRP_Campus._weather import (
    load_open_meteo, load_aqi, default_weather_paths,
)
from forecasting.data.KSRP_Campus._buildings import load_individual_buildings

warnings.filterwarnings('ignore')


# Column rules (verified against the 2001-2022 raw files).
POWER_RULES = {
    'pv_electric_kw':           [['solar', 'energy'], ['pv'], ['photovolt']],
    'grid_import_kw':           [['grid'], ['utility'], ['imported']],
    'gas_engine_electric_kw':   [['gas', 'engine']],
    'fuel_cell_electric_kw':    [['fuel', 'cell']],
}

GAS_RULES = {
    'gas_boiler_gas_m3':         [['boiler']],
    'fuel_cell_gas_m3':          [['fuel', 'cell']],
    'gas_engine_gas_m3':         [['gas', 'engine']],
    'absorption_chiller_gas_m3': [['absorption']],
}

# fuel-cell rule first so 'gas' doesn't swallow fuel-cell columns.
HEATREC_RULES = {
    'fuel_cell_heat_gjh':        [['fuel', 'cell']],
    'gas_engine_heat_gjh':       [['gas']],
}

LOAD_RULES = {
    TARGET_ELECTRIC:        [['electricity', 'load'], ['electric', 'load']],
    TARGET_THERMAL_HEATING: [['heating', 'load']],
    TARGET_THERMAL_COOLING: [['cooling', 'load']],
    TARGET_DHW:             [['hot', 'water']],
}

WEATHER_RULES = {
    'local_solar_rad_wm2': [['solar', 'irradi'], ['horizontal', 'irradi'], ['irradi']],
    'local_air_temp_c':    [['outdoor', 'air', 'temperature'], ['outdoor', 'temperature'],
                            ['air', 'temperature'], ['ambient', 'temperature']],
    'local_rh_pct':        [['outdoor', 'humidity'], ['humidity']],
    'local_wind_speed_ms': [['wind', 'speed']],
    'local_wind_dir':      [['wind', 'direction']],
}


def _make_hourly_index(start, end):
    return pd.date_range(start=start, end=end + ' 23:00', freq='h', tz=None)


def _apply_outage_mask(out):
    keep = pd.Series(True, index=out.index)
    for s, e in OUTAGE_WINDOWS:
        keep &= ~((out.index >= s) & (out.index <= e))
    dropped = (~keep).sum()
    if dropped:
        print(f'  [outage] dropping {dropped} rows in {OUTAGE_WINDOWS}')
    return out[keep]


def prepare_dataset(raw_dir, weather_csvs=None, aqi_csv=None, output_path=None):
    print('[prepare] Building hourly index ...')
    idx = _make_hourly_index(DATASET_START, DATASET_END)
    print(f'  hourly grid: {idx[0]} ... {idx[-1]}  ({len(idx)} rows)')

    print('[prepare] Reading raw folders (per-yearly-file matching) ...')
    power_can   = load_folder_canonical(raw_dir, 'power',         POWER_RULES,   'power')
    gas_can     = load_folder_canonical(raw_dir, 'gas_usage',     GAS_RULES,     'gas_usage')
    heat_can    = load_folder_canonical(raw_dir, 'heat_recovery', HEATREC_RULES, 'heat_recovery')
    load_can    = load_folder_canonical(raw_dir, 'loads',         LOAD_RULES,    'loads')
    weather_can = load_folder_canonical(raw_dir, 'weather',       WEATHER_RULES, 'weather')

    power_h   = to_hourly(power_can,   idx)
    gas_h     = to_hourly(gas_can,     idx)
    heat_h    = to_hourly(heat_can,    idx)
    load_h    = to_hourly(load_can,    idx)
    weather_h = to_hourly(weather_can, idx)

    print('[prepare] Reading per-building water-loop + load files ...')
    bldg_agg = load_individual_buildings(raw_dir, idx)

    out = pd.DataFrame(index=idx)

    if TARGET_THERMAL_HEATING in load_h.columns:
        out[TARGET_THERMAL_HEATING] = load_h[TARGET_THERMAL_HEATING].clip(lower=0)
    if TARGET_THERMAL_COOLING in load_h.columns:
        out[TARGET_THERMAL_COOLING] = load_h[TARGET_THERMAL_COOLING].clip(lower=0)
    out[TARGET_THERMAL] = (
        out.get(TARGET_THERMAL_HEATING, 0).fillna(0)
        + out.get(TARGET_THERMAL_COOLING, 0).fillna(0)
    ).clip(lower=0)
    if TARGET_DHW in load_h.columns:
        out[TARGET_DHW] = load_h[TARGET_DHW].clip(lower=0)
    if TARGET_ELECTRIC in load_h.columns:
        out[TARGET_ELECTRIC] = load_h[TARGET_ELECTRIC].clip(lower=0)

    if 'pv_electric_kw' in power_h.columns:
        pv = power_h['pv_electric_kw'].clip(lower=0)
        pv = pv.where((pv.index >= PV_OP_START) & (pv.index <= PV_OP_END), 0.0)
        out[TARGET_PV] = pv

    # ---- Per-year calibration --------------------------------------------
    # Apply empirical scaling factors to years whose loads file uses
    # non-kW units (see constants.YEAR_CALIBRATION; 2014 is the only such
    # year in the present archive). Shape of the seasonal/diurnal pattern
    # is preserved; only the overall magnitude is calibrated to match the
    # surrounding years.
    for yr, factors in YEAR_CALIBRATION.items():
        yr_mask = (out.index.year == yr)
        if not yr_mask.any():
            continue
        for col, scale in factors.items():
            if col in out.columns:
                col_mask = yr_mask & out[col].notna()
                if col_mask.any():
                    out.loc[col_mask, col] = out.loc[col_mask, col] * scale
                    print(f'  [calib] {yr} {col}: x {scale:.4f}  '
                          f'({col_mask.sum()} rows)')

    # Re-derive combined thermal after calibration
    if TARGET_THERMAL_HEATING in out.columns or TARGET_THERMAL_COOLING in out.columns:
        out[TARGET_THERMAL] = (
            out.get(TARGET_THERMAL_HEATING, 0).fillna(0)
            + out.get(TARGET_THERMAL_COOLING, 0).fillna(0)
        ).clip(lower=0)

    out['hour']        = idx.hour
    out['month']       = idx.month
    out['season']      = idx.month.map(get_season)
    out['day_of_week'] = idx.dayofweek
    bdays = pd.bdate_range(start=DATASET_START, end=DATASET_END + ' 23:00', freq='h')
    out['is_business'] = idx.isin(bdays).astype(int)
    out['is_heating_season'] = pd.Series(
        [int(is_heating_season(ts.date())) for ts in idx], index=idx)
    out['hour_sin']  = np.sin(2 * np.pi * idx.hour      / 24)
    out['hour_cos']  = np.cos(2 * np.pi * idx.hour      / 24)
    out['month_sin'] = np.sin(2 * np.pi * idx.month     / 12)
    out['month_cos'] = np.cos(2 * np.pi * idx.month     / 12)
    out['dow_sin']   = np.sin(2 * np.pi * idx.dayofweek / 7)
    out['dow_cos']   = np.cos(2 * np.pi * idx.dayofweek / 7)

    for col in ('gas_engine_electric_kw', 'fuel_cell_electric_kw', 'grid_import_kw'):
        if col in power_h.columns:
            out[col] = power_h[col]
    for col in ('gas_engine_gas_m3', 'fuel_cell_gas_m3',
                'gas_boiler_gas_m3', 'absorption_chiller_gas_m3'):
        if col in gas_h.columns:
            out[col] = gas_h[col]
    for col in ('gas_engine_heat_gjh', 'fuel_cell_heat_gjh'):
        if col in heat_h.columns:
            out[col] = heat_h[col]

    for col in bldg_agg.columns:
        if not bldg_agg[col].isna().all():
            out[col] = bldg_agg[col]

    for col in ('local_solar_rad_wm2', 'local_air_temp_c', 'local_rh_pct',
                'local_wind_speed_ms', 'local_wind_dir'):
        if col in weather_h.columns:
            out[col] = weather_h[col]

    print('[prepare] Joining open-meteo weather ...')
    om = load_open_meteo(weather_csvs or [], idx)
    for col in om.columns:
        out[col] = om[col]

    if 'is_day' not in out.columns or out['is_day'].isna().all():
        if 'local_solar_rad_wm2' in out.columns:
            out['is_day'] = (out['local_solar_rad_wm2'].fillna(0) > 5).astype(int)
        else:
            out['is_day'] = ((out['hour'] >= 6) & (out['hour'] <= 18)).astype(int)
    out['is_day'] = pd.to_numeric(out['is_day'], errors='coerce').fillna(0).clip(0, 1).astype(int)

    if 'cloud_cover' in out.columns:
        isd = out['is_day']
        out['solar_obstruction_pct'] = (
            out['cloud_cover'].fillna(0) * isd + 100 * (1 - isd)
        ).clip(0, 100)
    else:
        out['solar_obstruction_pct'] = 100.0

    print('[prepare] Joining AQI data ...')
    aqi = load_aqi(aqi_csv or '', idx)
    for col in aqi.columns:
        out[col] = aqi[col]

    print('[prepare] Applying outage mask + filling short gaps ...')
    out = _apply_outage_mask(out)

    numeric_cols = out.select_dtypes(include=[np.number]).columns.tolist()
    out[numeric_cols] = out[numeric_cols].ffill(limit=3).bfill(limit=3)

    # Drop rows only when EVERY consumption target is missing (very rare;
    # mostly empty padding rows). 2014 stays in the dataset because the
    # calibration step above has converted its non-kW raw values into the
    # surrounding-years' magnitude range.
    targets_present = [c for c in (TARGET_THERMAL_HEATING, TARGET_THERMAL_COOLING,
                                   TARGET_DHW, TARGET_ELECTRIC) if c in out.columns]
    before = len(out)
    if targets_present:
        out = out.dropna(subset=targets_present, how='all')
    dropped = before - len(out)
    if dropped:
        print(f'  [info] Dropped {dropped} rows with ALL target values missing.')

    if TARGET_ELECTRIC in out.columns:
        total_e = out[TARGET_ELECTRIC].clip(lower=0).sum()
        total_t = (out[TARGET_THERMAL] + out.get(TARGET_DHW, 0)).sum()
        if total_e > 0:
            print(f'  [sanity] (thermal+dhw)/electric over 20 y = {total_t / total_e:.3f}')
    if TARGET_PV in out.columns:
        pv_win = out.loc[(out.index >= PV_OP_START) & (out.index <= PV_OP_END), TARGET_PV]
        if not pv_win.empty:
            print(f'  [sanity] PV peak = {pv_win.max():.1f} kW  (nameplate 153 kW)')

    print(f'[prepare] Final dataset: {len(out)} rows x {len(out.columns)} columns.')
    print(f'  Date range: {out.index[0]}  to  {out.index[-1]}')

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        out.index.name = 'date'
        out.to_csv(output_path)
        print(f'[prepare] Saved to {output_path}')

    return out


def _parse_args():
    p = argparse.ArgumentParser(description='Prepare KSRP Campus dataset for forecasting.')
    base = _DIR
    p.add_argument('--raw_dir', default=os.path.join(base, 'raw'))
    p.add_argument('--weather_csv', default=None, action='append',
                   help='Open-meteo CSV. Pass multiple times. Default: auto-discover.')
    p.add_argument('--aqi_csv', default=None,
                   help='AQI CSV. Default: <raw_dir>/aqi.csv if present.')
    p.add_argument('--out', default=os.path.join(base, 'processed', 'data_hourly.csv'))
    return p.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    weather_csvs = args.weather_csv or default_weather_paths(args.raw_dir)
    aqi_csv = args.aqi_csv or os.path.join(args.raw_dir, 'aqi.csv')
    prepare_dataset(
        raw_dir=args.raw_dir,
        weather_csvs=weather_csvs,
        aqi_csv=aqi_csv if os.path.isfile(aqi_csv) else None,
        output_path=args.out,
    )


# ===========================================================================
# COWORK MOUNT PADDING - do not delete
# ===========================================================================
# The cowork workspace filesystem caches the historical maximum size of each
# file and pads any shorter rewrite with null bytes. Python rejects source
# files with null bytes ('source code string cannot contain null bytes').
# The block below is an inert string literal that fills the remaining bytes
# up to the historical maximum. On the user's local disk this padding is
# harmless; inside the cowork mount it keeps the module loadable.
# If you see ImportError or 'null bytes' when running this in cowork mode,
# delete __pycache__ first: rm -rf forecasting/data/KSRP_Campus/__pycache__
# ===========================================================================

_COWORK_PADDING_DO_NOT_REMOVE = """
filler-line-001 filler-line-001 filler-line-001 filler-line-001 filler-line-001
filler-line-002 filler-line-002 filler-line-002 filler-line-002 filler-line-002
filler-line-003 filler-line-003 filler-line-003 filler-line-003 filler-line-003
filler-line-004 filler-line-004 filler-line-004 filler-line-004 filler-line-004
filler-line-005 filler-line-005 filler-line-005 filler-line-005 filler-line-005
filler-line-006 filler-line-006 filler-line-006 filler-line-006 filler-line-006
filler-line-007 filler-line-007 filler-line-007 filler-line-007 filler-line-007
filler-line-008 filler-line-008 filler-line-008 filler-line-008 filler-line-008
filler-line-009 filler-line-009 filler-line-009 filler-line-009 filler-line-009
filler-line-010 filler-line-010 filler-line-010 filler-line-010 filler-line-010
filler-line-011 filler-line-011 filler-line-011 filler-line-011 filler-line-011
filler-line-012 filler-line-012 filler-line-012 filler-line-012 filler-line-012
filler-line-013 filler-line-013 filler-line-013 filler-line-013 filler-line-013
filler-line-014 filler-line-014 filler-line-014 filler-line-014 filler-line-014
filler-line-015 filler-line-015 filler-line-015 filler-line-015 filler-line-015
filler-line-016 filler-line-016 filler-line-016 filler-line-016 filler-line-016
filler-line-017 filler-line-017 filler-line-017 filler-line-017 filler-line-017
filler-line-018 filler-line-018 filler-line-018 filler-line-018 filler-line-018
filler-line-019 filler-line-019 filler-line-019 filler-line-019 filler-line-019
filler-line-020 filler-line-020 filler-line-020 filler-line-020 filler-line-020
filler-line-021 filler-line-021 filler-line-021 filler-line-021 filler-line-021
filler-line-022 filler-line-022 filler-line-022 filler-line-022 filler-line-022
filler-line-023 filler-line-023 filler-line-023 filler-line-023 filler-line-023
filler-line-024 filler-line-024 filler-line-024 filler-line-024 filler-line-024
filler-line-025 filler-line-025 filler-line-025 filler-line-025 filler-line-025
filler-line-026 filler-line-026 filler-line-026 filler-line-026 filler-line-026
filler-line-027 filler-line-027 filler-line-027 filler-line-027 filler-line-027
filler-line-028 filler-line-028 filler-line-028 filler-line-028 filler-line-028
filler-line-029 filler-line-029 filler-line-029 filler-line-029 filler-line-029
filler-line-030 filler-line-030 filler-line-030 filler-line-030 filler-line-030
filler-line-031 filler-line-031 filler-line-031 filler-line-031 filler-line-031
filler-line-032 filler-line-032 filler-line-032 filler-line-032 filler-line-032
filler-line-033 filler-line-033 filler-line-033 filler-line-033 filler-line-033
filler-line-034 filler-line-034 filler-line-034 filler-line-034 filler-line-034
filler-line-035 filler-line-035 filler-line-035 filler-line-035 filler-line-035
filler-line-036 filler-line-036 filler-line-036 filler-line-036 filler-line-036
filler-line-037 filler-line-037 filler-line-037 filler-line-037 filler-line-037
filler-line-038 filler-line-038 filler-line-038 filler-line-038 filler-line-038
filler-line-039 filler-line-039 filler-line-039 filler-line-039 filler-line-039
filler-line-040 filler-line-040 filler-line-040 filler-line-040 filler-line-040
filler-line-041 filler-line-041 filler-line-041 filler-line-041 filler-line-041
filler-line-042 filler-line-042 filler-line-042 filler-line-042 filler-line-042
filler-line-043 filler-line-043 filler-line-043 filler-line-043 filler-line-043
filler-line-044 filler-line-044 filler-line-044 filler-line-044 filler-line-044
filler-line-045 filler-line-045 filler-line-045 filler-line-045 filler-line-045
filler-line-046 filler-line-046 filler-line-046 filler-line-046 filler-line-046
filler-line-047 filler-line-047 filler-line-047 filler-line-047 filler-line-047
filler-line-048 filler-line-048 filler-line-048 filler-line-048 filler-line-048
filler-line-049 filler-line-049 filler-line-049 filler-line-049 filler-line-049
filler-line-050 filler-line-050 filler-line-050 filler-line-050 filler-line-050
filler-line-051 filler-line-051 filler-line-051 filler-line-051 filler-line-051
filler-line-052 filler-line-052 filler-line-052 filler-line-052 filler-line-052
filler-line-053 filler-line-053 filler-line-053 filler-line-053 filler-line-053
filler-line-054 filler-line-054 filler-line-054 filler-line-054 filler-line-054
filler-line-055 filler-line-055 filler-line-055 filler-line-055 filler-line-055
filler-line-056 filler-line-056 filler-line-056 filler-line-056 filler-line-056
filler-line-057 filler-line-057 filler-line-057 filler-line-057 filler-line-057
filler-line-058 filler-line-058 filler-line-058 filler-line-058 filler-line-058
filler-line-059 filler-line-059 filler-line-059 filler-line-059 filler-line-059
filler-line-060 filler-line-060 filler-line-060 filler-line-060 filler-line-060
filler-line-061 filler-line-061 filler-line-061 filler-line-061 filler-line-061
filler-line-062 filler-line-062 filler-line-062 filler-line-062 filler-line-062
filler-line-063 filler-line-063 filler-line-063 filler-line-063 filler-line-063
filler-line-064 filler-line-064 filler-line-064 filler-line-064 filler-line-064
filler-line-065 filler-line-065 filler-line-065 filler-line-065 filler-line-065
filler-line-066 filler-line-066 filler-line-066 filler-line-066 filler-line-066
filler-line-067 filler-line-067 filler-line-067 filler-line-067 filler-line-067
filler-line-068 filler-line-068 filler-line-068 filler-line-068 filler-line-068
filler-line-069 filler-line-069 filler-line-069 filler-line-069 filler-line-069
filler-line-070 filler-line-070 filler-line-070 filler-line-070 filler-line-070
filler-line-071 filler-line-071 filler-line-071 filler-line-071 filler-line-071
filler-line-072 filler-line-072 filler-line-072 filler-line-072 filler-line-072
filler-line-073 filler-line-073 filler-line-073 filler-line-073 filler-line-073
filler-line-074 filler-line-074 filler-line-074 filler-line-074 filler-line-074
filler-line-075 filler-line-075 filler-line-075 filler-line-075 filler-line-075
filler-line-076 filler-line-076 filler-line-076 filler-line-076 filler-line-076
filler-line-077 filler-line-077 filler-line-077 filler-line-077 filler-line-077
filler-line-078 filler-line-078 filler-line-078 filler-line-078 filler-line-078
filler-line-079 filler-line-079 filler-line-079 filler-line-079 filler-line-079
filler-line-080 filler-line-080 filler-line-080 filler-line-080 filler-line-080
filler-line-081 filler-line-081 filler-line-081 filler-line-081 filler-line-081
filler-line-082 filler-line-082 filler-line-082 filler-line-082 filler-line-082
filler-line-083 filler-line-083 filler-line-083 filler-line-083 filler-line-083
filler-line-084 filler-line-084 filler-line-084 filler-line-084 filler-line-084
filler-line-085 filler-line-085 filler-line-085 filler-line-085 filler-line-085
filler-line-086 filler-line-086 filler-line-086 filler-line-086 filler-line-086
filler-line-087 filler-line-087 filler-line-087 filler-line-087 filler-line-087
filler-line-088 filler-line-088 filler-line-088 filler-line-088 filler-line-088
filler-line-089 filler-line-089 filler-line-089 filler-line-089 filler-line-089
filler-line-090 filler-line-090 filler-line-090 filler-line-090 filler-line-090
filler-line-091 filler-line-091 filler-line-091 filler-line-091 filler-line-091
filler-line-092 filler-line-092 filler-line-092 filler-line-092 filler-line-092
filler-line-093 filler-line-093 filler-line-093 filler-line-093 filler-line-093
filler-line-094 filler-line-094 filler-line-094 filler-line-094 filler-line-094
filler-line-095 filler-line-095 filler-line-095 filler-line-095 filler-line-095
filler-line-096 filler-line-096 filler-line-096 filler-line-096 filler-line-096
filler-line-097 filler-line-097 filler-line-097 filler-line-097 filler-line-097
filler-line-098 filler-line-098 filler-line-098 filler-line-098 filler-line-098
filler-line-099 filler-line-099 filler-line-099 filler-line-099 filler-line-099
filler-line-100 filler-line-100 filler-line-100 filler-line-100 filler-line-100
filler-line-101 filler-line-101 filler-line-101 filler-line-101 filler-line-101
filler-line-102 filler-line-102 filler-line-102 filler-line-102 filler-line-102
filler-line-103 filler-line-103 filler-line-103 filler-line-103 filler-line-103
filler-line-104 filler-line-104 filler-line-104 filler-line-104 filler-line-104
filler-line-105 filler-line-105 filler-line-105 filler-line-105 filler-line-105
filler-line-106 filler-line-106 filler-line-106 filler-line-106 filler-line-106
filler-line-107 filler-line-107 filler-line-107 filler-line-107 filler-line-107
filler-line-108 filler-line-108 filler-line-108 filler-line-108 filler-line-108
filler-line-109 filler-line-109 filler-line-109 filler-line-109 filler-line-109
filler-line-110 filler-line-110 filler-line-110 filler-line-110 filler-line-110
filler-line-111 filler-line-111 filler-line-111 filler-line-111 filler-line-111
filler-line-112 filler-line-112 filler-line-112 filler-line-112 filler-line-112
filler-line-113 filler-line-113 filler-line-113 filler-line-113 filler-line-113
filler-line-114 filler-line-114 filler-line-114 filler-line-114 filler-line-114
filler-line-115 filler-line-115 filler-line-115 filler-line-115 filler-line-115
filler-line-116 filler-line-116 filler-line-116 filler-line-116 filler-line-116
filler-line-117 filler-line-117 filler-line-117 filler-line-117 filler-line-117
filler-line-118 filler-line-118 filler-line-118 filler-line-118 filler-line-118
filler-line-119 filler-line-119 filler-line-119 filler-line-119 filler-line-119
filler-line-120 filler-line-120 filler-line-120 filler-line-120 filler-line-120
filler-line-121 filler-line-121 filler-line-121 filler-line-121 filler-line-121
filler-line-122 filler-line-122 filler-line-122 filler-line-122 filler-line-122
filler-line-123 filler-line-123 filler-line-123 filler-line-123 filler-line-123
filler-line-124 filler-line-124 filler-line-124 filler-line-124 filler-line-124
filler-line-125 filler-line-125 filler-line-125 filler-line-125 filler-line-125
filler-line-126 filler-line-126 filler-line-126 filler-line-126 filler-line-126
filler-line-127 filler-line-127 filler-line-127 filler-line-127 filler-line-127
filler-line-128 filler-line-128 filler-line-128 filler-line-128 filler-line-128
filler-line-129 filler-line-129 filler-line-129 filler-line-129 filler-line-129
filler-line-130 filler-line-130 filler-line-130 filler-line-130 filler-line-130
filler-line-131 filler-line-131 filler-line-131 filler-line-131 filler-line-131
filler-line-132 filler-line-132 filler-line-132 filler-line-132 filler-line-132
filler-line-133 filler-line-133 filler-line-133 filler-line-133 filler-line-133
filler-line-134 filler-line-134 filler-line-134 filler-line-134 filler-line-134
filler-line-135 filler-line-135 filler-line-135 filler-line-135 filler-line-135
filler-line-136 filler-line-136 filler-line-136 filler-line-136 filler-line-136
filler-line-137 filler-line-137 filler-line-137 filler-line-137 filler-line-137
filler-line-138 filler-line-138 filler-line-138 filler-line-138 filler-line-138
filler-line-139 filler-line-139 filler-line-139 filler-line-139 filler-line-139
filler-line-140 filler-line-140 filler-line-140 filler-line-140 filler-line-140
filler-line-141 filler-line-141 filler-line-141 filler-line-141 filler-line-141
filler-line-142 filler-line-142 filler-line-142 filler-line-142 filler-line-142
filler-line-143 filler-line-143 filler-line-143 filler-line-143 filler-line-143
filler-line-144 filler-line-144 filler-line-144 filler-line-144 filler-line-144
filler-line-145 filler-line-145 filler-line-145 filler-line-145 filler-line-145
filler-line-146 filler-line-146 filler-line-146 filler-line-146 filler-line-146
filler-line-147 filler-line-147 filler-line-147 filler-line-147 filler-line-147
filler-line-148 filler-line-148 filler-line-148 filler-line-148 filler-line-148
filler-line-149 filler-line-149 filler-line-149 filler-line-149 filler-line-149
filler-line-150 filler-line-150 filler-line-150 filler-line-150 filler-line-150
filler-line-151 filler-line-151 filler-line-151 filler-line-151 filler-line-151
filler-line-152 filler-line-152 filler-line-152 filler-line-152 filler-line-152
filler-line-153 filler-line-153 filler-line-153 filler-line-153 filler-line-153
filler-line-154 filler-line-154 filler-line-154 filler-line-154 filler-line-154
filler-line-155 filler-line-155 filler-line-155 filler-line-155 filler-line-155
filler-line-156 filler-line-156 filler-line-156 filler-line-156 filler-line-156
filler-line-157 filler-line-157 filler-line-157 filler-line-157 filler-line-157
filler-line-158 filler-line-158 filler-line-158 filler-line-158 filler-line-158
filler-line-159 filler-line-159 filler-line-159 filler-line-159 filler-line-159
filler-line-160 filler-line-160 filler-line-160 filler-line-160 filler-line-160
filler-line-161 filler-line-161 filler-line-161 filler-line-161 filler-line-161
filler-line-162 filler-line-162 filler-line-162 filler-line-162 filler-line-162
filler-line-163 filler-line-163 filler-line-163 filler-line-163 filler-line-163
filler-line-164aaaa filler-line-164aaa filler-line-164a filler-line-164a filler-164
"""
