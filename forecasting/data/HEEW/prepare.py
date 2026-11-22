"""
Data preparation pipeline for the HEEW (Hierarchical Energy, Emissions, and
Weather) 9-year community-aggregated dataset.

Source publication:
  H. Dong et al., "A hierarchical dataset on multiple energy consumption and
  PV generation with emissions and weather information," Scientific Data
  12:1727 (2025). DOI: 10.1038/s41597-025-06010-8.

Default community: CN003 (Arizona State University - Tempe Campus).

The HEEW raw layout is much simpler than KSRP's per-yearly-Excel archive,
so this pipeline is correspondingly leaner:

  CN003_energy.csv   ->  Electricity (kW), PV (kW), Cooling (Ton),
                         Heat (mmBTU/hr), Emission (Ton), Total Energy (mmBTU/hr)
  Total_weather.csv  ->  National Weather Service local readings (Imperial)
  weather.csv        ->  open-meteo bulk weather (metric, with hourly + daily blocks)
  aqi.csv            ->  daily AQI (pm25, pm10, o3, no2, so2, co)

Unit conversions (per Methods Sec. "Refined data estimation"):
  Cooling : Ton          -> kW    (x 3.51685)
  Heat    : mmBTU/hr     -> kW    (/ 0.003412  ~= x 293.07)
  Electric: kW           -> kW    (no change)
  PV      : kW           -> kW    (no change)
  Pressure: inHg         -> hPa   (x 33.8639)
  Wind    : mph          -> m/s   (x 0.44704)
  Temp/dp : F            -> C     ((F - 32) * 5/9)
  Precip  : in           -> mm    (x 25.4)

Usage
-----
    python forecasting/data/HEEW/prepare.py
    python forecasting/data/HEEW/prepare.py --raw_dir raw --out processed/data_hourly.csv
    python forecasting/data/HEEW/prepare.py --energy_csv raw/CN001_energy.csv

If you see ``ImportError`` or 'source code string cannot contain null bytes',
delete any stale ``__pycache__`` next to this file:
    rm -rf forecasting/data/HEEW/__pycache__
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
    TARGET_ELECTRIC, TARGET_PV,
)
from forecasting.utils import get_season, is_heating_season
from forecasting.data.HEEW.constants import (
    DATASET_START, DATASET_END, OUTAGE_WINDOWS, PV_OP_START, PV_OP_END,
    TON_TO_KW, MMBTU_HR_TO_KW, KW_TO_MMBTU_HR,
    F_TO_C_OFFSET, F_TO_C_SCALE, MPH_TO_MS, INHG_TO_HPA, INCH_TO_MM,
)
from forecasting.data.HEEW._io import (
    load_energy_csv, load_nws_weather_csv, reindex_hourly,
)
from forecasting.data.HEEW._weather import (
    load_open_meteo, load_aqi, default_weather_paths,
)

warnings.filterwarnings('ignore')


def _make_hourly_index(start, end):
    return pd.date_range(start=start, end=end + ' 23:00', freq='h', tz=None)


def _apply_outage_mask(out):
    if not OUTAGE_WINDOWS:
        return out
    keep = pd.Series(True, index=out.index)
    for s, e in OUTAGE_WINDOWS:
        keep &= ~((out.index >= s) & (out.index <= e))
    dropped = (~keep).sum()
    if dropped:
        print(f'  [outage] dropping {dropped} rows in {OUTAGE_WINDOWS}')
    return out[keep]


def _materialise_energy(energy_raw: pd.DataFrame, idx: pd.DatetimeIndex) -> pd.DataFrame:
    """Convert raw HEEW energy columns to canonical kW targets on ``idx``.

    Source units (HEEW Table 2):
        Electricity  : kW
        PV           : kW
        Cooling      : Ton (refrigeration)
        Heat         : mmBTU/hr (per Methods, "interpreted as mmBTU/hr by
                                 assuming a 1-hour time resolution")
        Emission     : Ton (CO2-eq, per hour)
        Total Energy : mmBTU/hr (sum of electric + cooling + heat, in mmBTU)
    """
    if energy_raw.empty:
        return pd.DataFrame(index=idx)

    e = reindex_hourly(energy_raw, idx, ffill_limit=3)
    out = pd.DataFrame(index=idx)

    # ---- Electricity (kW direct) -----------------------------------------
    if 'Electricity' in e.columns:
        out[TARGET_ELECTRIC] = pd.to_numeric(
            e['Electricity'], errors='coerce').clip(lower=0)

    # ---- Cooling (Ton -> kW) ---------------------------------------------
    if 'Cooling' in e.columns:
        cooling_kw = pd.to_numeric(e['Cooling'], errors='coerce') * TON_TO_KW
        out[TARGET_THERMAL_COOLING] = cooling_kw.clip(lower=0)

    # ---- Heat (mmBTU/hr -> kW) -------------------------------------------
    if 'Heat' in e.columns:
        heat_kw = pd.to_numeric(e['Heat'], errors='coerce') * MMBTU_HR_TO_KW
        out[TARGET_THERMAL_HEATING] = heat_kw.clip(lower=0)

    # ---- Combined thermal target -----------------------------------------
    out[TARGET_THERMAL] = (
        out.get(TARGET_THERMAL_HEATING, 0).fillna(0)
        + out.get(TARGET_THERMAL_COOLING, 0).fillna(0)
    ).clip(lower=0)

    # ---- PV (kW direct) ---------------------------------------------------
    if 'PV' in e.columns:
        pv = pd.to_numeric(e['PV'], errors='coerce').clip(lower=0)
        pv = pv.where((pv.index >= PV_OP_START) & (pv.index <= PV_OP_END), 0.0)
        out[TARGET_PV] = pv

    # ---- Auxiliary diagnostics (NOT targets) ------------------------------
    # Preserve GHG emissions and the publisher's Total Energy aggregate as
    # auxiliary features. These are not forecast targets in the existing
    # consumption / generation pipelines but are useful for cross-checks.
    if 'Emission' in e.columns:
        out['emission_ton'] = pd.to_numeric(e['Emission'], errors='coerce')
    if 'Total Energy' in e.columns:
        out['total_energy_mmbtu_hr'] = pd.to_numeric(e['Total Energy'], errors='coerce')

    return out


def _materialise_nws_local(nws_raw: pd.DataFrame, idx: pd.DatetimeIndex) -> pd.DataFrame:
    """Convert HEEW Total_weather.csv (Imperial) into canonical ``local_*`` features.

    Mirrors the KSRP convention so feature-list TXT files referencing
    ``local_air_temp_c``, ``local_rh_pct`` etc. transfer across datasets.
    """
    if nws_raw is None or nws_raw.empty:
        return pd.DataFrame(index=idx)

    n = reindex_hourly(nws_raw, idx, ffill_limit=3)
    out = pd.DataFrame(index=idx)

    if 'Temperature' in n.columns:
        out['local_air_temp_c'] = (
            pd.to_numeric(n['Temperature'], errors='coerce') - F_TO_C_OFFSET
        ) * F_TO_C_SCALE
    if 'Dew Point' in n.columns:
        out['local_dew_point_c'] = (
            pd.to_numeric(n['Dew Point'], errors='coerce') - F_TO_C_OFFSET
        ) * F_TO_C_SCALE
    if 'Humidity' in n.columns:
        out['local_rh_pct'] = pd.to_numeric(n['Humidity'], errors='coerce')
    if 'Wind Speed' in n.columns:
        out['local_wind_speed_ms'] = (
            pd.to_numeric(n['Wind Speed'], errors='coerce') * MPH_TO_MS
        )
    if 'Wind Gust' in n.columns:
        out['local_wind_gust_ms'] = (
            pd.to_numeric(n['Wind Gust'], errors='coerce') * MPH_TO_MS
        )
    if 'Pressure' in n.columns:
        out['local_pressure_hpa'] = (
            pd.to_numeric(n['Pressure'], errors='coerce') * INHG_TO_HPA
        )
    if 'Precip' in n.columns:
        out['local_precip_mm'] = (
            pd.to_numeric(n['Precip'], errors='coerce') * INCH_TO_MM
        )

    return out


def prepare_dataset(raw_dir, energy_csv=None, nws_csv=None,
                    weather_csvs=None, aqi_csv=None, output_path=None):
    print('[prepare] Building hourly index ...')
    idx = _make_hourly_index(DATASET_START, DATASET_END)
    print(f'  hourly grid: {idx[0]} ... {idx[-1]}  ({len(idx):,} rows)')

    # ---- Energy & local NWS --------------------------------------------
    energy_csv = energy_csv or os.path.join(raw_dir, 'CN03_energy.csv')
    nws_csv    = nws_csv    or os.path.join(raw_dir, 'Total_weather.csv')

    print('[prepare] Reading community energy CSV ...')
    energy_raw = load_energy_csv(energy_csv)

    print('[prepare] Reading NWS local weather CSV ...')
    nws_raw = load_nws_weather_csv(nws_csv)

    energy_h = _materialise_energy(energy_raw, idx)
    nws_h    = _materialise_nws_local(nws_raw, idx)

    out = pd.DataFrame(index=idx)
    for col in energy_h.columns:
        out[col] = energy_h[col]

    # ---- Calendar / cyclic features -----------------------------------
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

    # ---- Local NWS weather --------------------------------------------
    for col in nws_h.columns:
        out[col] = nws_h[col]

    # ---- Open-meteo bulk weather --------------------------------------
    print('[prepare] Joining open-meteo weather ...')
    om = load_open_meteo(weather_csvs or [], idx)
    for col in om.columns:
        out[col] = om[col]

    # ---- Solar obstruction derived feature -----------------------------
    if 'is_day' not in out.columns or out['is_day'].isna().all():
        if 'direct_radiation' in out.columns:
            out['is_day'] = (out['direct_radiation'].fillna(0) > 5).astype(int)
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

    # ---- AQI ----------------------------------------------------------
    print('[prepare] Joining AQI data ...')
    aqi = load_aqi(aqi_csv or '', idx)
    for col in aqi.columns:
        out[col] = aqi[col]

    # ---- Outage mask + short-gap fill ----------------------------------
    print('[prepare] Applying outage mask + filling short gaps ...')
    out = _apply_outage_mask(out)

    numeric_cols = out.select_dtypes(include=[np.number]).columns.tolist()
    out[numeric_cols] = out[numeric_cols].ffill(limit=3).bfill(limit=3)

    # Drop rows only when EVERY consumption target is missing (rare, since
    # HEEW raw data is pre-cleaned via TCCA + CRAC).
    targets_present = [c for c in (TARGET_THERMAL_HEATING, TARGET_THERMAL_COOLING,
                                   TARGET_ELECTRIC) if c in out.columns]
    before = len(out)
    if targets_present:
        out = out.dropna(subset=targets_present, how='all')
    dropped = before - len(out)
    if dropped:
        print(f'  [info] Dropped {dropped} rows with ALL target values missing.')

    # ---- Sanity ratios -------------------------------------------------
    if TARGET_ELECTRIC in out.columns:
        total_e = out[TARGET_ELECTRIC].clip(lower=0).sum()
        total_t = out.get(TARGET_THERMAL, pd.Series(0, index=out.index)).sum()
        if total_e > 0:
            print(f'  [sanity] thermal/electric over 9 y = {total_t / total_e:.3f}')
    if TARGET_PV in out.columns:
        pv_win = out.loc[(out.index >= PV_OP_START) & (out.index <= PV_OP_END), TARGET_PV]
        if not pv_win.empty:
            print(f'  [sanity] PV peak = {pv_win.max():.1f} kW')
    if 'total_energy_mmbtu_hr' in out.columns and TARGET_ELECTRIC in out.columns:
        # Cross-check the publisher's Total Energy column against our
        # reconstructed sum-of-channels (electricity in kW -> mmBTU/hr +
        # cooling kW -> mmBTU/hr + heat already mmBTU/hr).
        recon = (
            out[TARGET_ELECTRIC].fillna(0) * KW_TO_MMBTU_HR
            + out.get(TARGET_THERMAL_COOLING, 0).fillna(0) * KW_TO_MMBTU_HR
            + out.get(TARGET_THERMAL_HEATING, 0).fillna(0) * KW_TO_MMBTU_HR
        )
        diff = (recon - out['total_energy_mmbtu_hr']).abs().mean()
        print(f'  [sanity] |reconstructed - Total Energy| mean abs diff = {diff:.4f} mmBTU/hr')

    print(f'[prepare] Final dataset: {len(out):,} rows x {len(out.columns)} columns.')
    print(f'  Date range: {out.index[0]}  to  {out.index[-1]}')

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        out.index.name = 'date'
        out.to_csv(output_path)
        print(f'[prepare] Saved to {output_path}')

    return out


def _parse_args():
    p = argparse.ArgumentParser(description='Prepare HEEW dataset for forecasting.')
    base = _DIR
    p.add_argument('--raw_dir',    default=os.path.join(base, 'raw'))
    p.add_argument('--energy_csv', default=None,
                   help='Community energy CSV. Default: <raw_dir>/CN03_energy.csv')
    p.add_argument('--nws_csv',    default=None,
                   help='NWS local-weather CSV. Default: <raw_dir>/Total_weather.csv')
    p.add_argument('--weather_csv', default=None, action='append',
                   help='Open-meteo CSV. Pass multiple times. Default: auto-discover.')
    p.add_argument('--aqi_csv',    default=None,
                   help='AQI CSV. Default: <raw_dir>/aqi.csv if present.')
    p.add_argument('--out',        default=os.path.join(base, 'processed', 'data_hourly.csv'))
    return p.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    weather_csvs = args.weather_csv or default_weather_paths(args.raw_dir)
    aqi_csv = args.aqi_csv or os.path.join(args.raw_dir, 'aqi.csv')
    prepare_dataset(
        raw_dir=args.raw_dir,
        energy_csv=args.energy_csv,
        nws_csv=args.nws_csv,
        weather_csvs=weather_csvs,
        aqi_csv=aqi_csv if os.path.isfile(aqi_csv) else None,
        output_path=args.out,
    )


# ===========================================================================
# COWORK MOUNT PADDING - do not delete (see KSRP_Campus/prepare.py footer).
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
"""
