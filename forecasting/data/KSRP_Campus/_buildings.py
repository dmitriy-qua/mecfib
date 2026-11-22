"""
KSRP Campus — per-building water-loop and load aggregator.

Split out of prepare.py to keep each file under the cowork workspace's
single-write byte budget.

The published archive has a 7-building sub-tree under
``Cooling and heating load data of individual buildings/<building>/``:

    Communication building
    Experimental building
    GYM
    Industrial and educational building
    Measurement building
    Media building
    Research building

Each building's yearly Excel exposes:
    Date, Supply water temperature, Return water temperature, Water flow,
    Load, Cooling load, Heating load.

This module reads all of them, aggregates the per-building columns to the
district level (mean for temperatures, sum for flow and loads), and returns
a single hourly frame.
"""

import os

import pandas as pd

from forecasting.data.KSRP_Campus._io import (
    resolve_folder, list_yearly_files, read_yearly,
    match_columns, sum_matched,
)


BLDG_RULES = {
    'supply_water_temp_c':  [['supply', 'water', 'temperature'], ['supply', 'temperature']],
    'return_water_temp_c':  [['return', 'water', 'temperature'], ['return', 'temperature']],
    'water_flow_m3h':       [['water', 'flow'], ['flow']],
    'bldg_total_load_kw':   [['load']],            # column literally "Load (kW)"
    'bldg_cool_load_kw':    [['cooling', 'load']],
    'bldg_heat_load_kw':    [['heating', 'load']],
}


def load_individual_buildings(raw_dir: str, idx: pd.DatetimeIndex) -> pd.DataFrame:
    folder = resolve_folder(raw_dir, 'buildings')
    if folder is None:
        print('  [buildings] folder not found — district water-loop columns will be NaN.')
        return pd.DataFrame(index=idx)

    bldg_dirs = [os.path.join(folder, d) for d in sorted(os.listdir(folder))
                 if os.path.isdir(os.path.join(folder, d))]
    if not bldg_dirs:
        return pd.DataFrame(index=idx)

    print(f'  [buildings] {len(bldg_dirs)} sub-folders:  '
          + ', '.join(os.path.basename(b) for b in bldg_dirs))

    per_bldg = {}
    for bd in bldg_dirs:
        name = os.path.basename(bd)
        paths = list_yearly_files(bd)
        if not paths:
            continue
        frames = []
        for p in paths:
            try:
                frames.append(read_yearly(p))
            except Exception as e:
                print(f'    [warn] {name}: {os.path.basename(p)}: {e}')
        if not frames:
            continue
        df = pd.concat(frames).sort_index()
        df = df[~df.index.duplicated(keep='first')]
        matches = match_columns(df.columns, BLDG_RULES)
        canon = sum_matched(df, matches, agg='sum')
        canon = canon.resample('h').mean().reindex(idx).ffill(limit=3).bfill(limit=3)
        per_bldg[name] = canon

    if not per_bldg:
        return pd.DataFrame(index=idx)

    # District aggregates: mean for temperatures, sum for flow and loads.
    sup  = pd.concat([d['supply_water_temp_c'] for d in per_bldg.values()], axis=1).mean(axis=1)
    ret  = pd.concat([d['return_water_temp_c'] for d in per_bldg.values()], axis=1).mean(axis=1)
    flow = pd.concat([d['water_flow_m3h']      for d in per_bldg.values()], axis=1).sum(axis=1, min_count=1)
    cool = pd.concat([d['bldg_cool_load_kw']   for d in per_bldg.values()], axis=1).sum(axis=1, min_count=1)
    heat = pd.concat([d['bldg_heat_load_kw']   for d in per_bldg.values()], axis=1).sum(axis=1, min_count=1)
    tot  = pd.concat([d['bldg_total_load_kw']  for d in per_bldg.values()], axis=1).sum(axis=1, min_count=1)

    return pd.DataFrame({
        'district_supply_temp_c':   sup,
        'district_return_temp_c':   ret,
        'district_water_flow_m3h':  flow,
        'bldg_sum_cooling_load_kw': cool,
        'bldg_sum_heating_load_kw': heat,
        'bldg_sum_total_load_kw':   tot,
    }, index=idx)
