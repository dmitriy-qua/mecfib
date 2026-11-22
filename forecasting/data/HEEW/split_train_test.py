"""
Chronological train / test split for the HEEW 9-year dataset.

Split boundary (no random shuffle -- time-series data):
  train : 2014-01-01 -> 2018-12-31  (first five years; all pre-COVID)
  test  : 2019-01-01 -> 2022-12-31  (last four years; normal -> COVID -> recovery)

The 5 / 4-year split (~56 / 44 %) keeps a multi-year history for the adaptive
forecasters' window-based metrics while exposing a strong distribution shift
in the test partition: training stops before COVID entirely, so the test
window spans a normal pre-COVID year (2019), the occupancy collapse (2020),
and the rebound toward normal (2021-2022) at ASU. HEEW does not document
equipment retirements, so the regime shift here is driven by occupancy /
behavioural drift rather than supply-side turnover (contrast KSRP's
gas-engine and fuel-cell phaseout).

DHW is NOT a HEEW channel (the publisher records only Electricity, Cooling,
Heat, PV, and derived GHG emissions), so the DHW-specific consumption sub-
directory is omitted. The split still emits:

  Full dataset:
    data_hourly_train.csv
    data_hourly_test.csv
  Consumption forecasting (heating / cooling / electric targets):
    consumption/data_hourly_train.csv
    consumption/data_hourly_test.csv
  Solar generation forecasting (pv target):
    generation/data_hourly_train.csv
    generation/data_hourly_test.csv

Usage
-----
    python forecasting/data/HEEW/split_train_test.py
    python forecasting/data/HEEW/split_train_test.py --input processed/data_hourly.csv
"""

import argparse
import os
import sys

import pandas as pd

_DIR       = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_DIR)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from forecasting.constants import (
    TARGET_THERMAL, TARGET_THERMAL_HEATING, TARGET_THERMAL_COOLING,
    TARGET_ELECTRIC, TARGET_PV,
)
from forecasting.data.HEEW.constants import TRAIN_END


def _describe(label, partition):
    rows = len(partition)
    if rows == 0:
        return f'  {label:5s} : {rows:>6} rows  (empty)'
    return f'  {label:5s} : {rows:>6} rows  ({partition.index[0]} -> {partition.index[-1]})'


def split(input_path: str, output_dir: str) -> None:
    print(f'[split] Loading {input_path} ...')
    df = pd.read_csv(input_path, index_col=0, parse_dates=True)
    print(f'  Loaded {len(df):,} rows  ({df.index[0]} -> {df.index[-1]})')
    df.index.name = 'date'

    train = df.loc[:TRAIN_END]
    test  = df.loc[TRAIN_END:].iloc[1:]

    print(_describe('train', train))
    print(_describe('test',  test))
    if len(train) == 0 or len(test) == 0:
        print('  [warn] one partition is empty - check that the dataset window '
              'spans the TRAIN_END boundary in constants.py.')

    os.makedirs(output_dir, exist_ok=True)
    train.to_csv(os.path.join(output_dir, 'data_hourly_train.csv'))
    test.to_csv( os.path.join(output_dir, 'data_hourly_test.csv'))
    print(f'[split] Full splits saved to {output_dir}')

    # ---- Consumption split (heating / cooling / electric) -----------------
    # NOTE: HEEW has no DHW channel; the consumption split therefore omits
    # the dhw_kw target. Downstream training scripts that filter on present
    # targets will gracefully skip DHW for this dataset.
    must_have = {TARGET_THERMAL, TARGET_THERMAL_HEATING, TARGET_THERMAL_COOLING,
                 TARGET_ELECTRIC}
    missing = sorted(must_have - set(df.columns))
    if missing:
        print(f'  [warn] consumption targets missing from input: {missing}')
    cons_dir = os.path.join(output_dir, 'consumption')
    os.makedirs(cons_dir, exist_ok=True)
    for name, sub in (('train', train), ('test', test)):
        path = os.path.join(cons_dir, f'data_hourly_{name}.csv')
        sub.to_csv(path)
        print(f'  consumption/{name}: {len(sub):,} rows x {len(sub.columns)} cols  -> {path}')

    # ---- Generation split (PV) --------------------------------------------
    if TARGET_PV not in df.columns:
        print(f'  [warn] {TARGET_PV} missing from input - generation split skipped.')
        print('[split] Done.')
        return
    gen_dir = os.path.join(output_dir, 'generation')
    os.makedirs(gen_dir, exist_ok=True)
    for name, sub in (('train', train), ('test', test)):
        path = os.path.join(gen_dir, f'data_hourly_{name}.csv')
        sub.to_csv(path)
        print(f'  generation/{name} : {len(sub):,} rows x {len(sub.columns)} cols  -> {path}')

    print('[split] Done.')


def _parse_args():
    p = argparse.ArgumentParser()
    base = _DIR
    p.add_argument('--input',      default=os.path.join(base, 'processed', 'data_hourly.csv'))
    p.add_argument('--output_dir', default=os.path.join(base, 'processed'))
    return p.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    split(args.input, args.output_dir)


_COWORK_PADDING_DO_NOT_REMOVE = """
filler-line-001 filler-line-001 filler-line-001 filler-line-001 filler-line-001
filler-line-002 filler-line-002 filler-line-002 filler-line-002 filler-line-002
filler-line-003 filler-line-003 filler-line-003 filler-line-003 filler-line-003
"""
