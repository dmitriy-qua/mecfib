"""
Chronological train / test split for NEST UMAR forecasting.

Split boundaries (no random shuffle -- time-series data):
  train : 2019-07-01 -> 2021-06-30  (~17 568 rows, 2 years)
  test  : 2021-07-01 -> 2023-06-30  (~17 520 rows, 2 years)

50 / 50 split by 2-year window so both splits cover every season twice.
No separate validation file is produced. Early stopping inside the LSTM-CNN
and GBDT training scripts uses an internal random split of the training data.

UMAR has no on-site generation (PV / solar-thermal), so only consumption
splits are produced; the generation/ folder from HouseZero is omitted.

Outputs (all written to --output_dir):
  Full dataset:
    data_hourly_train.csv
    data_hourly_test.csv

  Consumption forecasting (4 targets: heating / cooling / dhw / electric):
    consumption/data_hourly_train.csv
    consumption/data_hourly_test.csv

Usage
-----
    python forecasting/data/NEST_UMAR/split_train_test.py
    python forecasting/data/NEST_UMAR/split_train_test.py --input processed/data_hourly.csv
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
    TARGET_DHW, TARGET_ELECTRIC,
)

TRAIN_END = '2021-06-30 23:00'


def split(input_path: str, output_dir: str) -> None:
    print(f'[split] Loading {input_path} ...')
    df = pd.read_csv(input_path, index_col=0, parse_dates=True)
    print(f'  Loaded {len(df)} rows  ({df.index[0]} -> {df.index[-1]})')

    df.index.name = 'date'

    train = df.loc[:TRAIN_END]
    test  = df.loc[TRAIN_END:].iloc[1:]   # exclude shared boundary row

    print(f'  train : {len(train):>6} rows  ({train.index[0]} -> {train.index[-1]})')
    print(f'  test  : {len(test):>6} rows  ({test.index[0]} -> {test.index[-1]})')

    # ---- save full splits --------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)
    train.to_csv(os.path.join(output_dir, 'data_hourly_train.csv'))
    test.to_csv( os.path.join(output_dir, 'data_hourly_test.csv'))
    print(f'[split] Full splits saved to {output_dir}')

    # ---- consumption-specific splits ---------------------------------------
    # Write every prepared column through to the consumption CSVs so the
    # training-screen feature selector can expose per-room sensors (temp_*,
    # setp_*, window_*, y*_*) alongside the model's default whitelist. The
    # LSTM/GBDT pipelines enforce LSTM_COLUMNS at training time, so extra
    # columns sitting in the CSV are harmless. Sanity-check that the must-have
    # target columns are present and warn if any are missing.
    print('[split] Building consumption-specific splits ...')
    must_have = {TARGET_THERMAL, TARGET_THERMAL_HEATING, TARGET_THERMAL_COOLING,
                 TARGET_DHW, TARGET_ELECTRIC}
    missing_targets = sorted(must_have - set(df.columns))
    if missing_targets:
        print(f'  [warn] consumption targets missing from input: {missing_targets}')
    cons_dir = os.path.join(output_dir, 'consumption')
    os.makedirs(cons_dir, exist_ok=True)

    for split_name, split_df in [('train', train), ('test', test)]:
        out_path = os.path.join(cons_dir, f'data_hourly_{split_name}.csv')
        split_df.to_csv(out_path)
        print(f'  consumption/{split_name}: {len(split_df)} rows x {len(split_df.columns)} cols  -> {out_path}')

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
