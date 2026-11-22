"""Evaluate trained XGBoost consumption models (4 targets)."""

import argparse
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
import xgboost as xgb

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_DIR)))))

from forecasting.utils import calculate_metrics, calculate_mase, weekly_naive_calendar
from forecasting.consumption.constants import (
    GBDT_FEATURE_COLUMNS, OPTIONAL_GBDT_FEATURE_COLUMNS, TARGET_COLUMNS,
)
from forecasting.consumption.gbdt.train import (
    add_lag_features, _build_features, _TAG,
    TARGET_MIN_KW, TARGET_SEASON,
    _apply_active_season_filter, _apply_dataset_filter_overrides,
    _apply_dataset_seasonal_gate,
)


def evaluate(csv_path, model_prefix, model_dir=None,
             target: str | None = None,
             disable_active_season_filter: bool = False) -> dict:
    """Evaluate the GBDT consumption ensemble.

    When ``target`` is None (default), evaluates all four heads
    (heating/cooling/dhw/electric). Pass a single target column name to
    evaluate just that head — mirrors the generation gbdt evaluator's
    single-head mode.
    """
    if model_dir is None:
        model_dir = _DIR

    # Peek at the header to determine which targets this dataset provides, so
    # datasets like HEEW that have no DHW channel don't force dhw_kw into
    # dropna / lag features / the eval loop. Mirrors train.py's resolution.
    _csv_header = pd.read_csv(csv_path, nrows=0).columns.tolist()
    available_targets = [t for t in TARGET_COLUMNS if t in _csv_header]

    if target is not None and target not in available_targets:
        raise ValueError(
            f'target "{target}" is not one of this dataset\'s trained consumption '
            f'heads {available_targets}.'
        )
    targets = list(available_targets) if target is None else [target]

    load_cols = (
        ['date'] + TARGET_COLUMNS + GBDT_FEATURE_COLUMNS
        + list(OPTIONAL_GBDT_FEATURE_COLUMNS)
    )
    data = pd.read_csv(
        csv_path, parse_dates=['date'], index_col='date',
        usecols=lambda c: c in load_cols,
    ).dropna(subset=available_targets).ffill().fillna(0)

    data = add_lag_features(data, available_targets)

    _apply_dataset_filter_overrides(csv_path)
    _apply_dataset_seasonal_gate(csv_path)

    pfx = model_prefix
    metrics_out: dict = {}
    for target in targets:
        tag  = _TAG[target]
        name = tag.replace('_', ' ').title()

        # Same filter as training so metrics reflect active-season performance.
        if not disable_active_season_filter:
            data_t = _apply_active_season_filter(
                data, target, TARGET_MIN_KW.get(target, 0.0), TARGET_SEASON.get(target),
            )
        else:
            data_t = data

        m = xgb.Booster()
        m.load_model(os.path.join(model_dir, f'{pfx}_{tag}_initial.json'))
        with open(os.path.join(model_dir, f'scaler_features_{tag}_{pfx}_initial.sav'), 'rb') as f:
            sf = pickle.load(f)
        with open(os.path.join(model_dir, f'scaler_{tag}_{pfx}_initial.sav'), 'rb') as f:
            st = pickle.load(f)

        X = sf.transform(_build_features(data_t, target))
        y_pred = st.inverse_transform(m.predict(xgb.DMatrix(X)).reshape(-1, 1)).flatten()
        y_true = data_t[target].values

        # Weekly-naive baseline ŷ(τ) = y(τ - 168 h) by *calendar* time, so the
        # active-season filter dropping rows does not corrupt the lag (a plain
        # 168-row shift on the filtered frame would). Rows with no observation
        # one week earlier (e.g. the first active week) are NaN and masked out.
        y_naive = weekly_naive_calendar(data_t[target]).to_numpy()
        valid   = ~np.isnan(y_naive)
        mase    = calculate_mase(y_true[valid], y_pred[valid], y_naive[valid])

        m_res = calculate_metrics(y_true, y_pred)
        metrics_out[tag] = {
            'mae':       float(m_res['mae']),
            'rmse':      float(m_res['rmse']),
            'cv_rmse':   float(m_res['cv_rmse']),
            'mase':      float(mase),
            'n_samples': int(len(y_true)),
        }
        print(f'{name} ({target}):  MAE={m_res["mae"]:.4f}  RMSE={m_res["rmse"]:.4f}  '
              f'CV(RMSE)={m_res["cv_rmse"]:.2f}%  MASE={mase:.4f}')

    metrics_path = os.path.join(model_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics_out, f, indent=2)
    print(f'\nSaved metrics -> {metrics_path}')
    return metrics_out


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    base = os.path.join(os.path.dirname(_DIR), '..', 'data', 'HouseZero', 'processed')
    p.add_argument('--csv',       default=os.path.join(base, 'consumption', 'data_hourly_test.csv'))
    p.add_argument('--prefix',    default='xgb_consumption')
    p.add_argument('--model_dir', default=_DIR)
    p.add_argument('--target',    default=None,
                   help='Optional single target column to evaluate '
                        '(e.g. thermal_heating_kw). Default: all 4 heads.')
    p.add_argument('--disable_active_season_filter', action='store_true',
                   help='Skip the heating/cooling season + near-zero row filter '
                        '(evaluate on full year). Use to A/B vs the legacy pipeline.')
    args = p.parse_args()
    evaluate(args.csv, args.prefix, args.model_dir, target=args.target,
             disable_active_season_filter=args.disable_active_season_filter)
