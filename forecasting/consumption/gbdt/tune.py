"""
XGBoost hyperparameter tuning for consumption forecasting.
===========================================================

Uses Optuna to search over the most impactful XGBoost hyperparameters.
One shared hyperparameter set is optimised across all four targets
(the objective is the average MAE on the temporal validation split).

Usage
-----
    pip install optuna
    python -m forecasting.consumption.gbdt.tune
    python -m forecasting.consumption.gbdt.tune --n_trials 100 --csv path/to/train.csv

After tuning, re-run the trainer with the printed best parameters, e.g.:
    python -m forecasting.consumption.gbdt.train \\
        --max_depth 5 --learning_rate 0.008 --subsample 0.75 ...
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_DIR)))))

from forecasting.consumption.constants import TARGET_COLUMNS, LAG_STEPS
from forecasting.consumption.gbdt.train import (
    add_lag_features, _build_features, _TAG,
    TARGET_MIN_KW, TARGET_SEASON,
    _apply_active_season_filter, _apply_dataset_filter_overrides,
)


def parse_args():
    p = argparse.ArgumentParser(description='Tune XGBoost hyperparameters with Optuna.')
    base = os.path.join(os.path.dirname(_DIR), '..', 'data', 'HouseZero', 'processed')
    p.add_argument('--csv',       default=os.path.join(base, 'consumption', 'data_hourly_train.csv'))
    p.add_argument('--n_trials',  type=int,   default=60,
                   help='Number of Optuna trials')
    p.add_argument('--train_frac', type=float, default=0.70)
    p.add_argument('--val_frac',   type=float, default=0.15)
    p.add_argument('--timeout',   type=int,   default=None,
                   help='Stop tuning after this many seconds (optional)')
    return p.parse_args()


def _load_data(csv_path: str) -> pd.DataFrame:
    from forecasting.consumption.constants import GBDT_FEATURE_COLUMNS
    load_cols = ['date'] + TARGET_COLUMNS + GBDT_FEATURE_COLUMNS
    data = pd.read_csv(
        csv_path, parse_dates=['date'], index_col='date',
        usecols=lambda c: c in load_cols,
    ).dropna(subset=TARGET_COLUMNS).ffill().fillna(0)
    return add_lag_features(data, TARGET_COLUMNS, LAG_STEPS)


def _train_eval(data, target, train_frac, val_frac, params, n_estimators=5000,
                early_stopping_rounds=75):
    """Train one XGBoost model and return MAE on the validation slice.

    Applies the same per-target active-season + near-zero row filter that
    train.py uses, so Optuna's objective matches what training will optimize.
    Split indices are recomputed from the filtered length.
    """
    df = _apply_active_season_filter(
        data, target, TARGET_MIN_KW.get(target, 0.0), TARGET_SEASON.get(target),
    )
    X = _build_features(df, target)
    y = df[target]

    n = len(X)
    train_end = int(n * train_frac)
    val_end   = int(n * (train_frac + val_frac))

    sf = MinMaxScaler()
    st = MinMaxScaler()
    Xs = sf.fit_transform(X)
    ys = st.fit_transform(y.values.reshape(-1, 1)).flatten()

    Xtr, ytr = Xs[:train_end],        ys[:train_end]
    Xv,  yv  = Xs[train_end:val_end], ys[train_end:val_end]

    dtrain = xgb.DMatrix(Xtr, label=ytr)
    dval   = xgb.DMatrix(Xv,  label=yv)

    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=n_estimators,
        evals=[(dval, 'val')],
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=False,
    )

    y_pred = st.inverse_transform(
        model.predict(dval).reshape(-1, 1)
    ).flatten()
    y_true = st.inverse_transform(yv.reshape(-1, 1)).flatten()
    return mean_absolute_error(y_true, y_pred), model.best_iteration


def main():
    args = parse_args()

    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print('Optuna not found. Install with:  pip install optuna')
        sys.exit(1)

    _apply_dataset_filter_overrides(args.csv)

    print(f'Loading data from {args.csv}')
    data = _load_data(args.csv)
    n    = len(data)
    print(f'  {n} rows pre-filter  |  splits applied per-target after active-season filter')

    def objective(trial):
        params = {
            'objective':        'reg:squarederror',
            'verbosity':        0,
            'max_depth':        trial.suggest_int('max_depth', 3, 8),
            'learning_rate':    trial.suggest_float('learning_rate', 0.003, 0.05, log=True),
            'subsample':        trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha':        trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
            'reg_lambda':       trial.suggest_float('reg_lambda', 0.1, 10.0, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        }
        maes = []
        for t in TARGET_COLUMNS:
            mae, best_iter = _train_eval(data, t, args.train_frac, args.val_frac, params)
            maes.append(mae)
            trial.set_user_attr(f'{_TAG[t]}_mae', round(mae, 5))
            trial.set_user_attr(f'{_TAG[t]}_best_iter', best_iter)
        return float(np.mean(maes))

    study = optuna.create_study(direction='minimize',
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout,
                   show_progress_bar=True)

    best = study.best_trial
    print(f'\n{"="*60}')
    print(f'Best trial #{best.number}  mean_val_MAE = {best.value:.5f}')
    print(f'{"="*60}')
    print('\nPer-target MAE on validation split:')
    for t in TARGET_COLUMNS:
        tag = _TAG[t]
        print(f'  {tag:<10s}  MAE={best.user_attrs[f"{tag}_mae"]:.5f}  '
              f'best_iter={best.user_attrs[f"{tag}_best_iter"]}')

    print('\nBest hyperparameters:')
    for k, v in best.params.items():
        print(f'  {k:<22s} = {v}')

    # Print a ready-to-run training command
    cmd_parts = ['python -m forecasting.consumption.gbdt.train']
    for k, v in best.params.items():
        cmd_parts.append(f'--{k} {v:.6g}' if isinstance(v, float) else f'--{k} {v}')
    print(f'\nRun with best params:\n  {" ".join(cmd_parts)}')


if __name__ == '__main__':
    main()
