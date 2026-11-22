"""
Train XGBoost models for consumption forecasting.

One model per target: thermal_heating_kw, thermal_cooling_kw, dhw_kw,
electric_demand_kw.  Each model uses all other targets as cross-features.
"""

import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_DIR)))))

from forecasting.constants import (
    TARGET_THERMAL_HEATING, TARGET_THERMAL_COOLING, TARGET_DHW, TARGET_ELECTRIC,
)
from forecasting.consumption.constants import (
    GBDT_FEATURE_COLUMNS, TARGET_COLUMNS, LAG_STEPS, OPTIONAL_GBDT_FEATURE_COLUMNS,
    ACTIVITY_LAG_STEPS,
)

# Per-target XGBoost objective. DHW is zero-inflated (~90% zero, rare ~2 kW
# bursts) — Tweedie with variance_power=1.5 is the standard XGBoost choice
# for this distribution and stops squarederror from rewarding "predict the
# mean everywhere" behaviour. Other targets keep the original objective.
TARGET_OBJECTIVES: dict[str, tuple[str, dict]] = {
    TARGET_THERMAL_HEATING: ('reg:squarederror', {}),
    TARGET_THERMAL_COOLING: ('reg:squarederror', {}),
    TARGET_DHW:             ('reg:tweedie', {'tweedie_variance_power': 1.5}),
    TARGET_ELECTRIC:        ('reg:squarederror', {}),
}

# Per-target near-zero kW threshold. Heating/cooling baseline (TABS slab
# circulator standby + sensor noise) sits at ~0.02-0.04 kW; 0.05 kW is just
# above the noise floor without eating real low-demand hours. DHW/electric
# set to 0 — no filter.
TARGET_MIN_KW: dict[str, float] = {
    TARGET_THERMAL_HEATING: 0.05,
    TARGET_THERMAL_COOLING: 0.05,
    TARGET_DHW:             0.0,
    TARGET_ELECTRIC:        0.0,
}

# Seasonal mask: 1 = keep is_heating_season==1, 0 = keep ==0, None = no filter.
TARGET_SEASON: dict[str, int | None] = {
    TARGET_THERMAL_HEATING: 1,
    TARGET_THERMAL_COOLING: 0,
    TARGET_DHW:             None,
    TARGET_ELECTRIC:        None,
}

# Short file-name tags for each target
_TAG = {
    TARGET_THERMAL_HEATING: 'heating',
    TARGET_THERMAL_COOLING: 'cooling',
    TARGET_DHW:             'dhw',
    TARGET_ELECTRIC:        'electric',
}


def parse_args():
    p = argparse.ArgumentParser()
    base = os.path.join(os.path.dirname(_DIR), '..', 'data', 'HouseZero', 'processed')
    p.add_argument('csv_path', nargs='?',
                   default=os.path.join(base, 'consumption', 'data_hourly_train.csv'))
    p.add_argument('--train_frac',            type=float, default=0.70)
    p.add_argument('--val_frac',              type=float, default=0.15)
    p.add_argument('--model_prefix',          type=str,   default='xgb_consumption')
    p.add_argument('--output_dir',            type=str,   default=_DIR)
    p.add_argument('--n_estimators',          type=int,   default=15000)
    p.add_argument('--max_depth',             type=int,   default=4)
    p.add_argument('--learning_rate',         type=float, default=0.008)
    p.add_argument('--subsample',             type=float, default=0.96)
    p.add_argument('--colsample_bytree',      type=float, default=0.99)
    p.add_argument('--reg_alpha',             type=float, default=0.26)
    p.add_argument('--reg_lambda',            type=float, default=0.23)
    p.add_argument('--min_child_weight',      type=int,   default=6)
    p.add_argument('--early_stopping_rounds', type=int,   default=300)
    p.add_argument('--min_target_kw_heating', type=float,
                   default=TARGET_MIN_KW[TARGET_THERMAL_HEATING],
                   help='Drop in-heating-season rows where thermal_heating_kw < this. '
                        'Default just above the standby/noise floor.')
    p.add_argument('--min_target_kw_cooling', type=float,
                   default=TARGET_MIN_KW[TARGET_THERMAL_COOLING],
                   help='Drop in-cooling-season rows where thermal_cooling_kw < this.')
    p.add_argument('--disable_active_season_filter', action='store_true',
                   help='Skip the heating/cooling season + near-zero row filter '
                        '(train on full year). Use to A/B vs the legacy pipeline.')
    p.add_argument('--seed', type=int, default=None,
                   help='Random seed. Omit to use $FORECAST_SEED when set '
                        '(train_pipeline exports it), else 42.')
    return p.parse_args()


def add_lag_features(data: pd.DataFrame, target_cols: list = TARGET_COLUMNS,
                     lags: tuple = LAG_STEPS) -> pd.DataFrame:
    """
    Append lagged target columns ({target}_lag_{lag}) and drop rows that
    would have NaN lags (first max(lags) rows).

    The optional activity signals are replaced by their lagged values
    ({signal}_lag_{lag}, lags ACTIVITY_LAG_STEPS): the same-hour value is not
    available for a day-ahead forecast.
    """
    df = data.copy()
    for t in target_cols:
        for lag in lags:
            df[f'{t}_lag_{lag}'] = df[t].shift(lag)
    for c in OPTIONAL_GBDT_FEATURE_COLUMNS:
        if c in df.columns:
            for lag in ACTIVITY_LAG_STEPS:
                df[f'{c}_lag_{lag}'] = df[c].shift(lag)
            df = df.drop(columns=[c])
    return df.dropna()


def _build_features(data: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Feature columns: weather/temporal features + lagged target values.
    All contemporaneous target columns are excluded to prevent leakage.
    """
    exclude = set(TARGET_COLUMNS) | {'Unnamed: 0'}
    cols = [c for c in data.columns if c not in exclude]
    return data[cols]


def _apply_active_season_filter(data: pd.DataFrame, target: str,
                                min_kw: float, season: int | None) -> pd.DataFrame:
    """Drop rows outside the target's active season and rows with near-zero
    target value. Operates on the post-lag DataFrame so lag columns stay
    intact (lags are computed from the unfiltered series before this runs).
    Pass-through when both filters are disabled.
    """
    if season is None and min_kw <= 0.0:
        return data
    n0 = len(data)
    mask = pd.Series(True, index=data.index)
    if season is not None and 'is_heating_season' in data.columns:
        mask &= (data['is_heating_season'].astype(int) == int(season))
    if min_kw > 0.0:
        mask &= (data[target].abs() >= min_kw)
    out = data.loc[mask]
    print(f'  [filter:{_TAG[target]}] season={season} min_kw={min_kw} '
          f'kept {len(out)}/{n0} rows ({len(out)/max(n0,1):.1%})')
    return out


def _train_xgb(X_train, y_train, X_val, y_val, args, name, target=None):
    # Resolve per-target objective. Falls back to reg:squarederror for any
    # target not explicitly listed (and for callers that didn't pass `target`).
    obj_name, obj_extra = TARGET_OBJECTIVES.get(
        target, ('reg:squarederror', {})
    )
    params = {
        'objective':        obj_name,
        'max_depth':         args.max_depth,
        'learning_rate':     args.learning_rate,
        'subsample':         args.subsample,
        'colsample_bytree':  args.colsample_bytree,
        'reg_alpha':         args.reg_alpha,
        'reg_lambda':        args.reg_lambda,
        'min_child_weight':  args.min_child_weight,
        'verbosity':         1,
        # XGBoost has no global RNG; subsample / colsample draw from this.
        'seed':              int(getattr(args, 'seed', 0) or 0),
    }
    params.update(obj_extra)
    if obj_name != 'reg:squarederror':
        print(f'  [{name}] objective = {obj_name}'
              + (f'  (extra: {obj_extra})' if obj_extra else ''))
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval   = xgb.DMatrix(X_val,   label=y_val)
    print(f'\n=== Training XGBoost for {name} ===')
    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=args.n_estimators,
        evals=[(dtrain, 'train'), (dval, 'val')],
        early_stopping_rounds=args.early_stopping_rounds,
        verbose_eval=200,
    )
    return model


def _evaluate(model, X_test, y_test, scaler_tgt, name):
    y_pred_s = model.predict(xgb.DMatrix(X_test))
    y_pred   = scaler_tgt.inverse_transform(y_pred_s.reshape(-1, 1)).flatten()
    y_true   = scaler_tgt.inverse_transform(y_test.reshape(-1, 1)).flatten()
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f'  {name}: MAE={mae:.4f}  RMSE={rmse:.4f}')
    return {'mae': mae, 'rmse': rmse}


def _apply_dataset_objective_overrides(csv_path: str) -> None:
    """Let a dataset's ``TRAINING_HYPERPARAMS`` override TARGET_OBJECTIVES.

    Schema (in ``forecasting/data/<NAME>/constants.py``):

        TRAINING_HYPERPARAMS = {
            'gbdt': {
                'dhw_kw': {'objective': 'reg:tweedie',
                           'tweedie_variance_power': 1.5},
            },
        }

    Anything missing falls back to the module-level default in TARGET_OBJECTIVES.
    """
    try:
        from forecasting.consumption.adaptive import load_dataset_training_hyperparams
    except ImportError:
        return
    hp = load_dataset_training_hyperparams(csv_path)
    gbdt_hp = (hp.get('gbdt') or {}) if isinstance(hp, dict) else {}
    if not gbdt_hp:
        return
    applied: list[str] = []
    for target, cfg in gbdt_hp.items():
        if target not in TARGET_OBJECTIVES:
            continue
        if not isinstance(cfg, dict) or 'objective' not in cfg:
            continue
        obj_name = str(cfg['objective'])
        extras   = {k: v for k, v in cfg.items() if k != 'objective'}
        TARGET_OBJECTIVES[target] = (obj_name, extras)
        applied.append(f"{target}={obj_name}")
    if applied:
        print(f'[train] dataset GBDT objective overrides: {", ".join(applied)}')


def _apply_dataset_filter_overrides(csv_path: str) -> None:
    """Pull per-target ``min_target_kw`` from the dataset's TRAINING_HYPERPARAMS
    so heating/cooling thresholds can be tuned per site.

    Schema (in ``forecasting/data/<NAME>/constants.py``):

        TRAINING_HYPERPARAMS = {
            'gbdt': {
                'thermal_heating_kw': {'min_target_kw': 0.05},
                'thermal_cooling_kw': {'min_target_kw': 0.05},
            },
        }
    """
    try:
        from forecasting.consumption.adaptive import load_dataset_training_hyperparams
    except ImportError:
        return
    hp = load_dataset_training_hyperparams(csv_path)
    gbdt_hp = (hp.get('gbdt') or {}) if isinstance(hp, dict) else {}
    applied: list[str] = []
    for t, cfg in gbdt_hp.items():
        if t in TARGET_MIN_KW and isinstance(cfg, dict) and 'min_target_kw' in cfg:
            TARGET_MIN_KW[t] = float(cfg['min_target_kw'])
            applied.append(f'{t}.min_target_kw={TARGET_MIN_KW[t]}')
    if applied:
        print(f'[train] dataset GBDT filter overrides: {", ".join(applied)}')


def _apply_dataset_seasonal_gate(csv_path: str) -> None:
    """Drop the heating-season split for datasets with APPLY_SEASONAL_GATE=False.

    When a dataset sets ``APPLY_SEASONAL_GATE = False`` in its ``constants.py``
    (e.g. HEEW: a desert, cooling-dominated campus where both thermal channels
    run year-round), heating/cooling are trained all-year instead of on
    regime-restricted rows, matching the inference gate being skipped. The
    near-zero ``min_target_kw`` filter is left untouched. Defaults to the
    module-level TARGET_SEASON (heating-season split) for every other dataset.
    """
    try:
        from forecasting.consumption.adaptive import load_dataset_seasonal_gate
    except ImportError:
        return
    if load_dataset_seasonal_gate(csv_path):
        return
    TARGET_SEASON[TARGET_THERMAL_HEATING] = None
    TARGET_SEASON[TARGET_THERMAL_COOLING] = None
    print('[train] APPLY_SEASONAL_GATE=False: training heating/cooling all-year '
          '(no heating-season split).')


def main():
    args = parse_args()
    from forecasting.utils import seed_everything
    args.seed = seed_everything(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    _apply_dataset_objective_overrides(args.csv_path)
    _apply_dataset_filter_overrides(args.csv_path)
    _apply_dataset_seasonal_gate(args.csv_path)

    # Peek at the header to determine which targets this dataset provides.
    # Datasets like HEEW that have no DHW channel must not be in dropna/loop.
    _csv_header = pd.read_csv(args.csv_path, nrows=0).columns.tolist()
    available_targets = [t for t in TARGET_COLUMNS if t in _csv_header]
    if set(available_targets) != set(TARGET_COLUMNS):
        _missing_tgts = [t for t in TARGET_COLUMNS if t not in available_targets]
        print(f'[train] dataset is missing target column(s) {_missing_tgts}; those heads will be skipped.')

    # Load consumption-specific split (has only the relevant columns). The
    # optional activity columns (dhw_volume_m3h, valve_open_frac) are listed
    # too; the lambda silently drops any column that isn't actually in the CSV.
    load_cols = (
        ['date'] + TARGET_COLUMNS + GBDT_FEATURE_COLUMNS
        + list(OPTIONAL_GBDT_FEATURE_COLUMNS)
    )
    data = pd.read_csv(
        args.csv_path, parse_dates=['date'], index_col='date',
        usecols=lambda c: c in load_cols,
    ).dropna(subset=available_targets).ffill().fillna(0)

    print(f'Loaded {len(data)} rows x {len(data.columns)} cols from {args.csv_path}')

    # Add lagged cross-target features (replaces contemporaneous target leakage)
    data = add_lag_features(data, available_targets, LAG_STEPS)
    print(f'After lag features: {len(data)} rows x {len(data.columns)} cols '
          f'(dropped first {max(LAG_STEPS)} rows for NaN lags)')

    pfx = args.model_prefix
    out = args.output_dir

    for target in available_targets:
        tag  = _TAG[target]
        name = tag.replace('_', ' ').title()

        # Per-target active-season + near-zero row filter. Runs after lag
        # features (so lags are computed from the full series) and before
        # the train/val/test split (so split fractions refer to in-season
        # rows). DHW/electric have season=None and min_kw=0 -> pass-through.
        if not args.disable_active_season_filter:
            min_kw = (args.min_target_kw_heating if target == TARGET_THERMAL_HEATING
                      else args.min_target_kw_cooling if target == TARGET_THERMAL_COOLING
                      else TARGET_MIN_KW.get(target, 0.0))
            season = TARGET_SEASON.get(target)
            data_t = _apply_active_season_filter(data, target, min_kw, season)
        else:
            data_t = data

        X = _build_features(data_t, target)
        y = data_t[target]

        n         = len(X)
        train_end = int(n * args.train_frac)
        val_end   = int(n * (args.train_frac + args.val_frac))

        X_raw_tr = X.iloc[:train_end]
        X_raw_v  = X.iloc[train_end:val_end]
        X_raw_te = X.iloc[val_end:]
        y_raw_tr = y.iloc[:train_end].values.reshape(-1, 1)
        y_raw_v  = y.iloc[train_end:val_end].values.reshape(-1, 1)
        y_raw_te = y.iloc[val_end:].values.reshape(-1, 1)

        # Fit scalers on training data only — no leakage from val/test
        sf = MinMaxScaler()
        st = MinMaxScaler()
        Xtr = sf.fit_transform(X_raw_tr)
        ytr = st.fit_transform(y_raw_tr).flatten()
        Xv  = sf.transform(X_raw_v)
        yv  = st.transform(y_raw_v).flatten()
        Xte = sf.transform(X_raw_te)
        yte = st.transform(y_raw_te).flatten()

        model = _train_xgb(Xtr, ytr, Xv, yv, args, name, target=target)
        _evaluate(model, Xte, yte, st, name)

        model.save_model(os.path.join(out, f'{pfx}_{tag}_initial.json'))
        with open(os.path.join(out, f'scaler_features_{tag}_{pfx}_initial.sav'), 'wb') as f:
            pickle.dump(sf, f)
        with open(os.path.join(out, f'scaler_{tag}_{pfx}_initial.sav'), 'wb') as f:
            pickle.dump(st, f)
        print(f'  Saved {pfx}_{tag}_initial.json + scalers')

    print(f'\nAll {len(TARGET_COLUMNS)} models saved to {out}')


if __name__ == '__main__':
    main()
