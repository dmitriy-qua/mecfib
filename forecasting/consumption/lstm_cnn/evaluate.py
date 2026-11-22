"""Evaluate per-target LSTM-CNN consumption models."""

import argparse
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
import torch

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_DIR)))))

from forecasting.constants import FORECAST_HORIZON
from forecasting.utils import (
    create_sequences, calculate_metrics, calculate_mase, weekly_naive_from_window,
)
from forecasting.consumption.constants import LSTM_COLUMNS, TARGET_COLUMNS, SEQUENCE_LENGTH
from forecasting.consumption.lstm_cnn.model import LSTM_CNN_Model, POOL_MODE_ATTENTION
from forecasting.consumption.lstm_cnn.train import _TAG, _IS_HEATING_IDX, _SEASON_FILTER


def evaluate(csv_path: str, model_prefix: str, model_dir: str) -> dict:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    with open(os.path.join(model_dir, f'scaler_{model_prefix}_initial.sav'), 'rb') as f:
        scaler = pickle.load(f)

    # Prefer training-time window dimensions if persisted in _arch.json; fall
    # back to the module-level constants for legacy runs that pre-date them.
    arch_path = os.path.join(model_dir, f'{model_prefix}_arch.json')
    arch_meta: dict = {}
    if os.path.exists(arch_path):
        with open(arch_path) as _f:
            arch_meta = json.load(_f)
    seq_len = int(arch_meta.get('sequence_length',  SEQUENCE_LENGTH))
    horizon = int(arch_meta.get('forecast_horizon', FORECAST_HORIZON))

    # Drive the feature list off the persisted arch.json when present (so
    # training-time --features overrides survive). Fall back to LSTM_COLUMNS
    # for legacy runs that pre-date the persisted `features` key.
    cols = list(arch_meta.get('features') or LSTM_COLUMNS)

    # Which target heads to evaluate. Prefer the `targets` persisted by train.py
    # (a subset of TARGET_COLUMNS when the run trained fewer heads, e.g. a
    # dataset like HEEW with no dhw_kw channel). Fall back to TARGET_COLUMNS for
    # legacy runs that pre-date the persisted key, and intersect with `cols` so
    # a target the dataset never provided can't slip into dropna / the eval loop.
    eval_targets = [t for t in (arch_meta.get('targets') or TARGET_COLUMNS) if t in cols]

    # Which target columns were log1p-transformed at training time. The scaler
    # was fit in log space for these, so the inverse path must expm1 after
    # inverse_transform. Absent for pre-log1p runs.
    log1p_targets = set(arch_meta.get('log1p_targets') or [])

    data_raw = pd.read_csv(
        csv_path, parse_dates=['date'], index_col='date',
        usecols=['date'] + cols,
    ).dropna(subset=eval_targets).ffill().fillna(0)

    # Local indices into the resolved feature stack.
    hs_idx = cols.index('is_heating_season')

    # Mirror training: datasets with APPLY_SEASONAL_GATE=False (e.g. HEEW) train
    # heating/cooling all-year, so evaluate them all-year too rather than masking
    # to the heating-season split.
    try:
        from forecasting.consumption.adaptive import load_dataset_seasonal_gate
        _seasonal_gate = load_dataset_seasonal_gate(csv_path)
    except ImportError:
        _seasonal_gate = True
    season_filter = dict(_SEASON_FILTER)
    if not _seasonal_gate:
        season_filter['thermal_heating_kw'] = None
        season_filter['thermal_cooling_kw'] = None

    # Apply the same log1p transform on target columns the scaler was fit on.
    data = data_raw.copy()
    for t in log1p_targets:
        if t in data.columns:
            data[t] = np.log1p(np.clip(data[t].values, a_min=0.0, a_max=None))

    data_scaled  = scaler.transform(data[cols])
    X_all, y_all = create_sequences(data_scaled, seq_len, horizon)

    n_cols = len(cols)
    metrics_out: dict = {}

    for target in eval_targets:
        tag    = _TAG[target]
        t_idx  = cols.index(target)
        season = season_filter[target]

        if season is True:
            mask = y_all[:, 0, hs_idx] >= 0.5
        elif season is False:
            mask = y_all[:, 0, hs_idx] < 0.5
        else:
            mask = np.ones(len(y_all), dtype=bool)

        X_s = X_all[mask]
        y_s = y_all[mask, :, t_idx]   # (n, horizon) — scaled

        # Infer architecture from the checkpoint so the model always matches
        # the saved weights regardless of model.py defaults.
        pth_path = os.path.join(model_dir, f'{model_prefix}_{tag}_initial.pth')
        sd = torch.load(pth_path, map_location=device, weights_only=False)
        branch_keys  = sorted(k for k in sd if k.startswith('conv_branches.') and k.endswith('.weight'))
        cnn_channels = sd[branch_keys[0]].shape[0]
        kernel_sizes = [sd[k].shape[2] for k in branch_keys]
        hidden_dim   = sd['lstm.weight_hh_l0'].shape[1]   # hh: (4*H, H)
        num_layers   = sum(1 for k in sd if k.startswith('lstm.weight_ih_l'))
        # arch_meta was already loaded above; reuse the dropout entry.
        dropout = arch_meta.get('dropout', 0.2)
        # Pool mode: prefer arch.json, fall back to inferring from FC input dim.
        # fc.weight: (output, fc_in); fc_in == hidden_dim -> attention only;
        # fc_in == 3*hidden_dim -> attention+max+last concat head.
        pool_mode = arch_meta.get('pool_mode')
        if not pool_mode:
            fc_in = sd['fc.weight'].shape[1]
            pool_mode = 'attention_max_last' if fc_in == 3 * hidden_dim else POOL_MODE_ATTENTION

        model = LSTM_CNN_Model(
            X_s.shape[2], horizon, num_features=1,
            hidden_dim=hidden_dim, num_layers=num_layers,
            cnn_channels=cnn_channels, kernel_sizes=kernel_sizes,
            dropout=dropout, pool_mode=pool_mode,
        ).to(device)
        model.load_state_dict(sd)
        model.eval()

        batch_size = 256
        preds_chunks = []
        X_tensor = torch.tensor(X_s, dtype=torch.float32)
        with torch.no_grad():
            for start in range(0, len(X_tensor), batch_size):
                chunk = X_tensor[start:start + batch_size].to(device)
                preds_chunks.append(model(chunk).cpu().numpy())
        preds = np.concatenate(preds_chunks, axis=0)[:, :, 0]   # (n, horizon)
        if device == 'cuda':
            torch.cuda.empty_cache()

        # Inverse-transform via shared scaler: embed the single channel into a full-width array
        n_samples, horizon = y_s.shape
        n_flat = n_samples * horizon

        dummy_true = np.zeros((n_flat, n_cols))
        dummy_pred = np.zeros((n_flat, n_cols))
        dummy_true[:, t_idx] = y_s.flatten()
        dummy_pred[:, t_idx] = preds.flatten()

        # Inverse-transform: reshape back to (n_samples, horizon) for per-horizon slicing
        y_inv = scaler.inverse_transform(dummy_true)[:, t_idx].reshape(n_samples, horizon)
        p_inv = scaler.inverse_transform(dummy_pred)[:, t_idx].reshape(n_samples, horizon)

        # Weekly-naive baseline per sequence: ŷ(t+h) = y(t+h - 168). Because the
        # context window (seq_len=336) immediately precedes the forecast window,
        # the naive value for each step lives inside the context at offset
        # seq_len-168.. ; weekly_naive_from_window slices it for the full horizon.
        naive_scaled = weekly_naive_from_window(X_s, t_idx, seq_len, horizon)
        naive_steps  = naive_scaled.shape[1]
        dummy_naive  = np.zeros((n_samples * naive_steps, n_cols))
        dummy_naive[:, t_idx] = naive_scaled.flatten()
        if naive_steps:
            naive_inv = scaler.inverse_transform(dummy_naive)[:, t_idx].reshape(n_samples, naive_steps)
        else:      # context shorter than one week: no weekly-naive reference, MASE is nan
            naive_inv = np.zeros((n_samples, 0))

        # Undo the log1p that was applied before scaling, if this target used it.
        # Clamp to 0 first so expm1 of a tiny negative numerical-noise value
        # doesn't produce a meaningless slightly-negative kW.
        if target in log1p_targets:
            y_inv     = np.expm1(np.clip(y_inv,     a_min=0.0, a_max=None))
            p_inv     = np.expm1(np.clip(p_inv,     a_min=0.0, a_max=None))
            naive_inv = np.expm1(np.clip(naive_inv, a_min=0.0, a_max=None))

        # MASE over the steps the weekly-naive context can cover (the full horizon
        # whenever seq_len >= 168, which holds for the default 336). For legacy
        # sub-week contexts (seq_len < 168) the weekly lag is unavailable in-window
        # and MASE is left nan — use forecasting.baselines.run for the floor.
        if naive_steps == 0:
            mase_overall = float('nan')
        else:
            mase_overall = calculate_mase(
                y_inv[:, :naive_steps].flatten(),
                p_inv[:, :naive_steps].flatten(),
                naive_inv.flatten(),
            )

        overall = calculate_metrics(y_inv.flatten(), p_inv.flatten())
        metrics_out[tag] = {
            'mae':     float(overall['mae']),
            'rmse':    float(overall['rmse']),
            'cv_rmse': float(overall['cv_rmse']),
            'mase':    float(mase_overall),
            'n_samples': int(n_samples),
            'horizon':   int(horizon),
        }
        print(f'\n{tag}')
        print(f'  overall  :  MAE={overall["mae"]:.4f}  RMSE={overall["rmse"]:.4f}  '
              f'CV(RMSE)={overall["cv_rmse"]:.2f}%  MASE={mase_overall:.4f}')

        # Per-horizon buckets: split the horizon into three equal thirds.
        # Also persist each bucket's MAE / RMSE / CV(RMSE) to metrics_out so
        # they show up in the aggregated per-family metrics for articles.
        third = horizon // 3
        buckets = [
            ('h1_24',   0,       third,     f'0-{third}h'),
            ('h25_48',  third,   2 * third, f'{third}-{2*third}h'),
            ('h49_72',  2*third, horizon,   f'{2*third}-{horizon}h'),
        ]
        for key, start, end, plabel in buckets:
            end = min(end, horizon)
            if start >= horizon:
                break
            m = calculate_metrics(y_inv[:, start:end].flatten(), p_inv[:, start:end].flatten())
            metrics_out[tag][f'mae_{key}']     = float(m['mae'])
            metrics_out[tag][f'rmse_{key}']    = float(m['rmse'])
            metrics_out[tag][f'cv_rmse_{key}'] = float(m['cv_rmse'])
            print(f'  {plabel:8s}:  MAE={m["mae"]:.4f}  RMSE={m["rmse"]:.4f}  CV(RMSE)={m["cv_rmse"]:.2f}%')

    # Persist metrics alongside the models so the pipeline can aggregate them
    metrics_path = os.path.join(model_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics_out, f, indent=2)
    print(f'\nSaved metrics -> {metrics_path}')
    return metrics_out


def _latest_run_dir() -> str:
    """Return the most recently created subdirectory of runs/, or _DIR if none exist."""
    runs_root = os.path.join(_DIR, 'runs')
    if os.path.isdir(runs_root):
        entries = [
            os.path.join(runs_root, d) for d in os.listdir(runs_root)
            if os.path.isdir(os.path.join(runs_root, d))
        ]
        if entries:
            return max(entries, key=os.path.getctime)
    return _DIR


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    base = os.path.join(os.path.dirname(_DIR), '..', 'data', 'HouseZero', 'processed')
    p.add_argument('--csv',       default=os.path.join(base, 'consumption', 'data_hourly_test.csv'))
    p.add_argument('--prefix',    default='lstm_cnn_consumption')
    p.add_argument('--model_dir', default=None,
                   help='Path to a training run directory. Defaults to the latest runs/ subdirectory.')
    args = p.parse_args()
    model_dir = args.model_dir or _latest_run_dir()
    print(f'Using model dir: {model_dir}')
    evaluate(args.csv, args.prefix, model_dir)
