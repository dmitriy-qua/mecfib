"""Evaluate per-target TFT consumption models.

Mirrors ``forecasting.consumption.lstm_cnn.evaluate``: same data path,
seasonal slicing, MASE baseline, and per-third bucketing. Only the
checkpoint reload and model class differ.
"""

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
from forecasting.consumption.lstm_cnn.train import _TAG, _SEASON_FILTER
from forecasting.consumption.transformer.model import TFTModel


def _infer_arch_from_state_dict(sd: dict) -> dict:
    """Reconstruct the TFT hyperparameters needed to instantiate the model
    from a saved state dict, so we never depend on stale arch.json values.
    """
    hidden_dim  = sd['lstm_encoder.weight_hh_l0'].shape[1]
    lstm_layers = sum(1 for k in sd if k.startswith('lstm_encoder.weight_ih_l'))
    # MultiheadAttention packs Q, K, V into one (3H, H) projection weight.
    n_heads = None
    in_proj = sd.get('attention.in_proj_weight')
    if in_proj is not None and in_proj.shape[0] == 3 * hidden_dim:
        for h in (1, 2, 4, 8, 16):
            if hidden_dim % h == 0:
                n_heads = h
        n_heads = n_heads or 1
    return {
        'hidden_dim':  hidden_dim,
        'lstm_layers': lstm_layers,
        'n_heads':     n_heads or 4,
    }


def evaluate(csv_path: str, model_prefix: str, model_dir: str) -> dict:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    with open(os.path.join(model_dir, f'scaler_{model_prefix}_initial.sav'), 'rb') as f:
        scaler = pickle.load(f)

    arch_path = os.path.join(model_dir, f'{model_prefix}_arch.json')
    arch_meta: dict = {}
    if os.path.exists(arch_path):
        with open(arch_path) as _f:
            arch_meta = json.load(_f)
    seq_len = int(arch_meta.get('sequence_length',  SEQUENCE_LENGTH))
    horizon = int(arch_meta.get('forecast_horizon', FORECAST_HORIZON))
    cols    = list(arch_meta.get('features') or LSTM_COLUMNS)

    # Which target heads to evaluate. Prefer the `targets` persisted by train.py
    # (a subset of TARGET_COLUMNS when the run trained fewer heads, e.g. a
    # dataset like HEEW with no dhw_kw channel). Fall back to TARGET_COLUMNS for
    # legacy runs that pre-date the persisted key, and intersect with `cols` so
    # a target the dataset never provided can't slip into dropna / the eval loop.
    eval_targets = [t for t in (arch_meta.get('targets') or TARGET_COLUMNS) if t in cols]

    data = pd.read_csv(csv_path, parse_dates=['date'], index_col='date',
                       usecols=['date'] + cols).dropna(subset=eval_targets).ffill().fillna(0)
    hs_idx = cols.index('is_heating_season')

    # Mirror training: datasets with APPLY_SEASONAL_GATE=False (e.g. HEEW) train
    # heating/cooling all-year, so evaluate them all-year too.
    try:
        from forecasting.consumption.adaptive import load_dataset_seasonal_gate
        _seasonal_gate = load_dataset_seasonal_gate(csv_path)
    except ImportError:
        _seasonal_gate = True
    season_filter = dict(_SEASON_FILTER)
    if not _seasonal_gate:
        season_filter['thermal_heating_kw'] = None
        season_filter['thermal_cooling_kw'] = None

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
        y_s = y_all[mask, :, t_idx]

        pth_path = os.path.join(model_dir, f'{model_prefix}_{tag}_initial.pth')
        sd = torch.load(pth_path, map_location=device, weights_only=False)
        inferred = _infer_arch_from_state_dict(sd)
        dropout  = arch_meta.get('dropout', 0.1)

        model = TFTModel(
            input_size=X_s.shape[2],
            output_size=horizon,
            num_features=1,
            hidden_dim=inferred['hidden_dim'],
            n_heads=inferred['n_heads'],
            lstm_layers=inferred['lstm_layers'],
            dropout=dropout,
        ).to(device)
        model.load_state_dict(sd)
        model.eval()

        batch_size = 128
        preds_chunks = []
        X_tensor = torch.tensor(X_s, dtype=torch.float32)
        with torch.no_grad():
            for start in range(0, len(X_tensor), batch_size):
                chunk = X_tensor[start:start + batch_size].to(device)
                preds_chunks.append(model(chunk).cpu().numpy())
        preds = np.concatenate(preds_chunks, axis=0)[:, :, 0]
        if device == 'cuda':
            torch.cuda.empty_cache()

        n_samples, horizon_eff = y_s.shape
        n_flat = n_samples * horizon_eff

        dummy_true = np.zeros((n_flat, n_cols))
        dummy_pred = np.zeros((n_flat, n_cols))
        dummy_true[:, t_idx] = y_s.flatten()
        dummy_pred[:, t_idx] = preds.flatten()

        y_inv = scaler.inverse_transform(dummy_true)[:, t_idx].reshape(n_samples, horizon_eff)
        p_inv = scaler.inverse_transform(dummy_pred)[:, t_idx].reshape(n_samples, horizon_eff)

        # Weekly-naive baseline ŷ(t+h) = y(t+h - 168), drawn from the context
        # window that immediately precedes the forecast window.
        naive_scaled = weekly_naive_from_window(X_s, t_idx, seq_len, horizon_eff)
        naive_steps  = naive_scaled.shape[1]
        dummy_naive  = np.zeros((n_samples * naive_steps, n_cols))
        dummy_naive[:, t_idx] = naive_scaled.flatten()
        if naive_steps:
            naive_inv = scaler.inverse_transform(dummy_naive)[:, t_idx].reshape(n_samples, naive_steps)
        else:      # context shorter than one week: no weekly-naive reference, MASE is nan
            naive_inv = np.zeros((n_samples, 0))

        # Sub-week contexts (seq_len < 168) cannot form a weekly lag in-window;
        # MASE is left nan there (use forecasting.baselines.run for the floor).
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
            'mae':       float(overall['mae']),
            'rmse':      float(overall['rmse']),
            'cv_rmse':   float(overall['cv_rmse']),
            'mase':      float(mase_overall),
            'n_samples': int(n_samples),
            'horizon':   int(horizon_eff),
        }
        print(f'\n{tag}')
        print(f'  overall  :  MAE={overall["mae"]:.4f}  RMSE={overall["rmse"]:.4f}  '
              f'CV(RMSE)={overall["cv_rmse"]:.2f}%  MASE={mase_overall:.4f}')

        third = horizon_eff // 3
        buckets = [
            ('h1_24',   0,         third,         f'0-{third}h'),
            ('h25_48',  third,     2 * third,     f'{third}-{2*third}h'),
            ('h49_72',  2 * third, horizon_eff,   f'{2*third}-{horizon_eff}h'),
        ]
        for key, start, end, plabel in buckets:
            end = min(end, horizon_eff)
            if start >= horizon_eff:
                break
            m = calculate_metrics(y_inv[:, start:end].flatten(), p_inv[:, start:end].flatten())
            metrics_out[tag][f'mae_{key}']     = float(m['mae'])
            metrics_out[tag][f'rmse_{key}']    = float(m['rmse'])
            metrics_out[tag][f'cv_rmse_{key}'] = float(m['cv_rmse'])
            print(f'  {plabel:8s}:  MAE={m["mae"]:.4f}  RMSE={m["rmse"]:.4f}  CV(RMSE)={m["cv_rmse"]:.2f}%')

    metrics_path = os.path.join(model_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics_out, f, indent=2)
    print(f'\nSaved metrics -> {metrics_path}')
    return metrics_out


def _latest_run_dir() -> str:
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
    p.add_argument('--prefix',    default='tft_consumption')
    p.add_argument('--model_dir', default=None)
    args = p.parse_args()
    model_dir = args.model_dir or _latest_run_dir()
    print(f'Using model dir: {model_dir}')
    evaluate(args.csv, args.prefix, model_dir)
