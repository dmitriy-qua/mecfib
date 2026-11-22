"""TFT inference diagnostic for the adaptive consumption forecaster.

Use this script when the TFT branch of the ensemble is misbehaving (e.g.
3–4× overpredictions in raw kW). It loads the same checkpoints the
adaptive runner uses, runs ONE forecast window, and prints intermediate
values at every stage of the inverse-transform chain so the source of the
miscalibration is obvious from the output.

What it logs per (target, window):
    1. Scaler identity (which scaler instance, which feature names, what
       data_min_ / data_max_ values for the target column).
    2. Raw TFT output in scaled space — min/max/mean/std + a flag when the
       output drifts outside [0, 1] (the head is a raw Linear with no
       sigmoid).
    3. Post-inverse-transform output in raw kW — before and after the
       clamp/cap that ``_predict_tft`` now applies.
    4. Per-horizon comparison vs the actuals for the same window.

Usage
-----
    python -m forecasting.consumption.adaptive.diagnose_tft \\
        --csv path/to/data_hourly_test.csv \\
        --model_dir runs/<training-run-dir> \\
        --window_idx 2248
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_DIR)))))

from forecasting.consumption.adaptive import (
    AdaptiveForecaster, create_default_args, validate_args, apply_model_meta,
)
from forecasting.consumption.adaptive.utils import prepare_lstm_input
from forecasting.consumption.constants import TARGET_COLUMNS, LSTM_COLUMNS

_TAG = {
    'thermal_heating_kw': 'heating',
    'thermal_cooling_kw': 'cooling',
    'dhw_kw':             'dhw',
    'electric_demand_kw': 'electric',
}


def _summarise(arr: np.ndarray, name: str) -> str:
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return f'{name}: all-NaN'
    return (f'{name}: min={finite.min():.4f}  max={finite.max():.4f}  '
            f'mean={finite.mean():.4f}  std={finite.std():.4f}  '
            f'(n={finite.size})')


def _parse_args():
    p = argparse.ArgumentParser(
        description='Single-window TFT inference diagnostic.')
    p.add_argument('--csv', required=True,
                   help='Path to the test/eval CSV (date-indexed, hourly).')
    p.add_argument('--model_dir', default=None,
                   help='Training-run directory containing lstm_cnn/, gbdt/, '
                        'transformer/. When omitted, falls back to the '
                        'source-tree paths used by AdaptiveForecaster.')
    p.add_argument('--window_idx', type=int, default=0,
                   help='Which rolling-window index to inspect. 0 = first '
                        'forecastable window, etc. Mirrors the indices in '
                        'forecast_samples/sample_*_w####.png.')
    p.add_argument('--stride_hours', type=int, default=0,
                   help='Stride between windows; defaults to forecast horizon.')
    p.add_argument('--start_hour_offset', type=int, default=None)
    p.add_argument('--targets', default='',
                   help='Comma-separated subset of targets to inspect. Empty '
                        '= all four heads.')
    return p.parse_args()


def _resolve_window_start(args, data, horizon: int, seq_len: int) -> int:
    """Replicate the window-index → row-index logic from run.py."""
    n = len(data)
    i = seq_len
    if args.start_hour_offset is not None:
        target_hour = int(args.start_hour_offset) % 24
        j = i
        while j < n and data.index[j].hour != target_hour:
            j += 1
        if j < n:
            i = j
    stride = args.stride_hours if args.stride_hours and args.stride_hours > 0 else horizon
    return i + stride * int(args.window_idx)


def main():
    cli = _parse_args()
    if not os.path.exists(cli.csv):
        raise FileNotFoundError(f'CSV not found: {cli.csv}')
    data = pd.read_csv(cli.csv, parse_dates=['date'], index_col='date')
    missing = [c for c in TARGET_COLUMNS if c not in data.columns]
    if missing:
        raise ValueError(f'CSV missing target columns: {missing}')
    data = data.dropna(subset=TARGET_COLUMNS).ffill().fillna(0)
    print(f'[diagnose-tft] loaded {len(data)} rows from {cli.csv}')

    # Build args + load the same forecaster the adaptive runner uses.
    args = create_default_args()
    args.non_adaptive    = True   # diagnostic only, no retraining
    args.model_base_dir  = cli.model_dir
    apply_model_meta(args)
    validate_args(args)
    device     = 'cuda' if torch.cuda.is_available() else 'cpu'
    forecaster = AdaptiveForecaster(args, device)

    if not forecaster.tft_models:
        print('[diagnose-tft] FATAL: no TFT checkpoints loaded from this run dir.')
        print(f'  Looked under: {os.path.join(args.model_base_dir or "<source-tree>", "transformer/")}')
        return

    horizon = int(args.forecast_horizon)
    seq_len = int(args.sequence_length)
    i = _resolve_window_start(cli, data, horizon, seq_len)
    if i + horizon > len(data):
        raise IndexError(
            f'window_idx={cli.window_idx} resolves to row {i}, but data has '
            f'only {len(data)} rows (need {i + horizon}).')

    ts_ctx = data.index[i - seq_len:i]
    ts_fc  = data.index[i:i + horizon]
    print('\n' + '=' * 70)
    print(f'  Window idx {cli.window_idx}  (row {i})')
    print(f'  Context:  {ts_ctx[0]} → {ts_ctx[-1]}')
    print(f'  Forecast: {ts_fc[0]} → {ts_fc[-1]}')
    print('=' * 70)

    # ---- TFT scaler identity ------------------------------------------------
    scaler = forecaster.tft_scaler or forecaster.lstm_scaler
    cols = (list(scaler.feature_names_in_)
            if hasattr(scaler, 'feature_names_in_') else list(LSTM_COLUMNS))
    print('\n[1] Scaler identity')
    print(f'  using {"tft_scaler" if scaler is forecaster.tft_scaler else "lstm_scaler (FALLBACK)"}')
    print(f'  feature_names_in_ has {len(cols)} columns')
    for t in TARGET_COLUMNS:
        if t in cols:
            ti = cols.index(t)
            tmin = float(scaler.data_min_[ti])
            tmax = float(scaler.data_max_[ti])
            print(f'  {_TAG[t]:<10} data_min={tmin:.4f}  data_max={tmax:.4f}  '
                  f'(actual range in CSV: {data[t].min():.4f}..{data[t].max():.4f})')
            if data[t].max() > tmax * 1.05:
                print(f'    WARNING: CSV max exceeds scaler.data_max_ '
                      f'by {data[t].max() / max(tmax, 1e-6):.2f}x — '
                      f'training-time clip-quantile likely active.')

    # ---- Raw TFT output -----------------------------------------------------
    log1p_targets = (forecaster.lstm_log1p_targets
                      if scaler is forecaster.lstm_scaler else None)
    tft_input = prepare_lstm_input(
        data, i - seq_len, seq_len, scaler, log1p_targets=log1p_targets,
    ).to(device)

    targets_to_inspect = [t.strip() for t in cli.targets.split(',') if t.strip()]
    if not targets_to_inspect:
        targets_to_inspect = list(forecaster.tft_models.keys())
    actuals = {t: data[t].iloc[i:i + horizon].values for t in TARGET_COLUMNS}

    print('\n[2] Raw TFT outputs (scaled space)  /  inverse-transform chain')
    for t in targets_to_inspect:
        if t not in forecaster.tft_models:
            print(f'  {_TAG.get(t, t)}: no TFT model loaded — skipped')
            continue
        model = forecaster.tft_models[t]
        with torch.no_grad():
            raw_scaled = model(tft_input).cpu().numpy().flatten()
        oob_lo = int(np.sum(raw_scaled < 0.0))
        oob_hi = int(np.sum(raw_scaled > 1.0))
        ti   = cols.index(t)
        tmin = float(scaler.data_min_[ti])
        tmax = float(scaler.data_max_[ti])

        # Two parallel reconstructions: legacy (no clamp) vs current (clamp).
        legacy = raw_scaled * (tmax - tmin) + tmin
        legacy = np.clip(legacy, 0.0, None)
        clamped_scaled = np.clip(raw_scaled, 0.0, 1.0)
        current = clamped_scaled * (tmax - tmin) + tmin
        current = np.clip(current, 0.0, None)
        cap = tmax * 1.5
        current_capped = np.minimum(current, cap)

        act = actuals[t]
        print(f'\n  --- {_TAG[t]} ---')
        print(f'  {_summarise(raw_scaled, "raw_scaled")}')
        print(f'  out-of-[0,1]: lo={oob_lo}/{raw_scaled.size}  hi={oob_hi}/{raw_scaled.size}')
        print(f'  {_summarise(legacy, "legacy_inverse_kw   (no clamp)")}')
        print(f'  {_summarise(current_capped, "patched_inverse_kw (clamp [0,1] + cap)")}')
        print(f'  {_summarise(act, "actual_kw")}')
        # Highlight the gap so eye-balling the headline number is easy.
        if act.size:
            ratio_legacy  = float(np.nanmean(legacy)        / max(np.nanmean(act), 1e-6))
            ratio_patched = float(np.nanmean(current_capped) / max(np.nanmean(act), 1e-6))
            print(f'  mean ratio  legacy/actual  : {ratio_legacy:.2f}x')
            print(f'  mean ratio  patched/actual : {ratio_patched:.2f}x')

    print('\n[3] Per-horizon comparison (first 12 steps)')
    print(f'  {"h":>3}  {"actual":>10}  ' +
          '  '.join(f'{_TAG[t]:>10}' for t in targets_to_inspect))
    # Use the patched (clamp + cap) values for the side-by-side table —
    # that's what the runner now produces.
    patched_by_t = {}
    for t in targets_to_inspect:
        if t not in forecaster.tft_models:
            continue
        with torch.no_grad():
            raw_scaled = forecaster.tft_models[t](tft_input).cpu().numpy().flatten()
        ti   = cols.index(t)
        tmin = float(scaler.data_min_[ti]); tmax = float(scaler.data_max_[ti])
        v = np.clip(raw_scaled, 0.0, 1.0) * (tmax - tmin) + tmin
        v = np.clip(v, 0.0, None)
        patched_by_t[t] = np.minimum(v, tmax * 1.5)
    for h in range(min(12, horizon)):
        row = f'  h+{h+1:<2}'
        # Use the first target's actual as a reference column
        ref_t = targets_to_inspect[0] if targets_to_inspect else None
        if ref_t and ref_t in actuals:
            row += f'  {actuals[ref_t][h]:>10.3f}'
        else:
            row += f'  {"":>10}'
        for t in targets_to_inspect:
            v = patched_by_t.get(t, [float('nan')] * horizon)[h]
            row += f'  {v:>10.3f}'
        print(row)


if __name__ == '__main__':
    main()
