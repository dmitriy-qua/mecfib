"""
Per-horizon ensemble weight tuning for consumption forecasting.
===============================================================

Runs the adaptive forecaster (base models only) over the last val_frac of
training data and collects out-of-fold LSTM-CNN and GBDT predictions.

For each target and each horizon step h ∈ {1…24}, the optimal LSTM weight
w*[h] ∈ [0, 1] is found by grid search minimising MAE:

    combined[h] = w*[h] · lstm[h] + (1 − w*[h]) · gbdt[h]

The result is saved to weights_per_horizon.json in this directory.
The forecaster loads it automatically and falls back to fixed default weights
when the file is absent.

Usage
-----
    # Freeze base models (recommended after full training)
    python -m forecasting.consumption.adaptive.tune_weights --no_adaptive

    # Allow LSTM fine-tuning during OOF collection
    python -m forecasting.consumption.adaptive.tune_weights

    # Custom data path / validation fraction
    python -m forecasting.consumption.adaptive.tune_weights \\
        --csv path/to/data_hourly_train.csv --val_frac 0.15 --no_adaptive
"""

import argparse
import json
import os
import sys

import pandas as pd
import numpy as np
import torch

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_DIR)))))

from forecasting.consumption.adaptive import (
    AdaptiveForecaster, create_default_args, validate_args, apply_model_meta,
    apply_dataset_thresholds,
)
from forecasting.consumption.adaptive.utils import (
    tune_per_horizon_weights, calculate_metrics_adaptive, check_threshold_exceeded,
)
from forecasting.consumption.constants import TARGET_COLUMNS, SEQUENCE_LENGTH
from forecasting.constants import FORECAST_HORIZON  # default fallback only

_TAG = {
    'thermal_heating_kw': 'heating',
    'thermal_cooling_kw': 'cooling',
    'dhw_kw':             'dhw',
    'electric_demand_kw': 'electric',
}

_DEFAULT_WEIGHTS_PATH = os.path.join(_DIR, 'weights_per_horizon.json')

# GBDT-floor schedules offered to the validation-based selection (anchors at
# h = 1, 12, 24; linear interpolation between them). Ordered from least to
# most constrained; ties go to the least constrained.
FLOOR_HORIZONS = (1, 12, 24)
FLOOR_CANDIDATES = {
    'none':    (0.00, 0.00, 0.00),
    'low':     (0.50, 0.30, 0.20),
    'default': (0.80, 0.45, 0.30),
    'high':    (0.85, 0.55, 0.45),
}


def _blend(w, lstm_list, gbdt_list, tft_list=None) -> np.ndarray:
    """Apply per-horizon weights (2-way array or 3-way dict) to window lists."""
    L, G = np.stack(lstm_list), np.stack(gbdt_list)
    if isinstance(w, dict):
        T = np.stack(tft_list) if tft_list else np.zeros_like(L)
        return (np.asarray(w['lstm']) * L + np.asarray(w['gbdt']) * G
                + np.asarray(w.get('tft', 0.0)) * T)
    wl = np.asarray(w)
    return wl * L + (1.0 - wl) * G


def _print_weights(weights: dict, horizon: int):
    print('\n' + '=' * 60)
    print('  Per-horizon ensemble weights (OOF grid search)')
    print('=' * 60)
    for t, w_entry in weights.items():
        tag = _TAG.get(t, t)
        print(f'\n  {tag}')
        if isinstance(w_entry, dict):
            wl = w_entry['lstm']; wg = w_entry['gbdt']; wt = w_entry['tft']
            print(f'  {"Horizon":>8}  {"GBDT":>8}  {"LSTM-CNN":>10}  {"TFT":>8}')
            print(f'  {"-" * 40}')
            for h in range(horizon):
                print(f'  h+{h+1:>2}     {float(wg[h]):>8.3f}  {float(wl[h]):>10.3f}  {float(wt[h]):>8.3f}')
        else:
            print(f'  {"Horizon":>8}  {"GBDT weight":>12}  {"LSTM-CNN weight":>16}')
            print(f'  {"-" * 40}')
            for h in range(horizon):
                w_lstm = float(w_entry[h])
                print(f'  h+{h+1:>2}     {1-w_lstm:>12.3f}  {w_lstm:>16.3f}')


def parse_args():
    p = argparse.ArgumentParser(description='Tune per-horizon ensemble weights for consumption.')
    base = os.path.join(_DIR, '..', '..', 'data', 'HouseZero', 'processed')
    p.add_argument('--csv',         default=os.path.join(base, 'data_hourly_train.csv'),
                   help='Path to training CSV')
    p.add_argument('--val_frac',    type=float, default=0.15,
                   help='Fraction of training data used as OOF validation window')
    p.add_argument('--no_adaptive', action='store_true',
                   help='Freeze base models during OOF collection (no LSTM fine-tuning). '
                        'Faster and produces cleaner weight estimates when base models are '
                        'already well-trained.')
    p.add_argument('--output_dir', type=str, default=_DIR,
                   help='Directory where weights_per_horizon.json is saved')
    p.add_argument('--model_dir',  type=str, default=None,
                   help='Run directory containing lstm_cnn/ and gbdt/ subdirectories '
                        'with the trained models. When omitted, falls back to the '
                        'legacy source-tree paths.')
    p.add_argument('--no_tft', action='store_true',
                   help='Ignore the TFT head and tune two-way LSTM+GBDT weights '
                        '(for the two-head variant of a three-head run).')
    p.add_argument('--protocol', type=str, default='', choices=['', 'legacy', 'clean'],
                   help='Empty = inherit $FORECAST_PROTOCOL (set by train_pipeline).')
    p.add_argument('--floor_selection', type=str, default='', choices=['', 'dataset', 'validation'],
                   help="'dataset': GBDT floor from constants.py / default. 'validation': per "
                        "target, pick the floor schedule with the lowest MAE on the second half "
                        "of the validation windows (weights tuned on the first half). "
                        "Empty = 'validation' under the clean protocol, else 'dataset'.")
    return p.parse_args()


def main():
    args_cli = parse_args()

    data = pd.read_csv(args_cli.csv, parse_dates=['date'], index_col='date')
    # Datasets without a given channel (e.g. HEEW has no DHW) never ship that
    # target column, so resolve which of TARGET_COLUMNS this CSV actually
    # provides before dropna(subset=...) — otherwise pandas raises KeyError on
    # the absent column. The trained models only cover the present targets too.
    available_targets = [t for t in TARGET_COLUMNS if t in data.columns]
    if not available_targets:
        raise SystemExit(
            f'CSV {args_cli.csv} contains none of the target columns {TARGET_COLUMNS}.'
        )
    if set(available_targets) != set(TARGET_COLUMNS):
        _missing = [t for t in TARGET_COLUMNS if t not in available_targets]
        print(f'[tune_weights] dataset is missing target column(s) {_missing}; '
              f'those heads are skipped.')
    data = data.dropna(subset=available_targets).ffill().fillna(0)
    print(f'Loaded {len(data)} rows from {args_cli.csv}')

    if args_cli.protocol:
        os.environ['FORECAST_PROTOCOL'] = args_cli.protocol
    clean = os.environ.get('FORECAST_PROTOCOL', '').strip().lower() == 'clean'
    floor_sel = args_cli.floor_selection or ('validation' if clean else 'dataset')

    fore_args = create_default_args()
    if args_cli.no_tft:
        fore_args.members = 'lstm,gbdt'
    fore_args.non_adaptive = args_cli.no_adaptive
    fore_args.use_horizon_weights = False   # collect raw base-model outputs, not combined
    fore_args.plot_interval = 0
    fore_args.save_plots    = False
    fore_args.model_base_dir = args_cli.model_dir
    seq_len, horizon = apply_model_meta(fore_args)
    # Pick up dataset-specific GBDT-floor anchors (and any other thresholds
    # the dataset declares) so the offline weight tuning matches what the
    # online _retune_weights does at adaptive-run time.
    apply_dataset_thresholds(fore_args, args_cli.csv)

    n_val     = int(len(data) * args_cli.val_frac)
    val_start = max(seq_len, len(data) - n_val)
    print(f'Validation window: rows {val_start} -> {len(data) - 1}  ({len(data) - val_start} rows)')
    fore_args.start_index = val_start
    validate_args(fore_args)
    print(f'Base model mode: {"frozen" if args_cli.no_adaptive else "adaptive (fine-tuning enabled)"}')

    device     = 'cuda' if torch.cuda.is_available() else 'cpu'
    forecaster = AdaptiveForecaster(fore_args, device)

    # The forecaster resolves the present targets from the checkpoints it could
    # actually load; use that as the authoritative set for the OOF collection so
    # tuning stays consistent with the models being blended.
    targets = forecaster.targets

    lstm_wins = {t: [] for t in targets}
    gbdt_wins = {t: [] for t in targets}
    tft_wins  = {t: [] for t in targets}
    act_wins  = {t: [] for t in targets}
    naive_wins = {t: [] for t in targets}
    gate_wins  = {t: [] for t in targets}   # True when the runtime gates zero the window
    win_starts = []

    n_windows   = 0
    retrain_cnt = 0
    i           = val_start

    while i + horizon <= len(data):
        lstm_p, gbdt_p, tft_p, combined = forecaster.predict_components(data, i)

        end     = i + horizon
        actuals = {t: data[t].iloc[i:end].values for t in targets}

        for t in targets:
            lstm_wins[t].append(lstm_p[t])
            gbdt_wins[t].append(gbdt_p[t])
            if t in tft_p:
                tft_wins[t].append(tft_p[t])
            act_wins[t].append(actuals[t])
            _nv = (getattr(forecaster, 'last_stages', None) or {}).get('naive', {}).get(t)
            naive_wins[t].append(_nv if _nv is not None else np.full(horizon, np.nan))
            gate_wins[t].append(bool(np.all(np.asarray(combined[t]) == 0)))
        win_starts.append(data.index[i])

        if not args_cli.no_adaptive:
            metrics = calculate_metrics_adaptive(actuals, combined, data=data, start_idx=i)
            if (n_windows + 1) % fore_args.evaluation_frequency == 0:
                should, reasons = check_threshold_exceeded(
                    metrics, fore_args.mae_threshold, fore_args.rmse_threshold,
                    mase_thr=getattr(fore_args, 'mase_threshold', None),
                    per_target_mase_thr=getattr(fore_args, 'per_target_mase_threshold', None),
                    targets=targets,
                )
                if should and retrain_cnt < fore_args.max_retrains:
                    win_start = max(0, i - fore_args.retrain_window)
                    forecaster._finetune_lstm(data.iloc[win_start:i])
                    retrain_cnt += 1
                    print(f'  [adaptive] fine-tune #{retrain_cnt} at step {i}: {reasons}')

        n_windows += 1
        i += horizon

    ft_summary = f'fine-tunes: {retrain_cnt}' if not args_cli.no_adaptive else 'fine-tuning disabled'
    print(f'Collected {n_windows} OOF windows  |  {ft_summary}')

    have_tft = all(len(tft_wins[t]) == n_windows and n_windows > 0
                   for t in targets)
    mode_label = '3-way (LSTM+GBDT+TFT)' if have_tft else '2-way (LSTM+GBDT)'
    print(f'\nTuning per-horizon weights — {mode_label}...')
    floor_kw = {}
    if hasattr(fore_args, 'gbdt_floor_horizons') and hasattr(fore_args, 'gbdt_floor_weights'):
        floor_kw['gbdt_floor_horizons'] = fore_args.gbdt_floor_horizons
        floor_kw['gbdt_floor_weights']  = fore_args.gbdt_floor_weights

    def _tune(t, sl, fh=None, fw=None):
        kw = ({'gbdt_floor_horizons': fh, 'gbdt_floor_weights': fw} if fh is not None
              else dict(floor_kw))
        return tune_per_horizon_weights(
            {t: lstm_wins[t][sl]}, {t: gbdt_wins[t][sl]}, {t: act_wins[t][sl]}, [t], horizon,
            tft_wins=({t: tft_wins[t][sl]} if have_tft else None), **kw)[t]

    def _score(t, w, sl):
        pred = _blend(w, lstm_wins[t][sl], gbdt_wins[t][sl], tft_wins[t][sl] if have_tft else None)
        # Score what would be issued: windows the inactivity / seasonal gates
        # zero at run time count as zero forecasts here too.
        pred = np.where(np.asarray(gate_wins[t][sl], dtype=bool)[:, None], 0.0, pred)
        err  = np.stack(act_wins[t][sl]) - pred
        return (float(np.mean(np.abs(err))),
                float(np.mean(np.sqrt(np.mean(err ** 2, axis=1)))))   # mean per-window RMSE

    def _score_deployed(t, w, sl):
        """Validation error of the forecast as it is issued at run time:
        blend -> causal weekly-naive anchor -> runtime gates.

        The anchor weight is simulated over the validation windows in order,
        exactly as AdaptiveForecaster._update_causal_anchor does (last 28
        scored windows, weight 0 until 14 are available), then the windows in
        ``sl`` are scored. This is the level the clean-protocol retrain
        thresholds are derived from, so it has to include the anchor.
        """
        every_ = slice(None)
        pred = _blend(w, lstm_wins[t][every_], gbdt_wins[t][every_],
                      tft_wins[t][every_] if have_tft else None)
        naive = np.stack(naive_wins[t])
        act   = np.stack(act_wins[t])
        gate  = np.asarray(gate_wins[t], dtype=bool)
        grid  = np.round(np.arange(0.0, 0.91, 0.1), 2)
        out, buf, alpha = np.empty_like(pred), [], 0.0
        for i in range(len(pred)):
            ok = not np.any(np.isnan(naive[i]))
            a = alpha if ok else 0.0
            out[i] = (1.0 - a) * pred[i] + (a * naive[i] if ok else 0.0)
            if gate[i]:
                out[i] = 0.0
            if ok:
                buf.append(i)
                buf = buf[-28:]
                if len(buf) >= 14:
                    B, N, A = pred[buf], naive[buf], act[buf]
                    alpha = float(grid[int(np.argmin(
                        [np.mean(np.abs(A - ((1.0 - g) * B + g * N))) for g in grid]))])
                else:
                    alpha = 0.0
        err = (act - out)[sl]
        return (float(np.mean(np.abs(err))),
                float(np.mean(np.sqrt(np.mean(err ** 2, axis=1)))))

    half = n_windows // 2
    split_ok = half >= 4
    first, second, every = slice(0, half), slice(half, None), slice(None)
    weights, floor_meta, val_meta = {}, {}, {}
    for t in targets:
        if floor_sel == 'validation' and split_ok:
            cand = {}
            for name, fw in FLOOR_CANDIDATES.items():
                cand[name] = _score(t, _tune(t, first, FLOOR_HORIZONS, fw), second)[0]
            chosen = min(FLOOR_CANDIDATES, key=lambda k: cand[k])   # first minimum = least constrained
            fh, fw = FLOOR_HORIZONS, FLOOR_CANDIDATES[chosen]
            floor_meta[t] = {'name': chosen, 'horizons': list(fh), 'weights': list(fw),
                             'second_half_mae': cand}
            print(f'  [{_TAG.get(t, t)}] validation-selected GBDT floor: {chosen} {fw}  '
                  + ', '.join(f'{k}={v:.4g}' for k, v in cand.items()))
        else:
            fh = floor_kw.get('gbdt_floor_horizons')
            fw = floor_kw.get('gbdt_floor_weights')
            if fh is not None:
                print(f'  [{_TAG.get(t, t)}] GBDT floor anchors: {list(zip(fh, fw))}')
        # Validation error of the blend: weights from the first half of the
        # validation windows, scored on the second half (when there are enough).
        if split_ok:
            mae, rmse = _score_deployed(t, _tune(t, first, fh, fw), second)
        else:
            mae, rmse = _score_deployed(t, _tune(t, every, fh, fw), every)
        val_meta[t] = {'mae': mae, 'rmse': rmse}
        weights[t] = _tune(t, every, fh, fw)

    os.makedirs(args_cli.output_dir, exist_ok=True)
    weights_path = os.path.join(args_cli.output_dir, 'weights_per_horizon.json')
    if have_tft:
        payload = {'weights': {
            t: {k: v.tolist() for k, v in weights[t].items()} for t in targets
        }}
    else:
        payload = {'weights': {t: weights[t].tolist() for t in targets}}
    with open(weights_path, 'w') as fh:
        json.dump(payload, fh, indent=2)
    print(f'Saved  {os.path.basename(weights_path)}')

    # Sidecar: how the floor was chosen and the validation error levels the
    # clean protocol turns into retrain thresholds.
    meta = {
        'protocol':        'clean' if clean else 'legacy',
        'floor_selection': floor_sel if split_ok else 'dataset',
        'mode':            '3-way' if have_tft else '2-way',
        'n_windows':       int(n_windows),
        'validation_from': str(data.index[val_start]),
        'validation_to':   str(data.index[-1]),
        'floor':           floor_meta,
        'validation': {
            'scoring': (('weights tuned on the first half of the validation windows, '
                         'scored on the second half' if split_ok else 'in-sample')
                        + '; causal naive anchor and runtime gates applied'),
            'per_target':   val_meta,
            'overall_mae':  float(np.mean([v['mae']  for v in val_meta.values()])),
            'overall_rmse': float(np.mean([v['rmse'] for v in val_meta.values()])),
        },
    }
    with open(os.path.join(args_cli.output_dir, 'weights_meta.json'), 'w') as fh:
        json.dump(meta, fh, indent=2)
    print('Saved  weights_meta.json')

    # Base forecasts on the validation windows, for offline combiner studies.
    frames = []
    for t in targets:
        cols = {
            'target':       np.repeat(_TAG.get(t, t), n_windows * horizon),
            'window_idx':   np.repeat(np.arange(n_windows), horizon),
            'window_start': np.repeat(pd.DatetimeIndex(win_starts).values, horizon),
            'h':            np.tile(np.arange(1, horizon + 1), n_windows),
            'actual':       np.stack(act_wins[t]).ravel(),
            'naive_168h':   np.stack(naive_wins[t]).ravel(),
            'lstm':         np.stack(lstm_wins[t]).ravel(),
            'gbdt':         np.stack(gbdt_wins[t]).ravel(),
            'gate_zeroed':  np.repeat(np.asarray(gate_wins[t], dtype=np.int8), horizon),
        }
        if have_tft:
            cols['tft'] = np.stack(tft_wins[t]).ravel()
        frames.append(pd.DataFrame(cols))
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(
            os.path.join(args_cli.output_dir, 'oof_predictions.csv.gz'),
            index=False, float_format='%.6g')
        print('Saved  oof_predictions.csv.gz')

    _print_weights(weights, horizon)


if __name__ == '__main__':
    main()
