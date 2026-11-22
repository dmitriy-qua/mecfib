"""
Adaptive vs Non-Adaptive Consumption Forecasting — Comparison Script
=====================================================================

Runs both scenarios over the test dataset, prints a per-target metric table,
and saves comparison plots to --output_dir.

Usage
-----
    python -m forecasting.consumption.adaptive.run
    python -m forecasting.consumption.adaptive.run --csv path/to/test.csv --output_dir results/
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_DIR)))))

import plot_i18n

from forecasting.consumption.adaptive import (
    AdaptiveForecaster, create_default_args, validate_args,
    apply_model_meta, apply_dataset_thresholds,
)
from forecasting.consumption.adaptive.utils import (
    pooled_r2_per_target, pooled_error_metrics_per_target,
    pooled_mase_per_target,
)
from forecasting.consumption.adaptive.visualization import (
    plot_horizon_error, plot_horizon_skill_score,
    plot_pooled_scatter, plot_load_duration,
    plot_residual_violins, plot_window_mae_cdf,
    plot_ensemble_weights, plot_weight_drift,
    plot_mae_by_driver, plot_error_calendar, plot_peak_event_scatter,
    plot_daily_energy_scatter, plot_ramp_rate_errors,
)
from forecasting.consumption.adaptive.ensemble_diagnostics import (
    run_all as run_ensemble_diagnostics,
)
from forecasting.constants import (
    TARGET_THERMAL_HEATING, TARGET_THERMAL_COOLING, TARGET_DHW, TARGET_ELECTRIC,
    FORECAST_HORIZON,  # default fallback only
)
from forecasting.consumption.constants import TARGET_COLUMNS

_TAG = {
    TARGET_THERMAL_HEATING: 'heating',
    TARGET_THERMAL_COOLING: 'cooling',
    TARGET_DHW:             'dhw',
    TARGET_ELECTRIC:        'electric',
}
_ALL_TARGETS = list(_TAG.keys())


def _present_targets(keys) -> list:
    """Canonical-ordered subset of targets actually present in *keys*.

    Datasets that omit a channel (e.g. HEEW has no DHW) only carry the targets
    they trained, so plots/tables/CSVs must iterate what's present rather than
    the full four. Accepts either a container keyed by target name (a ``series``
    dict) or one keyed by ``'<target>_<metric>'`` (an ``agg`` / per-window
    history dict) — a target counts as present if its bare name is a key or any
    key is prefixed with ``'<target>_'``.
    """
    keyset = set(keys)
    out = []
    for t in _ALL_TARGETS:
        if t in keyset or any(k.startswith(f'{t}_') for k in keyset):
            out.append(t)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='Compare adaptive vs non-adaptive consumption forecasting.')
    base = os.path.join(_DIR, '..', '..', 'data', 'HouseZero', 'processed')
    p.add_argument('--csv',        default=os.path.join(base, 'consumption', 'data_hourly_train.csv'),
                   help='Path to test CSV')
    p.add_argument('--output_dir', default=os.path.join(_DIR, 'comparison_results'),
                   help='Directory for plots and summary')
    p.add_argument('--n_samples',  type=int, default=10,
                   help='Number of example forecast windows to plot (0 = skip)')
    p.add_argument('--no_plots',   action='store_true', help='Skip plot generation')
    p.add_argument('--non_adaptive_only', action='store_true',
                   help='Run only the non-adaptive (frozen) scenario; skip adaptive fine-tuning.')
    p.add_argument('--window_stride_hours', type=int, default=0,
                   help='Step between consecutive forecast windows, in hours. 0 (default) '
                        'means stride = forecast horizon (back-to-back, no overlap). Set to '
                        '6 to issue a fresh forecast every 6 hours. Smaller stride → more '
                        'windows → longer runtime but finer-grained metrics.')
    p.add_argument('--start_hour_offset', type=int, default=None,
                   help='Hour-of-day (0..23) at which the FIRST forecast window starts. '
                        'For example, set to 8 so every forecast begins at 08:00 — the '
                        'context window is the sequence_length hours before 08:00, the '
                        'forecast covers 08:00 → 08:00 + horizon. Subsequent windows step '
                        'by --window_stride_hours from there. Leave unset to start at the '
                        'first hour after the LSTM context window, regardless of clock.')
    p.add_argument('--model_dir', default=None,
                   help='Root of a training run containing lstm_cnn/, gbdt/, and '
                        'weights/ subdirectories. When omitted, falls back to the '
                        'legacy source-tree paths next to this script.')
    # Adaptive retraining thresholds. Pass 0 (the catalog/UI sentinel) to use
    # the per-dataset defaults from forecasting/data/<NAME>/constants.py;
    # any positive value overrides them.
    p.add_argument('--mae_threshold',  type=float, default=0.0,
                   help='MAE retraining threshold (kW). 0 = use dataset default.')
    p.add_argument('--rmse_threshold', type=float, default=0.0,
                   help='RMSE retraining threshold (kW). 0 = use dataset default.')
    p.add_argument('--mase_threshold', type=float, default=0.0,
                   help='MASE retraining threshold. 0 = use dataset default.')
    p.add_argument('--evaluation_frequency', type=int, default=0,
                   help='Check retraining thresholds every N forecast windows. '
                        '0 = use the DEFAULT_EVALUATION_FREQUENCY from '
                        'forecasting/constants.py.')
    p.add_argument('--retrain_window', type=int, default=0,
                   help='Hours of past data used when an adaptive retrain is '
                        'triggered. 0 = use the dataset default '
                        '(ADAPTIVE_RETRAIN_WINDOW in '
                        'forecasting/data/<NAME>/constants.py) or the global '
                        'RETRAIN_WINDOW (2000h) when no dataset override exists.')
    p.add_argument('--protocol', type=str, default='legacy', choices=['legacy', 'clean'],
                   help="'legacy': per-dataset overrides from constants.py. 'clean': rule-based "
                        "settings only (thresholds = factor x validation error on a block mean, "
                        "validation-selected GBDT floor, causal naive anchor, hold-out weight "
                        "re-tune). Needs weights_meta.json from tune_weights.py.")
    p.add_argument('--members', type=str, default='',
                   help="Comma list of base learners in the blend, e.g. 'lstm,gbdt' or 'gbdt'. "
                        "Empty = every model the run directory holds.")
    p.add_argument('--no_tft', action='store_true',
                   help="Shortcut for --members lstm,gbdt on a three-head run "
                        "(uses <model_dir>/weights_2way).")
    p.add_argument('--anchor_mode', type=str, default='', choices=['', 'fixed', 'causal', 'off'],
                   help='Weekly-naive anchor: fixed dataset weights, causal (chosen from '
                        'trailing windows) or off. Empty = protocol default.')
    p.add_argument('--retune_holdout_frac', type=float, default=-1.0,
                   help='Share of the retrain window held out for the weight re-tune. '
                        'Negative = protocol default (legacy 0, clean 0.2).')
    p.add_argument('--retrain_accept', action='store_true',
                   help='Keep a fine-tuned model only if it is not worse than its '
                        'predecessor on the hold-out part of the retrain window.')
    p.add_argument('--no_weight_retune', action='store_true',
                   help='Do not re-tune the blend weights after a retrain.')
    p.add_argument('--gbdt_retrain_mode', type=str, default='', choices=['', 'boost', 'refit', 'auto'],
                   help="'boost' adds trees to the existing booster; 'refit' trains a new one "
                        "on the retrain window; 'auto' (with --retrain_accept) keeps, per target, "
                        "the best of previous model, boost and refit on the hold-out.")
    p.add_argument('--weights_mode', type=str, default='', choices=['', 'static', 'rolling'],
                   help="'static' = validation-tuned per-horizon weights (default); 'rolling' = "
                        "weights chosen for each window from the previous --roll_window scored "
                        "windows (static weights until --roll_min_windows are available).")
    p.add_argument('--roll_window', type=int, default=42,
                   help='Rolling weights: number of past windows used.')
    p.add_argument('--roll_min_windows', type=int, default=14,
                   help='Rolling weights: windows needed before the rolling weights are used.')
    p.add_argument('--roll_blocks', type=int, default=0,
                   help='Rolling weights: equal horizon blocks with their own weights '
                        '(0 = one weight vector per horizon step, 1 = one for the whole horizon).')
    p.add_argument('--weather_noise', type=float, default=0.0,
                   help='Forecast-weather stress test: noise level (in standard deviations of '
                        'each weather variable at a 24-h lead) added to the weather of the '
                        'forecast window. 0 = measured weather.')
    p.add_argument('--trigger', type=str, default='', choices=['', 'threshold', 'scheduled'],
                   help="'scheduled' retrains at every evaluation step regardless of error.")
    p.add_argument('--trigger_stat', type=str, default='', choices=['', 'window', 'block_mean'],
                   help="Error the thresholds are compared with: the latest window, or the mean "
                        "over the windows since the previous evaluation. Empty = protocol default.")
    p.add_argument('--threshold_factor', type=float, default=1.5,
                   help='Clean protocol: retrain when block error exceeds this factor x the '
                        'validation error.')
    p.add_argument('--seed', type=int, default=None,
                   help='Random seed for the adaptive fine-tuning steps. Omit to '
                        'use $FORECAST_SEED when set, else 42. Re-applied at the '
                        'start of each scenario.')
    p.add_argument('--no_prediction_dump', action='store_true',
                   help='Do not write predictions_<scenario>.csv.gz (hourly '
                        'predictions of every base learner and of the blend).')
    plot_i18n.add_arg(p)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Run one scenario
# ---------------------------------------------------------------------------

def run_scenario(data: pd.DataFrame, non_adaptive: bool, label: str,
                 n_samples: int = 10, model_dir: str | None = None,
                 stride_hours: int = 0, start_hour_offset: int | None = None,
                 csv_path: str | None = None,
                 mae_threshold: float = 0.0,
                 rmse_threshold: float = 0.0,
                 mase_threshold: float = 0.0,
                 evaluation_frequency: int = 0,
                 retrain_window: int = 0,
                 seed: int | None = None,
                 output_dir: str | None = None,
                 save_predictions: bool = True,
                 opts: dict | None = None):
    """
    Returns (history, retrain_steps, series, samples).

    seed          : random seed, re-applied here so the adaptive scenario does
                    not depend on what ran before it in the same process.
    output_dir    : when given, the scenario writes run_config_<tag>.json
                    (effective settings, retrain events, weight history) and,
                    unless save_predictions is False, predictions_<tag>.csv.gz.

    history       : list of per-window metric dicts
    retrain_steps : list of evaluation-step indices where a retrain fired
    series        : dict[target -> {'actual', 'pred', 'timestamps',
                                     'pred_windows', 'actual_windows', 'window_starts'}]
                    full concatenated time-series of combined predictions plus
                    per-window matrices (overlap-safe under arbitrary stride).
    samples       : list of dicts, each with context + per-model forecasts for one window
    model_dir     : optional training-run root for run-scoped model loading.
    stride_hours  : step between consecutive forecast windows. 0 → use horizon
                    (back-to-back, no overlap, original behaviour).
    start_hour_offset : optional hour-of-day to anchor the first forecast.
    """
    import torch
    from forecasting.utils import seed_everything
    seed = seed_everything(seed)
    args = create_default_args()
    args.seed = seed
    args.non_adaptive = non_adaptive
    args.model_base_dir = model_dir
    opts  = dict(opts or {})
    clean = opts.get('protocol', 'legacy') == 'clean'
    # Read by config.apply_dataset_thresholds / load_dataset_training_hyperparams.
    os.environ['FORECAST_PROTOCOL'] = 'clean' if clean else 'legacy'
    apply_model_meta(args)   # restore the trained model's window from train_meta.json

    # Per-dataset retraining thresholds (auto-detected from CSV path); any
    # explicit caller-supplied positive value overrides the dataset default.
    apply_dataset_thresholds(args, csv_path)
    overrides: list[str] = []
    if mae_threshold  > 0:
        args.mae_threshold = float(mae_threshold)
        overrides.append(f'mae={mae_threshold}')
    if rmse_threshold > 0:
        args.rmse_threshold = float(rmse_threshold)
        overrides.append(f'rmse={rmse_threshold}')
    if mase_threshold > 0:
        args.mase_threshold = float(mase_threshold)
        overrides.append(f'mase={mase_threshold}')
    if evaluation_frequency > 0:
        args.evaluation_frequency = int(evaluation_frequency)
        overrides.append(f'eval_freq={evaluation_frequency}')
    if retrain_window > 0:
        args.retrain_window = int(retrain_window)
        overrides.append(f'retrain_window={retrain_window}')
    if overrides:
        print(f'[adaptive] CLI overrides applied: {", ".join(overrides)}')
    print(f'[adaptive] effective thresholds: '
          f'mae={args.mae_threshold}, rmse={args.rmse_threshold}, mase={args.mase_threshold} '
          f'| eval every {args.evaluation_frequency} window(s) '
          f'| retrain window {args.retrain_window}h')
    validate_args(args)

    # ── Scenario options (defaults reproduce the legacy behaviour) ─────────
    members = str(opts.get('members') or '').strip().lower()
    if members:
        args.members = members
        _m = {x.strip() for x in members.split(',') if x.strip()}
        if _m == {'lstm', 'gbdt'} and model_dir:
            if os.path.isdir(os.path.join(model_dir, 'transformer')):
                w2 = os.path.join(model_dir, 'weights_2way', 'weights_per_horizon.json')
                if not os.path.exists(w2):
                    raise FileNotFoundError(
                        f'{w2} not found. A two-head run on a three-head checkpoint needs '
                        f'two-way weights: run tune_weights.py with --no_tft first.')
                args.weights_dir_name = 'weights_2way'
    args.protocol          = 'clean' if clean else 'legacy'
    args.anchor_mode       = opts.get('anchor_mode') or ('causal' if clean else 'fixed')
    _hf = opts.get('retune_holdout_frac')
    args.retune_holdout_frac = (float(_hf) if _hf is not None and float(_hf) >= 0
                                else (0.2 if clean else 0.0))
    args.retrain_accept    = bool(opts.get('retrain_accept', False))
    args.weight_retune     = not bool(opts.get('no_weight_retune', False))
    args.gbdt_retrain_mode = opts.get('gbdt_retrain_mode') or 'boost'
    args.weights_mode      = opts.get('weights_mode') or 'static'
    args.roll_window       = int(opts.get('roll_window') or 42)
    args.roll_min_windows  = int(opts.get('roll_min_windows') or 14)
    args.roll_blocks       = int(opts.get('roll_blocks') or 0)
    args.weather_noise     = float(opts.get('weather_noise') or 0.0)
    args.floor_source      = 'validation' if clean else 'dataset'
    args.trigger_mode      = opts.get('trigger') or 'threshold'
    args.trigger_stat      = opts.get('trigger_stat') or ('block_mean' if clean else 'window')
    args.threshold_factor  = float(opts.get('threshold_factor') or 1.5)
    trigger_mode, trigger_stat = args.trigger_mode, args.trigger_stat

    device     = 'cuda' if torch.cuda.is_available() else 'cpu'
    forecaster = AdaptiveForecaster(args, device)

    if clean:
        # Rule-based thresholds: factor x validation error of the blend, taken
        # from weights_meta.json (written by tune_weights.py on the validation
        # slice of the training partition). No MASE or per-target gates.
        _val = (getattr(forecaster, 'weights_meta', None) or {}).get('validation') or {}
        if not (_val.get('overall_mae') and _val.get('overall_rmse')):
            raise RuntimeError(
                'Clean protocol needs validation error levels in weights_meta.json; '
                're-run tune_weights.py for this model directory.')
        args.mae_threshold  = args.threshold_factor * float(_val['overall_mae'])
        args.rmse_threshold = args.threshold_factor * float(_val['overall_rmse'])
        args.mase_threshold = None
        args.per_target_mase_threshold = {}
        print(f'[adaptive] clean protocol: thresholds = {args.threshold_factor} x validation '
              f'-> mae={args.mae_threshold:.4g}, rmse={args.rmse_threshold:.4g} '
              f'({trigger_stat}); anchor={args.anchor_mode}; '
              f'hold-out={args.retune_holdout_frac}; floor=validation')

    # The forecaster resolves the present targets from the checkpoints it could
    # load (a dataset without DHW never produces that head); drive every
    # per-target accumulator and loop below off this set, not the global four.
    targets = forecaster.targets

    retrain_steps  = []
    eval_step_counter = [0]

    # Accumulators for full time-series reconstruction
    all_actuals    = {t: [] for t in targets}
    all_preds      = {t: [] for t in targets}
    all_timestamps = []
    # Per-window matrices kept separately so horizon-bucket metrics still work
    # when stride < horizon (overlapping windows duplicate hours in the flat
    # time-series above, but the per-window arrays remain clean).
    win_actuals: dict[str, list] = {t: [] for t in targets}
    win_preds:   dict[str, list] = {t: [] for t in targets}
    # Per-base-learner per-window predictions (lstm / gbdt / tft) — collected
    # every window so the ensemble-vs-singles diagnostics can compare them
    # head-to-head against the combined output. TFT bucket stays empty when
    # no TFT checkpoint is loaded.
    win_preds_lstm: dict[str, list] = {t: [] for t in targets}
    win_preds_gbdt: dict[str, list] = {t: [] for t in targets}
    win_preds_tft:  dict[str, list] = {t: [] for t in targets}
    win_starts_ts: list = []
    samples        = []
    # Extra per-window records for the prediction dump. ``raw`` = base-learner
    # output before run_scenario zeroes it on gated windows; ``blend`` = convex
    # combination before the weekly-naive anchor; ``anchored`` = after the
    # anchor, before the physical / inactivity gates; ``naive`` = same hour one
    # week earlier (the MASE reference).
    win_raw:      dict[str, dict[str, list]] = {
        m: {t: [] for t in targets} for m in ('lstm', 'gbdt', 'tft')}
    win_blend:    dict[str, list] = {t: [] for t in targets}
    win_anchored: dict[str, list] = {t: [] for t in targets}
    win_naive:    dict[str, list] = {t: [] for t in targets}
    win_gated:    dict[str, list] = {t: [] for t in targets}
    win_alpha:    dict[str, list] = {t: [] for t in targets}
    block_hist:   list = []     # window metrics since the last trigger evaluation
    retune_wins:  list = []     # (retrain_no, replayed windows) per retrain
    win_ts:       list = []
    retrain_events: list = []
    weights_history: list = [{
        'after_window': -1, 'event': 'initial',
        'weights': _jsonable(getattr(forecaster, 'horizon_weights', None)),
    }]

    def _tracked_adaptive_forecast(data):
        args_     = forecaster.args
        n         = len(data)
        history   = []
        i         = args_.start_index
        seq_len   = args_.sequence_length
        horizon   = args_.forecast_horizon
        stride    = stride_hours if stride_hours and stride_hours > 0 else horizon
        if stride != horizon:
            print(f'  [run] window stride = {stride}h (horizon = {horizon}h, '
                  f'overlap = {max(0, horizon - stride)}h)')

        # If a start-hour-of-day was requested, advance i to the first row at
        # or after the LSTM context window whose timestamp.hour matches. This
        # anchors the FIRST forecast at <offset>:00; subsequent windows step
        # by `stride` from there.
        if start_hour_offset is not None:
            target_hour = int(start_hour_offset) % 24
            j = i
            while j < n and data.index[j].hour != target_hour:
                j += 1
            if j >= n:
                print(f'  [run] WARNING: no row in test data at hour-of-day '
                      f'{target_hour:02d}:00 after the context window — '
                      f'falling back to the first available start.')
            else:
                i = j
                print(f'  [run] first forecast anchored at {data.index[i]} '
                      f'(start_hour_offset={target_hour:02d}:00)')

        # Decide which window indices to sample (evenly spaced)
        n_windows = max((n - i - horizon) // stride + 1, 1)
        if n_samples > 0:
            sample_indices = set(
                round(k * (n_windows - 1) / max(n_samples - 1, 1))
                for k in range(n_samples)
            )
        else:
            sample_indices = set()

        from forecasting.consumption.adaptive.utils import (
            calculate_metrics_adaptive, check_threshold_exceeded,
        )

        window_idx = 0
        while i + horizon <= n:
            # Component predictions are collected EVERY window — predict() and
            # predict_components() do the same compute under the hood; the only
            # extra cost here is keeping references to the per-model arrays so
            # the ensemble-vs-singles diagnostics can compare them.
            lstm_p, gbdt_p, tft_p, comb_p = forecaster.predict_components(data, i)
            _stages = getattr(forecaster, 'last_stages', None) or {}
            _raw = {
                'lstm': {t: np.array(lstm_p[t], dtype=float, copy=True) for t in targets if t in lstm_p},
                'gbdt': {t: np.array(gbdt_p[t], dtype=float, copy=True) for t in targets if t in gbdt_p},
                'tft':  {t: np.array(tft_p[t],  dtype=float, copy=True) for t in targets if t in tft_p},
            }
            _gated = {t: bool(np.all(np.asarray(comb_p[t]) == 0)) for t in targets}
            # If the signal gate zeroed combined, zero individual models too
            # so per-model metrics see the same gated state as the ensemble.
            for t in list(comb_p):
                if np.all(comb_p[t] == 0):
                    lstm_p[t] = np.zeros_like(lstm_p[t])
                    gbdt_p[t] = np.zeros_like(gbdt_p[t])
                    if t in tft_p:
                        tft_p[t] = np.zeros_like(tft_p[t])

            if window_idx in sample_indices:
                ctx_start = i - seq_len
                samples.append({
                    'label':               label,
                    'window_idx':          window_idx,
                    'ts_context':          data.index[ctx_start:i],
                    'ts_forecast':         data.index[i:i + horizon],
                    'context':             {t: data[t].iloc[ctx_start:i].values   for t in targets},
                    'actual':              {t: data[t].iloc[i:i + horizon].values  for t in targets},
                    'lstm':                lstm_p,
                    'gbdt':                gbdt_p,
                    'tft':                 tft_p,
                    'combined':            comb_p,
                })
            result = comb_p

            end     = i + horizon
            actuals = {t: data[t].iloc[i:end].values for t in targets}
            metrics = calculate_metrics_adaptive(actuals, result, data=data, start_idx=i)
            history.append(metrics)

            # Feed the per-model sliding-window MAE tracker that the combine
            # sanity gate consults next window. Cheap: just keeps the last N
            # window MAEs per (model, target) in memory.
            forecaster.observe_window(lstm_p, gbdt_p, tft_p, actuals)
            # Rolling blend weights learn from the member forecasts as issued
            # (not zeroed by the gates); a no-op unless weights_mode == 'rolling'.
            forecaster._update_rolling_weights(_raw['lstm'], _raw['gbdt'], _raw['tft'], actuals)

            all_timestamps.extend(data.index[i:end].tolist())
            for t in targets:
                all_actuals[t].extend(actuals[t].tolist())
                all_preds[t].extend(result[t].tolist())

            # Per-window snapshots (overlap-safe). The per-model arrays let the
            # downstream ensemble-vs-singles plots avoid double-evaluating the
            # base learners.
            for t in targets:
                win_actuals[t].append(np.asarray(actuals[t], dtype=float))
                win_preds[t].append(np.asarray(result[t],   dtype=float))
                win_preds_lstm[t].append(np.asarray(lstm_p[t], dtype=float))
                win_preds_gbdt[t].append(np.asarray(gbdt_p[t], dtype=float))
                if t in tft_p:
                    win_preds_tft[t].append(np.asarray(tft_p[t], dtype=float))
                for _m in ('lstm', 'gbdt', 'tft'):
                    if t in _raw[_m]:
                        win_raw[_m][t].append(_raw[_m][t])
                if t in _stages.get('blend', {}):
                    win_blend[t].append(_stages['blend'][t])
                if t in _stages.get('anchored', {}):
                    win_anchored[t].append(_stages['anchored'][t])
                win_naive[t].append(_weekly_naive_window(data, t, i, horizon))
                win_gated[t].append(_gated[t])
                win_alpha[t].append(float((_stages.get('anchor_weights') or {}).get(t, 0.0) or 0.0))
            win_starts_ts.append(data.index[i])
            win_ts.append(np.asarray(data.index[i:end].values, dtype='datetime64[ns]'))

            eval_step_counter[0] += 1
            block_hist.append(metrics)
            if eval_step_counter[0] % args_.evaluation_frequency == 0:
                _trig = _block_mean(block_hist) if trigger_stat == 'block_mean' else metrics
                del block_hist[:]
                should, reasons = check_threshold_exceeded(
                    _trig,
                    args_.mae_threshold, args_.rmse_threshold,
                    mase_thr=getattr(args_, 'mase_threshold', None),
                    per_target_mase_thr=getattr(args_, 'per_target_mase_threshold', None),
                    targets=targets,
                )
                if trigger_mode == 'scheduled':
                    should, reasons = True, ['scheduled']
                if should and not non_adaptive and forecaster.retrain_count < args_.max_retrains:
                    win_start = max(0, i - args_.retrain_window)
                    forecaster._retrain(data.iloc[win_start:i])
                    forecaster.retrain_count += 1
                    retrain_steps.append(len(history) - 1)
                    print(f'  [{label}] retrain #{forecaster.retrain_count} at window {len(history)}: {reasons}')
                    retrain_events.append({
                        'retrain_no':        int(forecaster.retrain_count),
                        'after_window':      int(len(history) - 1),
                        'forecast_start':    str(data.index[i]),
                        'history_from':      str(data.index[win_start]),
                        'history_rows':      int(i - win_start),
                        'reasons':           [str(r) for r in reasons],
                    })
                    weights_history.append({
                        'after_window': int(len(history) - 1), 'event': 'retrain',
                        'weights': _jsonable(getattr(forecaster, 'horizon_weights', None)),
                    })
                    _lr = getattr(forecaster, 'last_retrain', None) or {}
                    retrain_events[-1].update({
                        k: _jsonable(_lr.get(k)) for k in
                        ('holdout', 'fit_rows', 'holdout_rows', 'weights_retuned', 'accept')})
                    if _lr.get('wins'):
                        retune_wins.append((int(forecaster.retrain_count), _lr['wins']))

            i          += stride
            window_idx += 1

        forecaster.performance_history = history
        return history

    print(f'\n{"="*60}')
    print(f'  Running: {label}')
    print(f'{"="*60}')
    history = _tracked_adaptive_forecast(data)
    print(f'  Completed {len(history)} forecast windows.')
    if not non_adaptive:
        print(f'  Total retrains: {forecaster.retrain_count}')

    timestamps = pd.DatetimeIndex(all_timestamps)
    win_starts_idx = pd.DatetimeIndex(win_starts_ts)

    def _stack(arrs):
        return np.stack(arrs) if arrs else np.empty((0, 0))

    series = {}
    for t in targets:
        per_model = {
            'lstm': _stack(win_preds_lstm[t]),
            'gbdt': _stack(win_preds_gbdt[t]),
        }
        if win_preds_tft[t]:
            per_model['tft'] = _stack(win_preds_tft[t])
        series[t] = {
            'actual':     np.array(all_actuals[t]),
            'pred':       np.array(all_preds[t]),
            'timestamps': timestamps,
            # Per-window arrays, shape (n_windows, horizon). These are what
            # horizon-bucket metrics use, so overlapping windows don't double-count.
            'pred_windows':   _stack(win_preds[t]),
            'actual_windows': _stack(win_actuals[t]),
            'window_starts':  win_starts_idx,
            # Per-base-learner predictions, same (n_windows, horizon) shape as
            # `pred_windows`. Consumed by ensemble_diagnostics.py to show how the
            # adaptive blend stacks up against each constituent in isolation.
            'per_model_pred_windows': per_model,
            # Dump-only extras (see save_hourly_predictions). Kept out of
            # per_model_pred_windows, which the diagnostics iterate over.
            'raw_model_pred_windows': {
                m: _stack(win_raw[m][t]) for m in ('lstm', 'gbdt', 'tft') if win_raw[m][t]},
            'blend_windows':     _stack(win_blend[t]),
            'anchored_windows':  _stack(win_anchored[t]),
            'naive_windows':     _stack(win_naive[t]),
            'gated_windows':     np.asarray(win_gated[t], dtype=bool),
            'anchor_alpha_windows': np.asarray(win_alpha[t], dtype=float),
            'timestamp_windows': _stack(win_ts),
        }

    if output_dir:
        tag = _scenario_tag(label)
        try:
            os.makedirs(output_dir, exist_ok=True)
            _write_run_config(
                os.path.join(output_dir, f'run_config_{tag}.json'),
                label=label, forecaster=forecaster, seed=seed, device=device,
                csv_path=csv_path, model_dir=model_dir,
                stride_hours=stride_hours, start_hour_offset=start_hour_offset,
                n_windows=len(history), retrain_events=retrain_events,
                weights_history=weights_history,
            )
            if save_predictions:
                save_hourly_predictions(
                    series, os.path.join(output_dir, f'predictions_{tag}.csv.gz'))
                if retune_wins:
                    save_retune_predictions(
                        retune_wins, os.path.join(output_dir, f'retune_predictions_{tag}.csv.gz'))
        except Exception as exc:  # never lose a multi-hour run to a dump error
            print(f'  [{label}] WARNING: run artefacts not fully written: {exc}')
    return history, retrain_steps, series, samples


# ---------------------------------------------------------------------------
# Run artefacts: hourly prediction dump + effective configuration
# ---------------------------------------------------------------------------

_NAIVE_PERIOD = 168   # hours; same reference as the MASE denominator


def _code_sha() -> str:
    try:
        from forecasting.utils import code_fingerprint
        return code_fingerprint()
    except Exception:
        return ''


def _scenario_tag(label: str) -> str:
    return str(label).strip().lower().replace('-', '_').replace(' ', '_')


def _jsonable(obj):
    """Recursively convert numpy / pandas / tuple values for json.dump."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        if isinstance(obj, float) and not np.isfinite(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, np.generic):
        return _jsonable(obj.item())
    return str(obj)


def _weekly_naive_window(data: pd.DataFrame, target: str, start_idx: int,
                         horizon: int) -> np.ndarray:
    """Weekly-seasonal-naive forecast issued at ``start_idx``.

    Step h uses the value 168 rows earlier; for horizons beyond one week the
    last observed week is repeated, so no value at or after the forecast
    origin is ever used. NaN where the history is too short.
    """
    out = np.full(horizon, np.nan, dtype=float)
    if start_idx < _NAIVE_PERIOD:
        return out
    col = data[target].to_numpy(dtype=float, copy=False)
    for h in range(horizon):
        out[h] = col[start_idx - _NAIVE_PERIOD + (h % _NAIVE_PERIOD)]
    return out


def build_prediction_frame(series: dict) -> pd.DataFrame:
    """Long table with one row per (target, window, horizon step).

    Columns
    -------
    target, window_idx, window_start, h (1-based), timestamp, actual,
    naive_168h        weekly-seasonal-naive reference,
    lstm, gbdt, tft   raw base-learner forecasts (before any gate),
    blend             convex combination before the weekly-naive anchor,
    anchored          after the anchor, before the physical / inactivity gates,
    ensemble          final forecast (what every reported metric is scored on),
    gate_zeroed       1 when the gates set the whole window to zero.
    Columns with no data for a run (e.g. tft in a two-head run) are omitted.
    """
    frames = []
    for target, s in series.items():
        actual = np.asarray(s.get('actual_windows', np.empty((0, 0))), dtype=float)
        if actual.ndim != 2 or actual.size == 0:
            continue
        n_win, hor = actual.shape
        starts = pd.DatetimeIndex(s['window_starts'])
        cols = {
            'target':       np.repeat(_TAG.get(target, target), n_win * hor),
            'window_idx':   np.repeat(np.arange(n_win), hor),
            'window_start': np.repeat(starts.values, hor),
            'h':            np.tile(np.arange(1, hor + 1), n_win),
        }
        ts = np.asarray(s.get('timestamp_windows', np.empty((0, 0))))
        if ts.shape == (n_win, hor):
            cols['timestamp'] = ts.ravel()
        cols['actual'] = actual.ravel()

        def _add(name, arr):
            arr = np.asarray(arr, dtype=float) if arr is not None else None
            if arr is not None and arr.shape == (n_win, hor):
                cols[name] = arr.ravel()

        _add('naive_168h', s.get('naive_windows'))
        raw = s.get('raw_model_pred_windows') or s.get('per_model_pred_windows') or {}
        for m in ('lstm', 'gbdt', 'tft'):
            _add(m, raw.get(m))
        _add('blend',    s.get('blend_windows'))
        _add('anchored', s.get('anchored_windows'))
        _add('ensemble', s.get('pred_windows'))
        gated = np.asarray(s.get('gated_windows', []), dtype=bool)
        if gated.shape == (n_win,):
            cols['gate_zeroed'] = np.repeat(gated.astype(np.int8), hor)
        alpha = np.asarray(s.get('anchor_alpha_windows', []), dtype=float)
        if alpha.shape == (n_win,):
            cols['anchor_alpha'] = np.repeat(alpha, hor)
        frames.append(pd.DataFrame(cols))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _block_mean(hist: list) -> dict:
    """Mean of each numeric window metric over a block of windows."""
    out: dict = {}
    if not hist:
        return out
    for k in hist[0]:
        vals = [h.get(k) for h in hist]
        vals = [float(v) for v in vals if isinstance(v, (int, float)) and np.isfinite(v)]
        out[k] = float(np.mean(vals)) if vals else float('nan')
    return out


def _run_opts(args) -> dict:
    """Scenario options from the CLI namespace (see parse_args)."""
    members = (args.members or '').strip()
    if getattr(args, 'no_tft', False) and not members:
        members = 'lstm,gbdt'
    return {
        'protocol':            args.protocol,
        'members':             members,
        'anchor_mode':         args.anchor_mode,
        'retune_holdout_frac': args.retune_holdout_frac,
        'retrain_accept':      args.retrain_accept,
        'no_weight_retune':    args.no_weight_retune,
        'gbdt_retrain_mode':   args.gbdt_retrain_mode,
        'weights_mode':        args.weights_mode,
        'roll_window':         args.roll_window,
        'roll_min_windows':    args.roll_min_windows,
        'roll_blocks':         args.roll_blocks,
        'weather_noise':       args.weather_noise,
        'trigger':             args.trigger,
        'trigger_stat':        args.trigger_stat,
        'threshold_factor':    args.threshold_factor,
    }


def save_retune_predictions(retune_wins: list, path: str) -> str | None:
    """Base forecasts replayed at each retrain (the weight re-tune sample).

    One row per (retrain, target, replay window, horizon step). With these a
    different combination rule can be fitted offline at every retrain.
    """
    frames = []
    for retrain_no, wins in retune_wins:
        starts = list(wins.get('start') or [])
        for target, acts in (wins.get('act') or {}).items():
            if not acts:
                continue
            A = np.stack(acts)
            n_win, hor = A.shape
            cols = {
                'retrain_no':   np.repeat(int(retrain_no), n_win * hor),
                'target':       np.repeat(_TAG.get(target, target), n_win * hor),
                'window_start': np.repeat(pd.DatetimeIndex(starts[:n_win]).values, hor),
                'h':            np.tile(np.arange(1, hor + 1), n_win),
                'actual':       A.ravel(),
            }
            for m in ('lstm', 'gbdt', 'tft'):
                w = wins.get(m, {}).get(target) or []
                if len(w) == n_win:
                    cols[m] = np.stack(w).ravel()
            frames.append(pd.DataFrame(cols))
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(path, index=False, float_format='%.6g')
    print(f'  [dump] {len(df):,} rows -> {os.path.basename(path)}')
    return path


def save_hourly_predictions(series: dict, path: str) -> str | None:
    """Write build_prediction_frame(series) to ``path`` (gzip CSV)."""
    df = build_prediction_frame(series)
    if df.empty:
        print('  [dump] no windows to write')
        return None
    df.to_csv(path, index=False, float_format='%.6g')
    print(f'  [dump] {len(df):,} rows -> {os.path.basename(path)}')
    return path


def _write_run_config(path: str, *, label, forecaster, seed, device, csv_path,
                      model_dir, stride_hours, start_hour_offset, n_windows,
                      retrain_events, weights_history) -> None:
    """Store the settings a scenario actually ran with.

    ``effective_args`` is the forecaster's args object after dataset constants
    and CLI overrides were applied: thresholds, retrain window, evaluation
    cadence, GBDT floor, naive-anchor weights, gates, per-target loss settings.
    """
    a = forecaster.args
    payload = {
        'scenario':          label,
        'non_adaptive':      bool(getattr(a, 'non_adaptive', False)),
        'seed':              int(seed),
        'code_sha256':       _code_sha(),
        'device':            str(device),
        'csv_path':          csv_path,
        'model_dir':         model_dir,
        'targets':           list(forecaster.targets),
        'has_tft':           bool(getattr(forecaster, 'tft_models', None)),
        'sequence_length':   int(a.sequence_length),
        'forecast_horizon':  int(a.forecast_horizon),
        'stride_hours':      int(stride_hours) if stride_hours else int(a.forecast_horizon),
        'start_hour_offset': start_hour_offset,
        'n_windows':         int(n_windows),
        'n_retrains':        int(getattr(forecaster, 'retrain_count', 0)),
        'retrain_events':    retrain_events,
        'weights_history':   weights_history,
        'effective_args':    _jsonable({k: v for k, v in vars(a).items()
                                        if not k.startswith('_')}),
    }
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(_jsonable(payload), fh, indent=2)
    print(f'  [dump] effective configuration -> {os.path.basename(path)}')


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------

def aggregate(history: list, series: dict | None = None) -> dict:
    """Mean ± std of each metric across all forecast windows.

    Pooled overrides
    ----------------
    Three metric families have their mean-of-per-window aggregation overridden
    by a pooled cross-window equivalent so the reported numbers are mutually
    consistent and free of small-denominator inflation:

      * MAE — pooled = mean-of-window (|·| is linear), so the value is
        unchanged but the std collapses to 0.
      * RMSE — pooled = sqrt over the entire (window × horizon) residual
        stream rather than the mean of per-window sqrt; the two diverge
        because sqrt is non-linear (Jensen's inequality), and mean-of-window
        RMSE is *systematically smaller* than the pooled RMSE when
        window-level errors vary. Without the override, a comparison_summary
        CSV that reports both an R² (which uses pooled MSE) and an RMSE
        (mean-of-window) would be internally inconsistent.
      * R² — pooled per target from the concatenated y/p across all windows;
        per-window R² is degenerate on zero-heavy targets where SS_tot
        shrinks to nothing.
      * MASE — pooled = (Σ_w Σ_h |y - ŷ|) / (Σ_w Σ_h |y - y_naive|), i.e.
        the ratio of the two summed numerators. The legacy mean-of-per-window
        aggregation was biased upward because it averaged ratios whose
        denominators (per-window naive MAE) varied by orders of magnitude —
        a window with naive MAE of 0.01 kW and model MAE of 5 kW contributes
        a ratio of 500 to the mean, while contributing a residual of 5 kW
        to the pooled sum that gets weighed against the rest of the test
        period's residuals. KSRP run 20260523-082538 per-target bias:
        DHW 3.82 → 2.46 pooled, heating 1.11 → ~0.86 pooled (model actually
        beats naive on the active heating windows). See
        ``pooled_mase_per_target`` for the formula.

    ``std`` is set to 0.0 on every overridden entry since the pooled value is
    a single number, not a per-window mean.

    When *series* is None (rare), the per-window mean-of-window numbers are
    kept for MAE/RMSE/R²; MASE is still overridden because its pooled form
    only needs the per-window numerator/denominator sums that are already on
    the history dicts (no concatenated series needed).
    """
    if not history:
        return {}
    keys = history[0].keys()
    agg = {
        k: {'mean': float(np.nanmean([h[k] for h in history])),
            'std':  float(np.nanstd([h[k]  for h in history]))}
        for k in keys
    }
    present = _present_targets(keys)
    if series is not None:
        pooled = pooled_error_metrics_per_target(series, present)
        for k, v in pooled.items():
            agg[k] = {'mean': v, 'std': 0.0}
    pooled_mase = pooled_mase_per_target(history, present)
    for k, v in pooled_mase.items():
        agg[k] = {'mean': v, 'std': 0.0}
    return agg


# ---------------------------------------------------------------------------
# Print comparison table
# ---------------------------------------------------------------------------

def print_table(agg_fixed: dict, agg_adaptive: dict):
    targets = [(t, _TAG[t]) for t in _present_targets(agg_fixed)]
    # (metric_key, higher_is_better)
    metrics = [('mae', False), ('rmse', False), ('r2', True), ('mase', False)]

    header = f"{'Target':<12} {'Metric':<6}  {'Non-Adaptive':>18}  {'Adaptive':>18}  {'Δ':>10}"
    print(f'\n{"="*len(header)}')
    print(header)
    print(f'{"="*len(header)}')

    for target, tag in targets:
        for metric, higher_better in metrics:
            key = f'{target}_{metric}'
            if key not in agg_fixed:
                continue
            fixed_mean = agg_fixed[key]['mean']
            adap_mean  = agg_adaptive[key]['mean']
            delta      = adap_mean - fixed_mean
            delta_str  = f'{delta:+.4f}'
            better = (delta > 0) if higher_better else (delta < 0)
            marker = ' <' if better else ''
            print(f'{tag:<12} {metric.upper():<6}  {fixed_mean:>18.4f}  {adap_mean:>18.4f}  {delta_str:>10}{marker}')
        print()

    print(f'{"─"*len(header)}')
    for metric, higher_better in metrics:
        key = f'overall_{metric}'
        fixed_mean = agg_fixed.get(key, {}).get('mean', float('nan'))
        adap_mean  = agg_adaptive.get(key, {}).get('mean', float('nan'))
        delta      = adap_mean - fixed_mean
        better = (delta > 0) if higher_better else (delta < 0)
        marker = ' <' if better else ''
        print(f'{"OVERALL":<12} {metric.upper():<6}  {fixed_mean:>18.4f}  {adap_mean:>18.4f}  {f"{delta:+.4f}":>10}{marker}')
    print(f'{"="*len(header)}')
    print('  < = scenario with better value  |  R² higher is better  |  MASE < 1.0 = beats weekly-naive')


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _rolling(vals, window=10):
    return pd.Series(vals).rolling(window, min_periods=1).mean().to_numpy()


def plot_rolling_metrics(hist_fixed, hist_adaptive, retrain_steps, output_dir):
    """Per-target rolling MAE / RMSE for both scenarios — one PNG per metric.

    R² is intentionally omitted: per-window R² explodes on zero-heavy windows
    (off-season heating/cooling, DHW) and a rolling mean of those values is
    dominated by the outliers. The pooled R² reported in the metrics table is
    the trustworthy summary.
    """
    metrics = [
        ('mae',  'MAE',  'MAE (kW)'),
        ('rmse', 'RMSE', 'RMSE (kW)'),
    ]
    targets = [(t, _TAG[t]) for t in
               (_present_targets(hist_fixed[0]) if hist_fixed else _ALL_TARGETS)]

    for suffix, title, ylabel in metrics:
        fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
        axes = axes.flatten()

        for ax, (target, tag) in zip(axes, targets):
            key = f'{target}_{suffix}'
            fixed_vals    = _rolling([h.get(key, np.nan) for h in hist_fixed])
            adaptive_vals = _rolling([h.get(key, np.nan) for h in hist_adaptive])

            ax.plot(fixed_vals,    label='Non-adaptive', color='steelblue',  linewidth=1.2)
            ax.plot(adaptive_vals, label='Adaptive',     color='tomato',     linewidth=1.2, linestyle='--')

            for s in retrain_steps:
                ax.axvline(x=s, color='grey', linewidth=0.6, linestyle=':', alpha=0.7)

            ax.set_title(f'{tag} — Rolling {title} (window=10)')
            ax.set_ylabel(ylabel)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel('Forecast Window')
        axes[-2].set_xlabel('Forecast Window')
        plt.suptitle(f'Rolling {title}: Adaptive vs Non-Adaptive', fontsize=13)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'rolling_{suffix}.png'), dpi=120)
        plt.close(fig)
        print(f'  Saved rolling_{suffix}.png')


def plot_mase_history(hist_fixed, hist_adaptive, output_dir):
    """MASE over forecast windows for each target (reference line at 1.0 = weekly naive)."""
    targets = [(t, _TAG[t]) for t in
               (_present_targets(hist_fixed[0]) if hist_fixed else _ALL_TARGETS)]
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
    axes = axes.flatten()

    for ax, (target, tag) in zip(axes, targets):
        key = f'{target}_mase'
        fixed_vals    = _rolling([h.get(key, np.nan) for h in hist_fixed])
        adaptive_vals = _rolling([h.get(key, np.nan) for h in hist_adaptive])

        ax.plot(fixed_vals,    label='Non-adaptive', color='steelblue', linewidth=1.2)
        ax.plot(adaptive_vals, label='Adaptive',     color='tomato',    linewidth=1.2, linestyle='--')
        ax.axhline(y=1.0, color='black', linewidth=0.8, linestyle='--', alpha=0.6,
                   label='Weekly-naive baseline')

        ax.set_title(f'{tag} — Rolling MASE (window=10)')
        ax.set_ylabel('MASE')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Forecast Window')
    axes[-2].set_xlabel('Forecast Window')
    plt.suptitle('Rolling MASE: Adaptive vs Non-Adaptive  '
                 '(< 1.0 = beats weekly-naive)', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rolling_mase.png'), dpi=120)
    plt.close(fig)
    print(f'  Saved rolling_mase.png')


def plot_summary_bars(agg_fixed, agg_adaptive, output_dir):
    """Side-by-side bar chart of mean MAE and MASE per target."""
    present      = _present_targets(agg_fixed)
    tags         = [_TAG[t] for t in present]
    mae_fixed    = [agg_fixed.get(f'{t}_mae',   {}).get('mean', 0) for t in present]
    mae_adaptive = [agg_adaptive.get(f'{t}_mae', {}).get('mean', 0) for t in present]
    mase_fixed   = [agg_fixed.get(f'{t}_mase',   {}).get('mean', np.nan) for t in present]
    mase_adaptive= [agg_adaptive.get(f'{t}_mase', {}).get('mean', np.nan) for t in present]

    x   = np.arange(len(tags))
    w   = 0.35
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.bar(x - w/2, mae_fixed,    w, label='Non-adaptive', color='steelblue', alpha=0.8)
    ax1.bar(x + w/2, mae_adaptive, w, label='Adaptive',     color='tomato',    alpha=0.8)
    ax1.set_xticks(x); ax1.set_xticklabels(tags)
    ax1.set_ylabel('Mean MAE (kW)')
    ax1.set_title('Mean MAE per Target')
    ax1.legend(); ax1.grid(True, axis='y', alpha=0.3)

    ax2.bar(x - w/2, mase_fixed,   w, label='Non-adaptive', color='steelblue', alpha=0.8)
    ax2.bar(x + w/2, mase_adaptive,w, label='Adaptive',     color='tomato',    alpha=0.8)
    ax2.axhline(y=1.0, color='black', linewidth=0.8, linestyle='--', alpha=0.6,
                label='Weekly-naive (MASE=1)')
    ax2.set_xticks(x); ax2.set_xticklabels(tags)
    ax2.set_ylabel('Mean MASE')
    ax2.set_title('Mean MASE per Target  (< 1 = beats weekly naive)')
    ax2.legend(); ax2.grid(True, axis='y', alpha=0.3)

    plt.suptitle('Summary: Adaptive vs Non-Adaptive', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'summary_bars.png'), dpi=120)
    plt.close(fig)
    print(f'  Saved summary_bars.png')


def plot_forecast_samples(samples: list, output_dir: str):
    """
    One PNG per sample window — 4 stacked panels (one per target).

    Style:
      - Context zone  : light grey background + thin grey line + grey fill under curve
      - Forecast zone : light blue background + actual fill_between + model lines
      - Vertical dashed separator at forecast start
    """
    sample_dir = os.path.join(output_dir, 'forecast_samples')
    os.makedirs(sample_dir, exist_ok=True)

    if not samples:
        return
    targets = [(t, _TAG[t]) for t in _present_targets(samples[0]['actual'])]

    _CTX_BG   = '#f5f5f5'   # light grey — context zone background
    _FC_BG    = '#eef4fb'   # light blue — forecast zone background
    _CTX_LINE = '#999999'   # grey line for context actual
    _CTX_FILL = '#cccccc'   # grey fill under context

    for s in samples:
        label      = s['label']
        win_idx    = s['window_idx']
        ts_ctx     = s['ts_context']
        ts_fc      = s['ts_forecast']

        fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=False)
        fig.suptitle(
            f'{label} — Window {win_idx}  |  '
            f'Context: {ts_ctx[0].strftime("%Y-%m-%d %H:%M")} → {ts_ctx[-1].strftime("%Y-%m-%d %H:%M")}  |  '
            f'Forecast: {ts_fc[0].strftime("%Y-%m-%d %H:%M")} → {ts_fc[-1].strftime("%Y-%m-%d %H:%M")}',
            fontsize=10,
        )

        for ax, (target, tag) in zip(axes, targets):
            ctx_vals  = s['context'][target]
            act_vals  = s['actual'][target]
            lstm_vals = s['lstm'][target]
            gbdt_vals = s['gbdt'][target]
            tft_vals  = s.get('tft', {}).get(target)
            comb_vals = s['combined'][target]

            # Zone backgrounds
            ax.axvspan(ts_ctx[0], ts_ctx[-1], color=_CTX_BG, alpha=1.0, zorder=0)
            ax.axvspan(ts_fc[0],  ts_fc[-1],  color=_FC_BG,  alpha=1.0, zorder=0)

            # Separator
            ax.axvline(x=ts_fc[0], color='#555555', linewidth=0.8, linestyle='--', alpha=0.6, zorder=3)

            # Context: fill + thin line
            ax.fill_between(ts_ctx, 0, ctx_vals, color=_CTX_FILL, alpha=0.55, zorder=1)
            ax.plot(ts_ctx, ctx_vals, color=_CTX_LINE, linewidth=0.7, zorder=2, label='Context')

            # Forecast actual: fill + line
            ax.fill_between(ts_fc, 0, act_vals, color='steelblue', alpha=0.18, zorder=1)
            ax.plot(ts_fc, act_vals, color='steelblue', linewidth=1.4, zorder=4, label='Actual')

            # Model forecasts: thin lines
            ax.plot(ts_fc, lstm_vals, color='darkorange', linewidth=1.0,
                    linestyle='--', zorder=5, label='LSTM-CNN')
            ax.plot(ts_fc, gbdt_vals, color='seagreen',   linewidth=1.0,
                    linestyle='--', zorder=5, label='GBDT')
            if tft_vals is not None:
                ax.plot(ts_fc, tft_vals, color='mediumpurple', linewidth=1.0,
                        linestyle='--', zorder=5, label='TFT')
            ax.plot(ts_fc, comb_vals, color='crimson',    linewidth=1.4,
                    linestyle='-',  zorder=6, label='Combined')

            ax.set_ylabel('kW', fontsize=8)
            ax.set_title(tag, fontsize=9, loc='left', pad=2)
            ax.legend(fontsize=7, loc='upper right', ncol=6, framealpha=0.85)
            ax.grid(True, alpha=0.2, zorder=0)
            ax.set_xlim(ts_ctx[0], ts_fc[-1])

        axes[-1].set_xlabel('Date / Hour')
        plt.tight_layout()

        fname = f'sample_{label.lower().replace(" ", "_")}_w{win_idx:04d}.png'
        plt.savefig(os.path.join(sample_dir, fname), dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f'  Saved forecast_samples/{fname}')


def plot_forecasts(series_fixed, series_adaptive, retrain_steps, output_dir,
                   horizon: int = FORECAST_HORIZON):
    """
    One figure per target: actual vs non-adaptive vs adaptive predictions
    over the full test period. Retrain events marked as vertical lines.
    """
    targets = [(t, _TAG[t]) for t in _present_targets(series_fixed)]
    for target, tag in targets:
        sf = series_fixed[target]
        sa = series_adaptive[target]
        ts = sa['timestamps']   # use adaptive timestamps (same for both)

        fig, ax = plt.subplots(figsize=(18, 4))
        ax.plot(ts, sf['actual'],   label='Actual',       color='steelblue', linewidth=1.0)
        ax.plot(ts, sf['pred'],     label='Non-adaptive', color='tomato',    linewidth=0.8,
                linestyle='--', alpha=0.85)
        ax.plot(ts, sa['pred'],     label='Adaptive',     color='seagreen',  linewidth=0.8,
                linestyle=':',  alpha=0.85)

        # Mark retrain events on the time axis
        # (`horizon` is now passed in by the caller — it equals the trained model's window.)
        args_start = series_adaptive[target]['timestamps'][0]
        for step_idx in retrain_steps:
            # step_idx is window index; convert to timestamp position
            flat_idx = step_idx * horizon
            if flat_idx < len(ts):
                ax.axvline(x=ts[flat_idx], color='grey', linewidth=0.7,
                           linestyle='--', alpha=0.6,
                           label='Retrain' if step_idx == retrain_steps[0] else None)

        ax.set_title(f'{tag} — Actual vs Predicted (full test period)')
        ax.set_ylabel('kW')
        ax.set_xlabel('Date')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.25)
        plt.tight_layout()

        path = os.path.join(output_dir, f'forecast_{tag}.png')
        plt.savefig(path, dpi=120)
        plt.close(fig)
        print(f'  Saved forecast_{tag}.png')


def plot_retrain_events(hist_adaptive, retrain_steps, output_dir):
    """Overall MAE time-series with retrain event markers."""
    vals = [h.get('overall_mae', np.nan) for h in hist_adaptive]
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(_rolling(vals), color='tomato', linewidth=1.2, label='Adaptive rolling MAE')

    for s in retrain_steps:
        ax.axvline(x=s, color='grey', linewidth=0.8, linestyle='--', alpha=0.8,
                   label='Retrain' if s == retrain_steps[0] else None)

    ax.set_title(f'Adaptive — Overall MAE with Retrain Events ({len(retrain_steps)} retrains)')
    ax.set_xlabel('Forecast Window')
    ax.set_ylabel('Overall MAE (kW)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'retrain_events.png'), dpi=120)
    plt.close(fig)
    print(f'  Saved retrain_events.png')


# ---------------------------------------------------------------------------
# Save CSV summary
# ---------------------------------------------------------------------------

def save_summary_csv(agg_fixed, agg_adaptive, output_dir):
    rows = []
    for t in _present_targets(agg_fixed):
        tag = _TAG[t]
        for metric in ('mae', 'rmse', 'r2', 'mase'):
            key = f'{t}_{metric}'
            rows.append({
                'target':              tag,
                'metric':              metric.upper(),
                'non_adaptive_mean':   round(agg_fixed.get(key,    {}).get('mean', np.nan), 5),
                'non_adaptive_std':    round(agg_fixed.get(key,    {}).get('std',  np.nan), 5),
                'adaptive_mean':       round(agg_adaptive.get(key, {}).get('mean', np.nan), 5),
                'adaptive_std':        round(agg_adaptive.get(key, {}).get('std',  np.nan), 5),
            })
    for metric in ('mae', 'rmse', 'r2', 'mase'):
        key = f'overall_{metric}'
        rows.append({
            'target':              'OVERALL',
            'metric':              metric.upper(),
            'non_adaptive_mean':   round(agg_fixed.get(key,    {}).get('mean', np.nan), 5),
            'non_adaptive_std':    round(agg_fixed.get(key,    {}).get('std',  np.nan), 5),
            'adaptive_mean':       round(agg_adaptive.get(key, {}).get('mean', np.nan), 5),
            'adaptive_std':        round(agg_adaptive.get(key, {}).get('std',  np.nan), 5),
        })
    path = os.path.join(output_dir, 'comparison_summary.csv')
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f'  Saved comparison_summary.csv')


# ---------------------------------------------------------------------------
# Overview chart  (thermal = heating + cooling + dhw, electric)
# ---------------------------------------------------------------------------

def plot_forecast_overview(series_fixed: dict, output_dir: str,
                           label_fixed: str = 'Non-Adaptive',
                           series_adaptive: dict = None,
                           label_adaptive: str = 'Adaptive'):
    """
    Two-panel overview chart mirroring targets_overview.png style:
      - Top    : Thermal demand (heating + cooling + dhw) — thin hourly line +
                 7-day rolling mean with shaded fill for actuals; dashed/dotted
                 lines for each forecast scenario.
      - Bottom : Electric demand — same treatment.

    Works for both single-scenario (series_adaptive=None) and comparison mode.
    """
    _roll = lambda arr: pd.Series(arr).rolling(168, min_periods=1).mean().to_numpy()

    # Only the thermal channels this dataset actually provides (HEEW has no DHW).
    thermal_targets = [t for t in (TARGET_THERMAL_HEATING, TARGET_THERMAL_COOLING, TARGET_DHW)
                       if t in series_fixed]
    ts = series_fixed[thermal_targets[0]]['timestamps']

    actual_thermal  = sum(series_fixed[t]['actual'] for t in thermal_targets)
    actual_electric = series_fixed[TARGET_ELECTRIC]['actual']
    pred_fixed_th   = sum(series_fixed[t]['pred']   for t in thermal_targets)
    pred_fixed_el   = series_fixed[TARGET_ELECTRIC]['pred']

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 8), sharex=True)
    fig.suptitle('Consumption Forecast Overview — Test Period', fontsize=13)

    # --- Thermal panel ---
    rm_th = _roll(actual_thermal)
    ax1.plot(ts, actual_thermal, color='#ffaaaa', linewidth=0.4, alpha=0.5)
    ax1.fill_between(ts, 0, rm_th, alpha=0.20, color='tomato')
    ax1.plot(ts, rm_th, color='tomato', linewidth=1.4, label='Actual (7-day mean)')
    ax1.plot(ts, _roll(pred_fixed_th), color='steelblue', linewidth=1.2,
             linestyle='--', label=f'{label_fixed} (7-day mean)')
    if series_adaptive is not None:
        pred_adap_th = sum(series_adaptive[t]['pred'] for t in thermal_targets)
        ax1.plot(ts, _roll(pred_adap_th), color='seagreen', linewidth=1.2,
                 linestyle=':', label=f'{label_adaptive} (7-day mean)')
    ax1.set_ylabel('Thermal Demand (kW)')
    ax1.legend(fontsize=8, loc='upper right')
    ax1.grid(True, alpha=0.25)

    # --- Electric panel ---
    rm_el = _roll(actual_electric)
    ax2.plot(ts, actual_electric, color='#aaaaff', linewidth=0.4, alpha=0.5)
    ax2.fill_between(ts, 0, rm_el, alpha=0.20, color='steelblue')
    ax2.plot(ts, rm_el, color='steelblue', linewidth=1.4, label='Actual (7-day mean)')
    ax2.plot(ts, _roll(pred_fixed_el), color='tomato', linewidth=1.2,
             linestyle='--', label=f'{label_fixed} (7-day mean)')
    if series_adaptive is not None:
        pred_adap_el = series_adaptive[TARGET_ELECTRIC]['pred']
        ax2.plot(ts, _roll(pred_adap_el), color='seagreen', linewidth=1.2,
                 linestyle=':', label=f'{label_adaptive} (7-day mean)')
    ax2.set_ylabel('Electric Demand (kW)')
    ax2.set_xlabel('Date')
    ax2.legend(fontsize=8, loc='upper right')
    ax2.grid(True, alpha=0.25)

    plt.tight_layout()
    path = os.path.join(output_dir, 'forecast_overview.png')
    plt.savefig(path, dpi=120)
    plt.close(fig)
    print(f'  Saved forecast_overview.png')


# ---------------------------------------------------------------------------
# Single-scenario output helpers
# ---------------------------------------------------------------------------

def print_single_table(agg: dict, label: str):
    """Metric table for a single scenario (no comparison column)."""
    targets = [(t, _TAG[t]) for t in _present_targets(agg)]
    metrics = ['mae', 'rmse', 'r2', 'mase']
    header  = f"{'Target':<12} {'Metric':<6}  {label:>22}"
    print(f'\n{"="*len(header)}')
    print(header)
    print(f'{"="*len(header)}')
    for target, tag in targets:
        for metric in metrics:
            key = f'{target}_{metric}'
            if key not in agg:
                continue
            print(f'{tag:<12} {metric.upper():<6}  {agg[key]["mean"]:>22.4f}')
        print()
    print(f'{"─"*len(header)}')
    for metric in metrics:
        key = f'overall_{metric}'
        val = agg.get(key, {}).get('mean', float('nan'))
        print(f'{"OVERALL":<12} {metric.upper():<6}  {val:>22.4f}')
    print(f'{"="*len(header)}')
    print('  R² higher is better  |  MASE < 1.0 = beats weekly-naive baseline')


def plot_forecast_single(series: dict, label: str, output_dir: str):
    """Full test-period time-series plot for a single scenario."""
    targets = [(t, _TAG[t]) for t in _present_targets(series)]
    for target, tag in targets:
        s  = series[target]
        ts = s['timestamps']
        fig, ax = plt.subplots(figsize=(18, 4))
        ax.plot(ts, s['actual'], label='Actual',    color='steelblue', linewidth=1.0)
        ax.plot(ts, s['pred'],   label=label,        color='tomato',    linewidth=0.8,
                linestyle='--', alpha=0.85)
        ax.set_title(f'{tag} — Actual vs Predicted (full test period)')
        ax.set_ylabel('kW')
        ax.set_xlabel('Date')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.25)
        plt.tight_layout()
        path = os.path.join(output_dir, f'forecast_{tag}.png')
        plt.savefig(path, dpi=120)
        plt.close(fig)
        print(f'  Saved forecast_{tag}.png')


# ---------------------------------------------------------------------------
# metrics.json emission (article-ready)
# ---------------------------------------------------------------------------

def _safe_mean(x):
    arr = np.asarray(x, dtype=float)
    return float(np.nanmean(arr)) if arr.size and not np.all(np.isnan(arr)) else float('nan')


def _safe_std(x):
    arr = np.asarray(x, dtype=float)
    return float(np.nanstd(arr)) if arr.size and not np.all(np.isnan(arr)) else float('nan')


def _per_target_stats(history: list, key_prefix: str,
                      series: dict | None = None) -> dict:
    """Extract mean/std/MASE-beats-rate per target from the per-window history.

    R² is pooled from *series* (concatenated y/p across all windows) when
    provided — per-window R² is not meaningful on zero-heavy targets. The
    ``r2_std`` field is 0.0 in that case since the pooled value is a single
    number, not a per-window mean.
    """
    present = _present_targets(history[0]) if history else _ALL_TARGETS
    pooled = pooled_r2_per_target(series, present) if series is not None else {}
    stats = {}
    for target in present:
        tag = _TAG[target]
        maes  = [h.get(f'{target}_mae')  for h in history]
        rmses = [h.get(f'{target}_rmse') for h in history]
        # Pooled cross-window MASE: sum-of-per-window numerator / sum-of-per-window
        # denominator. Equivalent to weighting every horizon-step error equally
        # regardless of which window it came from. Replaces the legacy
        # mean-of-per-window-ratios aggregation, which was biased upward whenever
        # the per-window naive MAE varied (heating in summer, DHW post-2016
        # regime change on KSRP) — KSRP run 20260523-082538 reported DHW MASE
        # 3.82 mean-of-ratios where the pooled value is 2.46 (~55 % inflation),
        # and reported heating 1.11 where the pooled value is ~0.86 (model
        # actually *beats* naive on the active heating windows).
        num_sums = [h.get(f'{target}_mase_num_sum') for h in history]
        den_sums = [h.get(f'{target}_mase_den_sum') for h in history]
        num_valid = [v for v in num_sums if v is not None and not np.isnan(v)]
        den_valid = [v for v in den_sums if v is not None and not np.isnan(v)]
        if num_valid and den_valid:
            total_num = float(np.sum(num_valid))
            total_den = float(np.sum(den_valid))
            mase_pooled = (total_num / total_den) if total_den > 1e-8 else float('nan')
        else:
            mase_pooled = float('nan')
        # The per-window-ratio list is still useful for two things:
        #   (i) the "beats naive" win-rate (the fraction of windows where the
        #       model beats the seasonal-naive baseline on a per-window basis,
        #       which is qualitatively different from "is the pooled ratio
        #       below 1"), and
        #   (ii) the rolling-MASE plot, which needs per-window granularity.
        # We retain it but report it as ``mase_per_window_mean`` to make it
        # clear that the headline ``mase_mean`` is the pooled value.
        mases = [h.get(f'{target}_mase') for h in history]
        mases_valid = [v for v in mases if v is not None and not np.isnan(v)]
        beats_rate = (100.0 * float(np.mean([v < 1.0 for v in mases_valid]))
                      if mases_valid else float('nan'))
        if series is not None:
            r2_mean = pooled.get(f'{target}_r2', float('nan'))
            r2_std  = 0.0
        else:
            r2s = [h.get(f'{target}_r2') for h in history]
            r2s_valid = [v for v in r2s if v is not None and not np.isnan(v)]
            r2_mean = _safe_mean(r2s_valid) if r2s_valid else float('nan')
            r2_std  = _safe_std(r2s_valid)  if r2s_valid else float('nan')
        stats[tag] = {
            f'{key_prefix}mae_mean':                  _safe_mean(maes),
            f'{key_prefix}mae_std':                   _safe_std(maes),
            f'{key_prefix}rmse_mean':                 _safe_mean(rmses),
            f'{key_prefix}rmse_std':                  _safe_std(rmses),
            f'{key_prefix}r2_mean':                   r2_mean,
            f'{key_prefix}r2_std':                    r2_std,
            f'{key_prefix}mase_mean':                 mase_pooled,
            f'{key_prefix}mase_std':                  0.0,
            f'{key_prefix}mase_per_window_mean':      _safe_mean(mases_valid) if mases_valid else float('nan'),
            f'{key_prefix}mase_per_window_std':       _safe_std(mases_valid)  if mases_valid else float('nan'),
            f'{key_prefix}mase_beats_naive_rate':     beats_rate,
            f'{key_prefix}windows':                   int(sum(1 for v in maes if v is not None)),
        }
    return stats


def _merge_target_stats(*groups: dict) -> dict:
    merged: dict = {}
    for group in groups:
        for tag, values in group.items():
            merged.setdefault(tag, {}).update(values)
    return merged


def _add_adaptive_vs_fixed(stats: dict) -> None:
    """In-place: add Δ MAE (adaptive − fixed) and % improvement per target."""
    for tag, values in stats.items():
        fixed_mae = values.get('fixed_mae_mean')
        adap_mae  = values.get('adaptive_mae_mean')
        if fixed_mae is None or adap_mae is None:
            continue
        if np.isnan(fixed_mae) or np.isnan(adap_mae):
            continue
        delta = float(adap_mae - fixed_mae)
        values['adaptive_vs_fixed_mae'] = delta
        values['adaptive_vs_fixed_pct'] = (
            float(-delta / fixed_mae * 100.0) if fixed_mae else float('nan')
        )


def _horizon_bucket_mae(series_by_target: dict, horizon: int = FORECAST_HORIZON) -> dict:
    """
    Split each target's per-window prediction/actual series into 3 horizon
    thirds and return MAE per bucket.

    Bucket KEYS are horizon-aware: ``mae_h{a}_{b}`` where (a, b) are the
    1-indexed bucket boundaries derived from the actual horizon. For the
    default 72-h horizon this yields the canonical mae_h1_24 / mae_h25_48 /
    mae_h49_72 keys; for a 24-h horizon you'd get mae_h1_8 / mae_h9_16 /
    mae_h17_24. The function also emits horizon-agnostic aliases
    ``mae_horizon_first_third`` / ``mae_horizon_mid_third`` /
    ``mae_horizon_last_third`` so consumers don't have to know the horizon.

    Prefers the per-window matrices ``pred_windows`` / ``actual_windows`` when
    they are present (overlap-safe under arbitrary stride). Falls back to
    reshaping the flat concatenated series for legacy callers — that fallback
    double-counts hours when stride < horizon.
    """
    out: dict = {}
    for target, s in series_by_target.items():
        tag    = _TAG[target]
        pred_w = s.get('pred_windows')
        act_w  = s.get('actual_windows')
        if pred_w is not None and act_w is not None and getattr(pred_w, 'size', 0) and getattr(act_w, 'size', 0):
            pred = np.asarray(pred_w)
            act  = np.asarray(act_w)
            h    = pred.shape[1]
        else:
            pred = np.asarray(s['pred'])
            act  = np.asarray(s['actual'])
            n    = len(pred)
            if n == 0:
                continue
            n_windows = n // horizon
            if n_windows == 0:
                continue
            pred = pred[:n_windows * horizon].reshape(n_windows, horizon)
            act  = act [:n_windows * horizon].reshape(n_windows, horizon)
            h    = horizon
        abs_err = np.abs(pred - act)
        third  = max(h // 3, 1)
        bounds = [(0, third), (third, 2 * third), (2 * third, h)]
        alias  = ['mae_horizon_first_third', 'mae_horizon_mid_third', 'mae_horizon_last_third']
        row: dict = {}
        for (start, end), key in zip(bounds, alias):
            mae_val = float(np.nanmean(abs_err[:, start:end]))
            row[f'mae_h{start + 1}_{end}'] = mae_val
            row[key]                       = mae_val
        out[tag] = row
    return out


def write_metrics_json(
    output_dir: str,
    *,
    hist_fixed: list,
    hist_adaptive: list | None,
    series_fixed: dict,
    series_adaptive: dict | None,
    retrain_steps: list,
    csv_path: str,
    n_samples: int,
    total_seconds: float,
    model_dir: str | None,
    non_adaptive_only: bool,
    horizon: int = FORECAST_HORIZON,
    stride_hours: int | None = None,
    energy_fixed: dict | None = None,
    energy_adaptive: dict | None = None,
) -> None:
    """Emit a rich metrics.json under *output_dir* for orchestrator + articles."""
    scenarios = ['non_adaptive'] if non_adaptive_only else ['non_adaptive', 'adaptive']

    fixed_stats = _per_target_stats(hist_fixed, key_prefix='fixed_',
                                    series=series_fixed)
    groups = [fixed_stats]

    if hist_adaptive is not None:
        adaptive_stats = _per_target_stats(hist_adaptive, key_prefix='adaptive_',
                                           series=series_adaptive)
        groups.append(adaptive_stats)

    # Horizon buckets for each scenario's full series
    horizon_fixed = _horizon_bucket_mae(series_fixed, horizon)
    groups.append({tag: {f'fixed_{k}': v for k, v in b.items()}
                   for tag, b in horizon_fixed.items()})
    if series_adaptive is not None:
        horizon_adaptive = _horizon_bucket_mae(series_adaptive, horizon)
        groups.append({tag: {f'adaptive_{k}': v for k, v in b.items()}
                       for tag, b in horizon_adaptive.items()})

    merged = _merge_target_stats(*groups)

    # Rename fixed_* keys to the canonical "mae_mean" set expected by the UI
    # schema when running in non-adaptive-only mode (no "adaptive" column).
    if non_adaptive_only:
        canonical: dict = {}
        for tag, values in merged.items():
            row = {}
            for k, v in values.items():
                if k.startswith('fixed_'):
                    row[k[len('fixed_'):]] = v
                else:
                    row[k] = v
            canonical[tag] = row
        merged = canonical
    else:
        _add_adaptive_vs_fixed(merged)

    if energy_fixed:
        _inject_energy_into_merged(
            merged, energy_fixed, energy_adaptive,
            non_adaptive_only=non_adaptive_only,
        )

    meta: dict = {
        'n_windows':       max(len(hist_fixed), len(hist_adaptive) if hist_adaptive else 0),
        'retrain_count':   len(retrain_steps),
        'retrain_windows': list(retrain_steps),
        'forecast_horizon': int(horizon),
        'window_stride':   int(stride_hours) if stride_hours and stride_hours > 0 else int(horizon),
        'total_seconds':   float(total_seconds),
        'n_sample_plots':  int(n_samples),
        'input_csv':       os.path.abspath(csv_path),
        'model_dir':       os.path.abspath(model_dir) if model_dir else None,
        'scenarios':       scenarios,
    }

    # Sanitise NaNs for strict JSON decoders downstream.
    def _clean(v):
        if isinstance(v, float) and np.isnan(v):
            return None
        if isinstance(v, dict):
            return {k: _clean(x) for k, x in v.items()}
        if isinstance(v, list):
            return [_clean(x) for x in v]
        return v

    payload = _clean({**merged, '_meta': meta})
    out_path = os.path.join(output_dir, 'metrics.json')
    with open(out_path, 'w', encoding='utf-8') as fh:
        json.dump(payload, fh, indent=2)
    print(f'  Saved metrics.json → {out_path}')


# ---------------------------------------------------------------------------
# Energy (kWh) — per-step energy across the horizon and per-window sums.
# Hourly data => kWh per step is numerically equal to kW per step (×1 h),
# but we surface it explicitly so units, totals, and comparison charts read
# as energy rather than instantaneous power.
# ---------------------------------------------------------------------------

def _energy_stats(series_by_target: dict) -> dict:
    """Compute kWh-domain stats per target from per-window arrays.

    Returns dict keyed by tag with:
      mean_actual_kwh_per_step / mean_pred_kwh_per_step : np.ndarray(horizon,)
      actual_kwh_total_per_window / pred_kwh_total_per_window : np.ndarray(n_windows,)
      mean_actual_kwh_total / mean_pred_kwh_total : float (mean across windows)
      total_actual_kwh / total_pred_kwh : float (sum across all windows)
      window_starts : DatetimeIndex
    """
    out: dict = {}
    for target, s in series_by_target.items():
        tag = _TAG[target]
        pred_w = np.asarray(s.get('pred_windows', np.empty((0, 0))))
        act_w  = np.asarray(s.get('actual_windows', np.empty((0, 0))))
        if pred_w.size == 0 or act_w.size == 0:
            continue
        pred_tot = pred_w.sum(axis=1)
        act_tot  = act_w.sum(axis=1)
        out[tag] = {
            'actual_kwh_per_step':         act_w,
            'pred_kwh_per_step':           pred_w,
            'actual_kwh_total_per_window': act_tot,
            'pred_kwh_total_per_window':   pred_tot,
            'mean_actual_kwh_per_step':    act_w.mean(axis=0),
            'mean_pred_kwh_per_step':      pred_w.mean(axis=0),
            'mean_actual_kwh_total':       float(act_tot.mean()),
            'mean_pred_kwh_total':         float(pred_tot.mean()),
            'std_actual_kwh_total':        float(act_tot.std()),
            'std_pred_kwh_total':          float(pred_tot.std()),
            'total_actual_kwh':            float(act_tot.sum()),
            'total_pred_kwh':              float(pred_tot.sum()),
            'window_starts':               s.get('window_starts'),
        }
    return out


def save_energy_csvs(energy_fixed: dict, output_dir: str,
                     energy_adaptive: dict | None = None,
                     label_fixed: str = 'non_adaptive',
                     label_adaptive: str = 'adaptive') -> None:
    """Two CSVs: per-window horizon-sum kWh, and mean per-step horizon profile."""
    if not energy_fixed:
        return
    # 1) Per-window horizon-sum kWh (long format: one row per target × window)
    rows = []
    for tag, e in energy_fixed.items():
        ws = e['window_starts']
        for i in range(len(e['pred_kwh_total_per_window'])):
            row = {
                'target':                          tag,
                'window_idx':                      i,
                'window_start':                    ws[i] if ws is not None and i < len(ws) else None,
                'actual_kwh_total':                float(e['actual_kwh_total_per_window'][i]),
                f'{label_fixed}_pred_kwh_total':   float(e['pred_kwh_total_per_window'][i]),
            }
            if energy_adaptive and tag in energy_adaptive:
                row[f'{label_adaptive}_pred_kwh_total'] = float(
                    energy_adaptive[tag]['pred_kwh_total_per_window'][i]
                )
            rows.append(row)
    pd.DataFrame(rows).to_csv(
        os.path.join(output_dir, 'forecast_energy_windows.csv'), index=False
    )
    print('  Saved forecast_energy_windows.csv')

    # 2) Mean per-step horizon profile (long format: one row per target × hour-in-horizon)
    rows = []
    for tag, e in energy_fixed.items():
        actual_prof = e['mean_actual_kwh_per_step']
        pred_prof   = e['mean_pred_kwh_per_step']
        for h in range(len(actual_prof)):
            row = {
                'target':                              tag,
                'hour_in_horizon':                     h + 1,
                'mean_actual_kwh':                     float(actual_prof[h]),
                f'mean_{label_fixed}_pred_kwh':        float(pred_prof[h]),
            }
            if energy_adaptive and tag in energy_adaptive:
                row[f'mean_{label_adaptive}_pred_kwh'] = float(
                    energy_adaptive[tag]['mean_pred_kwh_per_step'][h]
                )
            rows.append(row)
    pd.DataFrame(rows).to_csv(
        os.path.join(output_dir, 'forecast_energy_horizon_profile.csv'), index=False
    )
    print('  Saved forecast_energy_horizon_profile.csv')


def plot_energy_horizon_profile(energy_fixed: dict, output_dir: str,
                                energy_adaptive: dict | None = None) -> None:
    """Per-target line plot: mean kWh per hour-in-horizon, actual vs scenario(s)."""
    tags = [_TAG[t] for t in _ALL_TARGETS if _TAG[t] in energy_fixed]
    if not tags:
        return
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
    axes = axes.flatten()
    for ax, tag in zip(axes, tags):
        e = energy_fixed[tag]
        h = len(e['mean_actual_kwh_per_step'])
        x = np.arange(1, h + 1)
        ax.plot(x, e['mean_actual_kwh_per_step'], color='steelblue',
                linewidth=1.6, label='Actual')
        ax.fill_between(x, 0, e['mean_actual_kwh_per_step'],
                        color='steelblue', alpha=0.15)
        ax.plot(x, e['mean_pred_kwh_per_step'], color='tomato',
                linewidth=1.2, linestyle='--', label='Non-adaptive')
        if energy_adaptive and tag in energy_adaptive:
            ax.plot(x, energy_adaptive[tag]['mean_pred_kwh_per_step'],
                    color='seagreen', linewidth=1.2, linestyle=':',
                    label='Adaptive')
        ax.set_title(f'{tag} — mean energy by horizon hour')
        ax.set_ylabel('Energy (kWh)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    for ax in axes[-2:]:
        ax.set_xlabel('Hour in forecast horizon')
    plt.suptitle('Mean forecast energy across horizon (kWh per hour)', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'energy_horizon_profile.png'), dpi=120)
    plt.close(fig)
    print('  Saved energy_horizon_profile.png')


def plot_energy_totals(energy_fixed: dict, output_dir: str,
                       energy_adaptive: dict | None = None) -> None:
    """Per-target bar chart: mean kWh per horizon (sum over horizon), actual vs scenario(s)."""
    tags = [_TAG[t] for t in _ALL_TARGETS if _TAG[t] in energy_fixed]
    if not tags:
        return
    n_bars = 3 if energy_adaptive else 2
    w = 0.8 / n_bars
    x = np.arange(len(tags))
    offset = -(n_bars - 1) / 2 * w

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

    # Left panel: mean per-window kWh-per-horizon
    mean_act = [energy_fixed[t]['mean_actual_kwh_total'] for t in tags]
    mean_fix = [energy_fixed[t]['mean_pred_kwh_total']   for t in tags]
    ax1.bar(x + offset,       mean_act, w, label='Actual',       color='steelblue', alpha=0.85)
    ax1.bar(x + offset + w,   mean_fix, w, label='Non-adaptive', color='tomato',    alpha=0.85)
    if energy_adaptive:
        mean_adp = [energy_adaptive[t]['mean_pred_kwh_total'] for t in tags]
        ax1.bar(x + offset + 2 * w, mean_adp, w, label='Adaptive',
                color='seagreen', alpha=0.85)
    ax1.set_xticks(x); ax1.set_xticklabels(tags)
    ax1.set_ylabel('Mean energy per horizon (kWh)')
    ax1.set_title('Mean horizon-sum energy per target')
    ax1.legend(); ax1.grid(True, axis='y', alpha=0.3)

    # Right panel: total kWh summed across ALL forecast windows
    tot_act = [energy_fixed[t]['total_actual_kwh'] for t in tags]
    tot_fix = [energy_fixed[t]['total_pred_kwh']   for t in tags]
    ax2.bar(x + offset,       tot_act, w, label='Actual',       color='steelblue', alpha=0.85)
    ax2.bar(x + offset + w,   tot_fix, w, label='Non-adaptive', color='tomato',    alpha=0.85)
    if energy_adaptive:
        tot_adp = [energy_adaptive[t]['total_pred_kwh'] for t in tags]
        ax2.bar(x + offset + 2 * w, tot_adp, w, label='Adaptive',
                color='seagreen', alpha=0.85)
    ax2.set_xticks(x); ax2.set_xticklabels(tags)
    ax2.set_ylabel('Total energy across all windows (kWh)')
    ax2.set_title('Aggregate forecast energy per target')
    ax2.legend(); ax2.grid(True, axis='y', alpha=0.3)

    plt.suptitle('Forecast energy totals (kWh)', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'energy_totals.png'), dpi=120)
    plt.close(fig)
    print('  Saved energy_totals.png')


def _inject_energy_into_merged(merged: dict, energy_fixed: dict,
                               energy_adaptive: dict | None = None,
                               non_adaptive_only: bool = False) -> None:
    """Add energy (kWh) keys per target to the metrics.json merged dict."""
    for tag, e in energy_fixed.items():
        row = merged.setdefault(tag, {})
        row['mean_actual_kwh_total'] = e['mean_actual_kwh_total']
        row['total_actual_kwh']      = e['total_actual_kwh']
        if non_adaptive_only:
            row['mean_pred_kwh_total'] = e['mean_pred_kwh_total']
            row['total_pred_kwh']      = e['total_pred_kwh']
        else:
            row['mean_fixed_pred_kwh_total'] = e['mean_pred_kwh_total']
            row['total_fixed_pred_kwh']      = e['total_pred_kwh']
            if energy_adaptive and tag in energy_adaptive:
                ea = energy_adaptive[tag]
                row['mean_adaptive_pred_kwh_total'] = ea['mean_pred_kwh_total']
                row['total_adaptive_pred_kwh']      = ea['total_pred_kwh']


def save_summary_csv_single(agg: dict, label: str, output_dir: str):
    """CSV summary for a single scenario."""
    rows = []
    for t in _present_targets(agg):
        tag = _TAG[t]
        for metric in ('mae', 'rmse', 'r2', 'mase'):
            key = f'{t}_{metric}'
            rows.append({
                'target': tag,
                'metric': metric.upper(),
                f'{label}_mean': round(agg.get(key, {}).get('mean', np.nan), 5),
                f'{label}_std':  round(agg.get(key, {}).get('std',  np.nan), 5),
            })
    for metric in ('mae', 'rmse', 'r2', 'mase'):
        key = f'overall_{metric}'
        rows.append({
            'target': 'OVERALL',
            'metric': metric.upper(),
            f'{label}_mean': round(agg.get(key, {}).get('mean', np.nan), 5),
            f'{label}_std':  round(agg.get(key, {}).get('std',  np.nan), 5),
        })
    path = os.path.join(output_dir, 'summary.csv')
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f'  Saved summary.csv')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    plot_i18n.setup_from_args(args)
    os.makedirs(args.output_dir, exist_ok=True)
    from forecasting.utils import resolve_seed
    args.seed = resolve_seed(args.seed)

    t_start = time.time()

    if not os.path.exists(args.csv):
        raise FileNotFoundError(f'Test CSV not found: {args.csv}')

    print(f'Loading data from {args.csv}')
    from forecasting.consumption.constants import LSTM_COLUMNS, TARGET_COLUMNS as _TC
    data = pd.read_csv(args.csv, parse_dates=['date'], index_col='date')

    # Resolve which of the canonical targets this CSV actually provides.
    # Datasets like HEEW omit a channel (no DHW) and never ship that column, so
    # require only that SOME target is present — a CSV with none of them is the
    # real mistake (e.g. a generation/PV file selected by accident).
    available_targets = [c for c in _TC if c in data.columns]
    if not available_targets:
        raise ValueError(
            f'Input CSV has none of the consumption target columns {_TC}. '
            f'Did you select a generation/PV CSV by mistake? '
            f'Use a CSV from forecasting/data/<dataset>/processed/consumption/.'
        )
    if set(available_targets) != set(_TC):
        _missing = [c for c in _TC if c not in available_targets]
        print(f'  Dataset is missing target column(s) {_missing}; '
              f'forecasting {available_targets} only.')

    data = data.dropna(subset=available_targets).ffill().fillna(0)
    print(f'  {len(data)} rows  |  {data.shape[1]} columns')

    # Resolve the trained model's window so plot/metric helpers slice on the
    # actual training horizon instead of the module-level constant.
    _probe_args = create_default_args()
    _probe_args.model_base_dir = args.model_dir
    _, eff_horizon = apply_model_meta(_probe_args)

    # Report per-horizon weights status before launching scenarios. Mirrors the
    # resolution order in AdaptiveForecaster: when --model_dir is set, prefer
    # <model_dir>/weights/weights_per_horizon.json (written by tune_weights.py
    # during the training run); otherwise fall back to the source-tree copy
    # next to this script.
    if args.model_dir:
        _weights_path = os.path.join(args.model_dir, 'weights', 'weights_per_horizon.json')
        _weights_loc  = _weights_path
    else:
        _weights_path = os.path.join(_DIR, 'weights_per_horizon.json')
        _weights_loc  = 'adaptive/weights_per_horizon.json'
    if os.path.exists(_weights_path):
        print(f'\n[weights] Per-horizon ensemble weights loaded from {_weights_loc}')
    else:
        print(f'\n[weights] WARNING: weights_per_horizon.json not found at {_weights_loc}.')
        print('[weights]   Run tune_weights.py first to tune the per-horizon weights.')
        print('[weights]   Falling back to fixed ensemble weights for both scenarios.')

    hist_fixed, _, series_fixed, samples_fixed = run_scenario(
        data, non_adaptive=True, label='Non-Adaptive',
        n_samples=args.n_samples, model_dir=args.model_dir,
        stride_hours=args.window_stride_hours,
        start_hour_offset=args.start_hour_offset,
        csv_path=args.csv,
        mae_threshold=args.mae_threshold,
        rmse_threshold=args.rmse_threshold,
        mase_threshold=args.mase_threshold,
        evaluation_frequency=args.evaluation_frequency,
        retrain_window=args.retrain_window,
        seed=args.seed, output_dir=args.output_dir,
        save_predictions=not args.no_prediction_dump,
        opts=_run_opts(args))
    agg_fixed = aggregate(hist_fixed, series=series_fixed)

    if args.non_adaptive_only:
        print_single_table(agg_fixed, 'Non-Adaptive')
        energy_fixed = _energy_stats(series_fixed)
        if not args.no_plots:
            print(f'\nSaving plots to {args.output_dir}/')
            if args.n_samples > 0:
                plot_forecast_samples(samples_fixed, args.output_dir)
            plot_forecast_single(series_fixed, 'Non-Adaptive', args.output_dir)
            plot_forecast_overview(series_fixed, args.output_dir, label_fixed='Non-Adaptive')
            plot_energy_horizon_profile(energy_fixed, args.output_dir)
            plot_energy_totals(energy_fixed, args.output_dir)

            # --- Extended diagnostics (non-adaptive-only mode) ---------------
            plot_horizon_error(series_fixed, args.output_dir)
            plot_pooled_scatter(series_fixed, args.output_dir)
            plot_load_duration(series_fixed, args.output_dir)
            plot_residual_violins(series_fixed, args.output_dir)
            plot_mae_by_driver(series_fixed, data, args.output_dir,
                               label='non_adaptive')
            plot_error_calendar(series_fixed, args.output_dir,
                                label='non_adaptive')
            plot_peak_event_scatter(series_fixed, args.output_dir)
            plot_daily_energy_scatter(series_fixed, args.output_dir)
            plot_ramp_rate_errors(series_fixed, args.output_dir)
            # Ensemble-weight stacked area: relies on weights_per_horizon.json.
            try:
                if os.path.exists(_weights_path):
                    with open(_weights_path) as _f:
                        _w = json.load(_f)
                    plot_ensemble_weights(_w, args.output_dir)
            except Exception as exc:
                print(f'  [plots] ensemble weights plot skipped: {exc}')

            # Ensemble-vs-singles diagnostics: metrics table, win-rate,
            # error-correlation, oracle gap, regime breakdown. Non-adaptive
            # series still carries per-base-learner predictions because
            # predict_components is now called every window.
            try:
                run_ensemble_diagnostics(series_fixed, args.output_dir,
                                         label='non_adaptive')
            except Exception as exc:
                print(f'  [plots] ensemble diagnostics (non_adaptive) skipped: {exc}')

        save_summary_csv_single(agg_fixed, 'non_adaptive', args.output_dir)
        save_energy_csvs(energy_fixed, args.output_dir)

        write_metrics_json(
            args.output_dir,
            hist_fixed=hist_fixed, hist_adaptive=None,
            series_fixed=series_fixed, series_adaptive=None,
            retrain_steps=[], csv_path=args.csv,
            n_samples=args.n_samples,
            total_seconds=time.time() - t_start,
            model_dir=args.model_dir,
            non_adaptive_only=True,
            horizon=eff_horizon,
            stride_hours=args.window_stride_hours,
            energy_fixed=energy_fixed,
        )
    else:
        hist_adaptive, retrain_steps, series_adaptive, samples_adaptive = run_scenario(
            data, non_adaptive=False, label='Adaptive',
            n_samples=args.n_samples, model_dir=args.model_dir,
            stride_hours=args.window_stride_hours,
            start_hour_offset=args.start_hour_offset,
            csv_path=args.csv,
            mae_threshold=args.mae_threshold,
            rmse_threshold=args.rmse_threshold,
            mase_threshold=args.mase_threshold,
            evaluation_frequency=args.evaluation_frequency,
            retrain_window=args.retrain_window,
            seed=args.seed, output_dir=args.output_dir,
            save_predictions=not args.no_prediction_dump,
        opts=_run_opts(args))
        agg_adaptive = aggregate(hist_adaptive, series=series_adaptive)

        print_table(agg_fixed, agg_adaptive)

        energy_fixed    = _energy_stats(series_fixed)
        energy_adaptive = _energy_stats(series_adaptive)

        if not args.no_plots:
            print(f'\nSaving plots to {args.output_dir}/')
            if args.n_samples > 0:
                plot_forecast_samples(samples_fixed + samples_adaptive, args.output_dir)
            plot_forecasts(series_fixed, series_adaptive, retrain_steps, args.output_dir,
                           horizon=eff_horizon)
            plot_rolling_metrics(hist_fixed, hist_adaptive, retrain_steps, args.output_dir)
            plot_mase_history(hist_fixed, hist_adaptive, args.output_dir)
            plot_summary_bars(agg_fixed, agg_adaptive, args.output_dir)
            plot_forecast_overview(series_fixed, args.output_dir,
                                   label_fixed='Non-Adaptive',
                                   series_adaptive=series_adaptive,
                                   label_adaptive='Adaptive')
            plot_energy_horizon_profile(energy_fixed, args.output_dir,
                                        energy_adaptive=energy_adaptive)
            plot_energy_totals(energy_fixed, args.output_dir,
                               energy_adaptive=energy_adaptive)
            if retrain_steps:
                plot_retrain_events(hist_adaptive, retrain_steps, args.output_dir)

            # --- Extended diagnostics (adaptive vs non-adaptive) -------------
            plot_horizon_error(series_fixed, args.output_dir,
                               series_adaptive=series_adaptive)
            plot_horizon_skill_score(series_fixed, series_adaptive,
                                     args.output_dir)
            plot_pooled_scatter(series_fixed, args.output_dir,
                                series_adaptive=series_adaptive)
            plot_load_duration(series_fixed, args.output_dir,
                               series_adaptive=series_adaptive)
            plot_residual_violins(series_fixed, args.output_dir,
                                  series_adaptive=series_adaptive)
            plot_window_mae_cdf(series_fixed, series_adaptive, args.output_dir)
            plot_mae_by_driver(series_fixed, data, args.output_dir,
                               label='non_adaptive')
            plot_mae_by_driver(series_adaptive, data, args.output_dir,
                               label='adaptive')
            plot_error_calendar(series_fixed, args.output_dir,
                                label='non_adaptive')
            plot_error_calendar(series_adaptive, args.output_dir,
                                label='adaptive')
            plot_peak_event_scatter(series_fixed, args.output_dir,
                                    series_adaptive=series_adaptive)
            plot_daily_energy_scatter(series_fixed, args.output_dir,
                                      series_adaptive=series_adaptive)
            plot_ramp_rate_errors(series_fixed, args.output_dir,
                                  series_adaptive=series_adaptive)
            # Ensemble-weight stacked area: relies on weights_per_horizon.json.
            try:
                if os.path.exists(_weights_path):
                    with open(_weights_path) as _f:
                        _w = json.load(_f)
                    plot_ensemble_weights(_w, args.output_dir)
            except Exception as exc:
                print(f'  [plots] ensemble weights plot skipped: {exc}')

            # Ensemble-vs-singles diagnostics for BOTH scenarios — outputs are
            # tagged ``non_adaptive`` / ``adaptive`` so the file names don't
            # collide.
            for _series, _label in (
                (series_fixed,    'non_adaptive'),
                (series_adaptive, 'adaptive'),
            ):
                try:
                    run_ensemble_diagnostics(_series, args.output_dir, label=_label)
                except Exception as exc:
                    print(f'  [plots] ensemble diagnostics ({_label}) skipped: {exc}')

        save_summary_csv(agg_fixed, agg_adaptive, args.output_dir)
        save_energy_csvs(energy_fixed, args.output_dir,
                         energy_adaptive=energy_adaptive)

        write_metrics_json(
            args.output_dir,
            hist_fixed=hist_fixed, hist_adaptive=hist_adaptive,
            series_fixed=series_fixed, series_adaptive=series_adaptive,
            retrain_steps=retrain_steps, csv_path=args.csv,
            n_samples=args.n_samples,
            total_seconds=time.time() - t_start,
            model_dir=args.model_dir,
            non_adaptive_only=False,
            horizon=eff_horizon,
            stride_hours=args.window_stride_hours,
            energy_fixed=energy_fixed,
            energy_adaptive=energy_adaptive,
        )

    print(f'\nDone. Results in {args.output_dir}/')


if __name__ == '__main__':
    main()
