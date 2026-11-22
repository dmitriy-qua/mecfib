"""Train one LSTM-CNN model per consumption target.

Models
------
  heating  — thermal_heating_kw  (heating season only: Oct-15 → Apr-15)
  cooling  — thermal_cooling_kw  (cooling season only: Apr-15 → Oct-15)
  dhw      — dhw_kw              (all year)
  electric — electric_demand_kw  (all year)

Each model predicts a single target for FORECAST_HORIZON steps ahead.
Input is still the full LSTM_COLUMNS feature set (all 57 columns) so the
model can exploit cross-target history and weather context.
"""

import argparse
import json
import os
import pickle
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_DIR)))))

from forecasting.constants import FORECAST_HORIZON
from forecasting.utils import create_sequences, seed_everything
from forecasting.consumption.constants import (
    LSTM_COLUMNS, TARGET_COLUMNS, SEQUENCE_LENGTH, OPTIONAL_LSTM_COLUMNS,
)
from forecasting.consumption.lstm_cnn.model import (
    LSTM_CNN_Model, CNN_CHANNELS, DEFAULT_KERNEL_SIZES, DEFAULT_POOL_MODE,
    POOL_MODE_ATTENTION, POOL_MODE_ATTENTION_MAX_LAST,
)


def _asymmetric_mse(
    pred: torch.Tensor,
    actual: torch.Tensor,
    over_weight: float = 1.0,
    eps: float = 0.01,
    under_weight: float = 1.0,
    peak_threshold: float = 0.3,
    peak_weight_alpha: float = 0.0,
    loss_type: str = 'mse',
    huber_delta: float = 0.1,
) -> torch.Tensor:
    """Peak-aware regression loss for consumption targets.

    Base loss is per-cell squared error (``loss_type='mse'``) or Huber
    (``loss_type='huber'``, threshold ``huber_delta`` in scaled space). Huber
    has linear tails so rare peaks contribute proportionally less gradient
    noise than MSE while still pulling predictions in the right direction.

    Three multiplicative weights are then combined into ``weight``:

    * **over_weight**  (``actual < eps  and  pred > actual``) — heavier penalty
      on positive errors during ~zero hours. Used per-target so zero-heavy
      DHW can be trained with ``over_weight ~ 2`` while non-zero targets stay
      at 1.0. Values much above 2 bias toward under-prediction on rare active
      hours, defeating the point.
    * **under_weight** (``actual > peak_threshold  and  pred < actual``) —
      heavier penalty on under-prediction during peak hours. The dominant
      failure mode for heating/electric is collapse to the running mean;
      ``under_weight ~ 2`` pulls forecasts back up toward the peak.
    * **peak_weight_alpha** — multiplies a continuous ``(1 + alpha * actual)``
      term so every cell's contribution grows linearly with target magnitude
      (in scaled [0,1] space). Distinct from ``under_weight``: it always-on,
      reshapes the loss landscape rather than gating on a threshold, and lets
      the model see ramps not just isolated peaks. ``alpha ~ 3–5`` is typical.

    With all three weights at 1.0 / 0.0 and ``loss_type='mse'`` this collapses
    back to ordinary MSE.
    """
    err = pred - actual
    if loss_type == 'huber':
        abs_err  = err.abs()
        quad     = 0.5 * err * err
        lin      = huber_delta * (abs_err - 0.5 * huber_delta)
        per_cell = torch.where(abs_err <= huber_delta, quad, lin)
    else:
        per_cell = err * err

    weight = None
    if over_weight != 1.0:
        over_zero = (actual < eps) & (err > 0)
        weight = torch.where(over_zero, torch.full_like(err, over_weight),
                             torch.ones_like(err))
    if under_weight != 1.0:
        under_peak = (actual > peak_threshold) & (err < 0)
        uw = torch.where(under_peak, torch.full_like(err, under_weight),
                         torch.ones_like(err))
        weight = uw if weight is None else weight * uw
    if peak_weight_alpha > 0.0:
        pw = 1.0 + peak_weight_alpha * actual
        weight = pw if weight is None else weight * pw

    if weight is None:
        return per_cell.mean()
    return (weight * per_cell).mean()


def _season_sequences(data_scaled: np.ndarray, season_mask: np.ndarray,
                      seq_len: int, horizon: int):
    """
    Build (X, y) sequences exclusively from contiguous in-season row blocks.

    Sequences are never created across season boundaries — a sequence that
    starts on March 30 and ends on April 2 (straddling the season transition)
    is excluded entirely.  This gives heating/cooling models a completely clean,
    season-pure training set without any mixed-season context windows.

    Parameters
    ----------
    data_scaled : (n_rows, n_features) scaled array for the full dataset
    season_mask : (n_rows,) bool — True for rows belonging to this season
    seq_len     : context window length
    horizon     : forecast horizon

    Returns
    -------
    X : (n_seqs, seq_len, n_features)
    y : (n_seqs, horizon, n_features)
    """
    in_season_idx = np.where(season_mask)[0]
    if len(in_season_idx) == 0:
        n_feat = data_scaled.shape[1]
        return np.empty((0, seq_len, n_feat)), np.empty((0, horizon, n_feat))

    # Split into contiguous blocks (gap > 1 hour = new block)
    breaks      = np.where(np.diff(in_season_idx) > 1)[0] + 1
    block_edges = np.concatenate([[0], breaks, [len(in_season_idx)]])

    X_list, y_list = [], []
    for start, end in zip(block_edges[:-1], block_edges[1:]):
        block_rows = in_season_idx[start:end]
        # create_sequences needs len > seq_len + horizon (strictly) to emit at
        # least one (X, y) pair; a block of exactly seq_len + horizon rows makes
        # it return a 1-D np.array([]) which later breaks the 3-D y[:, :, idx]
        # slice. Require the extra row so every appended array is 3-D.
        if len(block_rows) < seq_len + horizon + 1:
            continue
        block_data = data_scaled[block_rows]
        X_b, y_b  = create_sequences(block_data, seq_len, horizon)
        X_list.append(X_b)
        y_list.append(y_b)

    if not X_list:
        n_feat = data_scaled.shape[1]
        return np.empty((0, seq_len, n_feat)), np.empty((0, horizon, n_feat))

    return np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0)

# Short tag used in filenames
_TAG = {
    'thermal_heating_kw': 'heating',
    'thermal_cooling_kw': 'cooling',
    'dhw_kw':             'dhw',
    'electric_demand_kw': 'electric',
}

# Index of the is_heating_season flag inside LSTM_COLUMNS (used for seasonal filtering)
_IS_HEATING_IDX = LSTM_COLUMNS.index('is_heating_season')

# Seasonal mask: True = keep heating-season sequences, False = cooling, None = all year
_SEASON_FILTER = {
    'thermal_heating_kw': True,
    'thermal_cooling_kw': False,
    'dhw_kw':             None,
    'electric_demand_kw': None,
}


def parse_args():
    p = argparse.ArgumentParser()
    base = os.path.join(os.path.dirname(_DIR), '..', 'data', 'HouseZero', 'processed')
    p.add_argument('csv_path', nargs='?', default=os.path.join(base, 'data_hourly_train.csv'))
    p.add_argument('--features',         type=str,   default='',
                   help='Comma-separated feature column names to use INSTEAD of '
                        'the canonical consumption LSTM_COLUMNS. Missing target '
                        'columns and is_heating_season are auto-prepended (they '
                        'drive the seasonal masks and the per-target output '
                        'slicing) so the caller can pass a pure feature list. '
                        'Leave empty to fall back to the canonical LSTM_COLUMNS.')
    p.add_argument('--targets',          type=str,   default='',
                   help='Comma-separated target columns to train. Pick any '
                        'subset of {' + ','.join(TARGET_COLUMNS) + '}. Leave '
                        'empty to train all four heads.')
    p.add_argument('--sequence_length',  type=int,   default=SEQUENCE_LENGTH)
    p.add_argument('--forecast_horizon', type=int,   default=FORECAST_HORIZON)
    p.add_argument('--learning_rate',    type=float, default=0.001)
    p.add_argument('--epochs',           type=int,   default=100)
    p.add_argument('--patience',         type=int,   default=5)
    p.add_argument('--warmup_epochs',    type=int,   default=5,
                   help='Number of initial epochs where early stopping and checkpointing are disabled.')
    p.add_argument('--batch_size',       type=int,   default=32)
    # Architecture hyperparameters (exposed for tuning; defaults match model.py constants)
    p.add_argument('--hidden_dim',       type=int,   default=128)
    p.add_argument('--num_layers',       type=int,   default=1)
    p.add_argument('--cnn_channels',     type=int,   default=CNN_CHANNELS,
                   help='Output channels per CNN branch (total into LSTM = cnn_channels * n_branches)')
    p.add_argument('--kernel_sizes',     type=str,
                   default=','.join(str(k) for k in DEFAULT_KERNEL_SIZES),
                   help='Comma-separated kernel sizes for multi-scale CNN branches (e.g. "3,7,24")')
    p.add_argument('--dropout',          type=float, default=0.2,
                   help='Dropout probability. Lowered from 0.4 to 0.2 because high '
                        'dropout regularises toward the running mean — exactly the '
                        'failure mode on peak-bearing targets (heating, electric).')
    p.add_argument('--lambda_smooth',    type=float, default=0.0,
                   help='Weight of temporal smoothness regularisation on forecast outputs. '
                        'Lower values let the model track sharp ups/downs; higher values '
                        'force smoother predictions. Default 0.0 (disabled) so peaks are '
                        'not penalised; values >0 trade peak fidelity for smoother curves.')
    p.add_argument('--target_clip_quantile', type=float, default=1.0,
                   help='Quantile at which to clip target columns before fitting the '
                        'shared MinMaxScaler. Default 1.0 disables clipping so the '
                        'model can learn the full peak range. Lower values (e.g. 0.999) '
                        'trim extreme outliers but also cap peak-prediction headroom: '
                        'the model will literally never learn to predict above the clip '
                        'threshold. Useful only when one or two extreme spikes dominate '
                        'the MinMax fit and starve the typical operating range of '
                        'gradient resolution.')
    p.add_argument('--over_weight_dhw', type=float, default=2.0,
                   help='Per-target asymmetric-MSE weight for DHW: how much '
                        'harder over-prediction is penalised than under-prediction '
                        'on near-zero actual hours. 1.0 = symmetric MSE; values >1 '
                        'discourage the "predict a small positive everywhere" '
                        'behaviour on zero-heavy targets but also bias the model '
                        'toward under-prediction on active hours. Set to 1.0 to disable.')
    p.add_argument('--over_weight_heating', type=float, default=1.0,
                   help='Per-target asymmetric-MSE OVER-prediction weight for heating. '
                        'Default 1.0 (symmetric). Raise to penalise the model for '
                        'predicting active heating during off-season.')
    p.add_argument('--over_weight_cooling', type=float, default=1.0,
                   help='Per-target asymmetric-MSE OVER-prediction weight for cooling. '
                        'Default 1.0 (symmetric). Raise to penalise over-prediction; '
                        'rarely needed because cooling under-prediction is the typical '
                        'failure mode (see --under_weight_cooling).')
    p.add_argument('--over_weight_electric', type=float, default=1.0,
                   help='Per-target asymmetric-MSE OVER-prediction weight for electric. '
                        'Default 1.0 (symmetric).')
    p.add_argument('--over_weight_other', type=float, default=1.0,
                   help='Legacy fallback weight applied to any target without its own '
                        '`--over_weight_<target>` flag. Default 1.0 = ordinary MSE.')
    p.add_argument('--under_weight_heating', type=float, default=2.0,
                   help='Per-target weight on UNDER-prediction during peak hours '
                        '(actual > --peak_threshold) for heating. Heating is the '
                        'canonical peak-collapse failure mode (model tracks the '
                        'running mean and flattens the 2-3 kW spikes); ~2.0 pulls '
                        'the forecast back up toward the peak. Set to 1.0 to disable.')
    p.add_argument('--under_weight_cooling', type=float, default=1.0,
                   help='Per-target UNDER-prediction weight for cooling. Default 1.0 '
                        '(off). Raise to ~2.0 on sites where cooling peaks in the test '
                        'period exceed training-range peaks (distribution-shift sites '
                        'such as KSRP Campus, where the LSTM scaler caps output at the '
                        'training max and symmetric MSE biases toward the running mean). '
                        'Mirrors the --under_weight_heating logic.')
    p.add_argument('--under_weight_dhw', type=float, default=1.0,
                   help='Per-target UNDER-prediction weight for DHW. Default 1.0 (off). '
                        'DHW is usually best served by --over_weight_dhw on zero-heavy '
                        'signals; raise this only on continuous DHW signals where peak '
                        'under-prediction is the dominant symptom.')
    p.add_argument('--under_weight_electric', type=float, default=1.0,
                   help='Per-target UNDER-prediction weight for electric. Default 1.0 (off). '
                        'Raise when symmetric MSE flattens electric peaks.')
    p.add_argument('--under_weight_other', type=float, default=1.0,
                   help='Legacy fallback weight applied to any target without its own '
                        '`--under_weight_<target>` flag. Default 1.0 (off).')
    p.add_argument('--peak_threshold', type=float, default=0.3,
                   help='Scaled-space threshold above which a target value is '
                        'considered a "peak" by the under_weight gate. Default 0.3 '
                        'targets roughly the top third of the scaled band; lower '
                        'values widen the gate (more hours treated as peaks).')
    p.add_argument('--peak_weight_alpha', type=float, default=4.0,
                   help='Continuous peak-weighting coefficient: every per-cell error '
                        'is multiplied by (1 + alpha * actual_scaled). Always-on, '
                        'grows linearly with magnitude — the model sees ramps rather '
                        'than only isolated peak tips. alpha=0 disables; 3-5 typical. '
                        'Combines multiplicatively with over/under-weight gates.')
    p.add_argument('--loss_type', type=str, default='huber',
                   choices=['mse', 'huber'],
                   help='Per-cell base loss before peak-weighting. Huber (default) has '
                        'linear tails beyond --huber_delta, damping gradient noise from '
                        'rare extreme errors while still tracking peaks. MSE is the '
                        'legacy choice and is more sensitive to outliers.')
    p.add_argument('--huber_delta', type=float, default=0.1,
                   help='Huber transition point in scaled space (~10%% of full range). '
                        'Errors below this use squared loss; above, linear.')
    p.add_argument('--pool_mode', type=str, default=DEFAULT_POOL_MODE,
                   choices=[POOL_MODE_ATTENTION, POOL_MODE_ATTENTION_MAX_LAST],
                   help='Decoder bottleneck for the LSTM output. "attention" is the '
                        'legacy single softmax-pooled context. "attention_max_last" '
                        '(default) concatenates attention + max-pool + last-step so '
                        'sharp features survive into the FC head.')
    p.add_argument('--target_log_transform', dest='target_log_transform',
                   action='store_true', default=True,
                   help='Apply log1p to target columns before fitting the shared '
                        'MinMaxScaler (inverse: expm1 after inverse_transform). '
                        'Compresses the heavy upper tail so typical-range values get '
                        'more gradient resolution and peaks are emitted more readily. '
                        'On by default; pass --no_target_log_transform for linear.')
    p.add_argument('--no_target_log_transform', dest='target_log_transform',
                   action='store_false',
                   help='Disable log1p target transform (linear MinMax only).')
    p.add_argument('--min_target_fraction', type=float, default=0.02,
                   help='Scaled value threshold below which a horizon step is considered zero.')
    p.add_argument('--min_nonzero_fraction', type=float, default=0.5,
                   help='Default minimum fraction of horizon steps that must be non-zero '
                        '(> min_target_fraction) to keep a sequence. Sequences with '
                        'predominantly zero demand are excluded - they carry no useful '
                        'gradient signal and bias the model toward predicting near-zero '
                        'everywhere. Used for heating/cooling and as the fallback for any '
                        'target without a target-specific override below.')
    p.add_argument('--min_nonzero_fraction_dhw',      type=float, default=0.1,
                   help='Per-target override for DHW. DHW is highly intermittent '
                        '(rare ~2 kW bursts on a ~90%-zero baseline), so requiring '
                        '50%-non-zero windows would discard nearly all training '
                        'sequences. 0.1 keeps the rare-active windows that carry '
                        'the only useful gradient signal for DHW magnitude.')
    p.add_argument('--min_nonzero_fraction_electric', type=float, default=0.0,
                   help='Per-target override for electric demand. Electric is '
                        'usually non-zero everywhere, so by default no filtering '
                        'is applied (0.0 disables the filter).')
    p.add_argument('--model_prefix',     type=str,   default='lstm_cnn_consumption')
    _runs_dir = os.path.join(_DIR, 'runs')
    _default_run = os.path.join(_runs_dir, datetime.now().strftime('%Y%m%d_%H%M%S'))
    p.add_argument('--output_dir',       type=str,   default=_default_run,
                   help='Directory where trained models and artefacts are saved (default: runs/<timestamp>)')
    p.add_argument('--seed', type=int, default=None,
                   help='Random seed. Omit to use $FORECAST_SEED when set '
                        '(train_pipeline exports it), else 42.')
    return p.parse_args()


def _apply_dataset_training_overrides(args) -> None:
    """Pull per-target LSTM-CNN training overrides from the dataset's
    ``TRAINING_HYPERPARAMS`` dict (if any) and write them onto ``args``.

    Schema (in ``forecasting/data/<NAME>/constants.py``):

        TRAINING_HYPERPARAMS = {
            'lstm_cnn': {
                'thermal_cooling_kw': {'under_weight': 2.0},
                'thermal_heating_kw': {'under_weight': 2.0},
                'dhw_kw':             {'over_weight': 2.0, 'min_nonzero_fraction': 0.1},
                'electric_demand_kw': {'over_weight': 1.5},
            },
        }

    Per-target knobs supported: ``over_weight``, ``under_weight``, and
    ``min_nonzero_fraction`` (the latter only on DHW and electric — heating
    / cooling share the global ``--min_nonzero_fraction`` because they're
    season-filtered already). Values are forwarded into the per-target
    argparse slot (``args.over_weight_<tag>`` etc.) and read by the per-target
    dispatch in ``main()``. Pre-2026.05 the loader only forwarded the DHW
    over-weight + the DHW/electric nonzero filters; the broader coverage
    here lets dataset descriptors enable cooling-peak asymmetric loss
    (KSRP-style distribution shift) without a code change.

    Anything the caller already passed on the command line stays — we only
    overwrite defaults sourced from this script's argparse. The "default"
    check looks at whether the user actually passed the flag by comparing
    to the parser default; since this runs after parse_args(), we instead
    snapshot the CLI-supplied values by reparsing without the dataset CSV
    if needed. For simplicity we apply dataset values unconditionally —
    the orchestrator path always lets the user pass overrides upstream.
    """
    try:
        from forecasting.consumption.adaptive import load_dataset_training_hyperparams
    except ImportError:
        return
    hp = load_dataset_training_hyperparams(getattr(args, 'csv_path', None))
    lstm_hp = (hp.get('lstm_cnn') or {}) if isinstance(hp, dict) else {}
    if not lstm_hp:
        return

    # Per-(target, knob) → argparse attribute name. Only the (target, knob)
    # pairs listed here can be overridden via dataset constants; anything
    # else in the dict is silently ignored (cheap typo guard).
    OVER_WEIGHT_ARGS = {
        'dhw_kw':             'over_weight_dhw',
        'thermal_heating_kw': 'over_weight_heating',
        'thermal_cooling_kw': 'over_weight_cooling',
        'electric_demand_kw': 'over_weight_electric',
    }
    UNDER_WEIGHT_ARGS = {
        'thermal_heating_kw': 'under_weight_heating',
        'thermal_cooling_kw': 'under_weight_cooling',
        'dhw_kw':             'under_weight_dhw',
        'electric_demand_kw': 'under_weight_electric',
    }
    NZF_ARGS = {
        'dhw_kw':             'min_nonzero_fraction_dhw',
        'electric_demand_kw': 'min_nonzero_fraction_electric',
    }

    applied: list[str] = []
    for target, cfg in lstm_hp.items():
        if not isinstance(cfg, dict):
            continue
        if 'over_weight' in cfg and target in OVER_WEIGHT_ARGS:
            attr = OVER_WEIGHT_ARGS[target]
            setattr(args, attr, float(cfg['over_weight']))
            applied.append(f'{attr}={cfg["over_weight"]}')
        if 'under_weight' in cfg and target in UNDER_WEIGHT_ARGS:
            attr = UNDER_WEIGHT_ARGS[target]
            setattr(args, attr, float(cfg['under_weight']))
            applied.append(f'{attr}={cfg["under_weight"]}')
        if 'min_nonzero_fraction' in cfg and target in NZF_ARGS:
            attr = NZF_ARGS[target]
            setattr(args, attr, float(cfg['min_nonzero_fraction']))
            applied.append(f'{attr}={cfg["min_nonzero_fraction"]}')
    if applied:
        print(f'[train] dataset LSTM overrides: {", ".join(applied)}')


def main():
    args = parse_args()
    args.seed = seed_everything(args.seed)
    _apply_dataset_training_overrides(args)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Peek at the CSV header once, early, so every TARGET_COLUMNS reference
    # below can be filtered to what this dataset actually provides.  Datasets
    # like HEEW that have no DHW channel must not have dhw_kw forced into
    # usecols or dropna(subset=...).
    _csv_header = pd.read_csv(args.csv_path, nrows=0).columns.tolist()
    available_targets = [t for t in TARGET_COLUMNS if t in _csv_header]
    if set(available_targets) != set(TARGET_COLUMNS):
        _missing_tgts = [t for t in TARGET_COLUMNS if t not in available_targets]
        print(f'[train] dataset is missing target column(s) {_missing_tgts}; those heads will be skipped.')

    # Resolve target heads to train. --targets selects a subset of available_targets;
    # empty means "train all available". Off-list values are rejected fail-fast.
    if args.targets.strip():
        targets_to_train = [t.strip() for t in args.targets.split(',') if t.strip()]
        unknown = [t for t in targets_to_train if t not in TARGET_COLUMNS]
        if unknown:
            raise SystemExit(
                f'--targets contains unknown column(s): {unknown}. '
                f'Must be a subset of {TARGET_COLUMNS}.'
            )
        print(f'[train] training {len(targets_to_train)} target head(s): {targets_to_train}')
    else:
        targets_to_train = list(available_targets)

    # Resolve feature list. --features overrides canonical LSTM_COLUMNS. The
    # available target columns and is_heating_season MUST be present in the
    # resolved stack — they drive per-target output slicing and the seasonal
    # masks — so we auto-prepend any that the caller omitted instead of
    # failing.  This lets the UI feature picker exclude targets (as it should)
    # without tripping the validator.
    if args.features.strip():
        user_features = [c.strip() for c in args.features.split(',') if c.strip()]
        required      = list(available_targets) + ['is_heating_season']
        feature_cols  = list(user_features)
        injected      = []
        for col in required:
            if col not in feature_cols:
                feature_cols.insert(0, col)
                injected.append(col)
        if injected:
            print(f'[train] auto-prepended required column(s) to --features: {injected}')
        print(f'[train] using {len(feature_cols)} feature columns '
              f'(canonical LSTM_COLUMNS overridden).')
    else:
        feature_cols = [c for c in LSTM_COLUMNS if c in _csv_header]

    # Auto-append optional activity features (e.g. dhw_volume_m3h) when the
    # CSV exposes them.
    _appended = [c for c in OPTIONAL_LSTM_COLUMNS
                 if c in _csv_header and c not in feature_cols]
    if _appended:
        feature_cols.extend(_appended)
        print(f'[train] auto-appended optional activity features: {_appended}')

    data = pd.read_csv(args.csv_path, parse_dates=['date'], index_col='date',
                       usecols=['date'] + feature_cols)
    data = data[feature_cols].dropna(subset=available_targets).ffill().fillna(0)

    # Local index of is_heating_season inside the resolved feature stack —
    # the module-level _IS_HEATING_IDX is computed from LSTM_COLUMNS and is
    # only valid when --features is unset.
    hs_idx = feature_cols.index('is_heating_season')

    # Cap rare target peaks before fitting the scaler. Heating/DHW (and to a
    # lesser extent cooling/electric) have a small number of extreme spikes
    # that push MinMaxScaler.data_max_ well above the typical operating range,
    # which compresses normal values into the bottom of the [0,1] scaled band
    # and starves the LSTM-CNN of usable gradient resolution. Only the target
    # columns are clipped — weather/temporal features are left untouched.
    q = float(args.target_clip_quantile)
    if 0.0 < q < 1.0:
        clip_summary: list[str] = []
        for t in TARGET_COLUMNS:
            if t not in data.columns:
                continue
            thr  = float(data[t].quantile(q))
            n_hi = int((data[t] > thr).sum())
            if n_hi > 0:
                data[t] = data[t].clip(upper=thr)
                clip_summary.append(f'{_TAG[t]}<=({q:.3f}q)={thr:.3f}kW ({n_hi} rows)')
        if clip_summary:
            print(f'[train] target peak clipping: {", ".join(clip_summary)}')

    # Optional log1p transform on target columns ONLY — applied before MinMax
    # so the heavy-tailed distribution is compressed and typical values get
    # more gradient resolution. The inverse path (expm1 after inverse_transform)
    # lives in evaluate.py and adaptive/utils.py and is keyed off the
    # `log1p_targets` list persisted in arch.json below.
    log1p_targets: list[str] = []
    if getattr(args, 'target_log_transform', False):
        for t in TARGET_COLUMNS:
            if t in data.columns:
                data[t] = np.log1p(np.clip(data[t].values, a_min=0.0, a_max=None))
                log1p_targets.append(t)
        if log1p_targets:
            print(f'[train] applied log1p to targets: {log1p_targets}')

    # Fit one shared scaler on the full feature set — used by all 4 models for X
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, f'scaler_{args.model_prefix}_initial.sav'), 'wb') as f:
        pickle.dump(scaler, f)

    # Per-dataset seasonal gate. When the dataset sets APPLY_SEASONAL_GATE=False
    # (e.g. HEEW: desert, cooling-dominated, both thermal channels run all year),
    # heating/cooling train as all-year models instead of regime-restricted ones,
    # keeping training consistent with the inference gate being skipped.
    try:
        from forecasting.consumption.adaptive import load_dataset_seasonal_gate
        _seasonal_gate = load_dataset_seasonal_gate(getattr(args, 'csv_path', None))
    except ImportError:
        _seasonal_gate = True
    season_filter = dict(_SEASON_FILTER)
    if not _seasonal_gate:
        season_filter['thermal_heating_kw'] = None
        season_filter['thermal_cooling_kw'] = None
        print('[train] APPLY_SEASONAL_GATE=False: training heating/cooling all-year '
              '(no heating-season split).')

    # Guard: is_heating_season must span both 0 and 1 so the 0.5 threshold works.
    # Only relevant when at least one target is still season-filtered.
    if any(s is not None for s in season_filter.values()):
        _hs_col = data_scaled[:, hs_idx]
        assert _hs_col.min() < 0.5 < _hs_col.max(), (
            "Training data must include both heating and non-heating rows so that "
            "MinMaxScaler maps is_heating_season to [0, 1] and the 0.5 filter threshold works."
        )

    # Season masks on the raw (pre-scaled) DataFrame rows.
    # is_heating_season is 0/1 so its scaled value is also 0.0/1.0.
    _hs_raw = data['is_heating_season'].values.astype(bool)

    # Build independent sequence sets for each regime up front.
    # Heating/cooling: sequences come exclusively from contiguous in-season blocks
    # so no context window can straddle a season boundary.
    # DHW/electric: full dataset, no season filtering needed.
    print('Building sequence sets...')
    X_heating, y_heating = _season_sequences(
        data_scaled, _hs_raw, args.sequence_length, args.forecast_horizon)
    X_cooling, y_cooling = _season_sequences(
        data_scaled, ~_hs_raw, args.sequence_length, args.forecast_horizon)
    X_all,     y_all     = create_sequences(
        data_scaled, args.sequence_length, args.forecast_horizon)
    print(f'  heating:  {len(X_heating)} sequences')
    print(f'  cooling:  {len(X_cooling)} sequences')
    print(f'  all-year: {len(X_all)} sequences')

    for target in targets_to_train:
        tag    = _TAG[target]
        t_idx  = feature_cols.index(target)
        season = season_filter[target]

        if season is True:
            X_s = X_heating
            y_s = y_heating[:, :, t_idx : t_idx + 1]   # (n, horizon, 1)
        elif season is False:
            X_s = X_cooling
            y_s = y_cooling[:, :, t_idx : t_idx + 1]
        else:
            X_s = X_all
            y_s = y_all[:, :, t_idx : t_idx + 1]

        # Drop training sequences where most of the forecast horizon is
        # near-zero — they bias the model toward predicting near-zero
        # everywhere. Threshold is per-target: heating/cooling default to
        # the seasonal threshold (0.5), DHW relaxes to 0.1 because the
        # rare-active windows are the only meaningful gradient signal,
        # and electric defaults to 0.0 (no filter) since it's rarely zero.
        # Pass 0.0 for any target to disable the filter for that target.
        _per_target_min_nonzero = {
            'thermal_heating_kw': args.min_nonzero_fraction,
            'thermal_cooling_kw': args.min_nonzero_fraction,
            'dhw_kw':             args.min_nonzero_fraction_dhw,
            'electric_demand_kw': args.min_nonzero_fraction_electric,
        }
        min_nz = float(_per_target_min_nonzero.get(target, args.min_nonzero_fraction))
        if min_nz > 0.0 and len(y_s) > 0:
            nonzero_steps = (y_s[:, :, 0] > args.min_target_fraction)
            nonzero_frac  = nonzero_steps.mean(axis=1)
            keep = nonzero_frac >= min_nz
            X_s = X_s[keep]
            y_s = y_s[keep]
            print(f'  [{tag}] {int(keep.sum())}/{len(keep)} sequences kept '
                  f'(>={min_nz:.0%} non-zero horizon steps)')

        n         = len(X_s)
        train_end = int(n * 0.70)
        val_end   = int(n * 0.85)
        X_train, y_train = X_s[:train_end],        y_s[:train_end]
        X_val,   y_val   = X_s[train_end:val_end], y_s[train_end:val_end]

        def to_tensor(arr):
            return torch.tensor(arr, dtype=torch.float32).to(device)

        train_loader = DataLoader(TensorDataset(to_tensor(X_train), to_tensor(y_train)),
                                  batch_size=args.batch_size, shuffle=True)
        val_loader   = DataLoader(TensorDataset(to_tensor(X_val),   to_tensor(y_val)),
                                  batch_size=args.batch_size, shuffle=False)

        kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]
        model = LSTM_CNN_Model(
            input_size=X_train.shape[2],
            output_size=args.forecast_horizon,
            num_features=1,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            cnn_channels=args.cnn_channels,
            kernel_sizes=kernel_sizes,
            dropout=args.dropout,
            pool_mode=args.pool_mode,
        ).to(device)

        # Per-target asymmetric-MSE weights. Every target has its own
        # `--over_weight_<target>` / `--under_weight_<target>` flag (also
        # overridable via the dataset's `TRAINING_HYPERPARAMS['lstm_cnn']`
        # block — see _apply_dataset_training_overrides). `_other` is a legacy
        # fallback for anything not enumerated here.
        over_weight = float({
            'dhw_kw':             args.over_weight_dhw,
            'thermal_heating_kw': args.over_weight_heating,
            'thermal_cooling_kw': args.over_weight_cooling,
            'electric_demand_kw': args.over_weight_electric,
        }.get(target, args.over_weight_other))
        under_weight = float({
            'thermal_heating_kw': args.under_weight_heating,
            'thermal_cooling_kw': args.under_weight_cooling,
            'dhw_kw':             args.under_weight_dhw,
            'electric_demand_kw': args.under_weight_electric,
        }.get(target, args.under_weight_other))
        bits = []
        if over_weight  != 1.0: bits.append(f'over={over_weight}')
        if under_weight != 1.0: bits.append(f'under={under_weight}')
        if args.peak_weight_alpha > 0.0: bits.append(f'alpha={args.peak_weight_alpha}')
        if args.loss_type != 'mse': bits.append(f'loss={args.loss_type}')
        if bits:
            print(f'  [{tag}] loss config: {", ".join(bits)}')

        def criterion(pred, actual, _ow=over_weight, _uw=under_weight):
            return _asymmetric_mse(
                pred, actual,
                over_weight=_ow,
                eps=args.min_target_fraction,
                under_weight=_uw,
                peak_threshold=args.peak_threshold,
                peak_weight_alpha=args.peak_weight_alpha,
                loss_type=args.loss_type,
                huber_delta=args.huber_delta,
            )

        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
        )

        print(f'\n=== {tag} | {n} sequences  train={train_end}  val={val_end - train_end} '
              f'| warmup={args.warmup_epochs} epochs ===')
        best_val_loss  = float('inf')
        epochs_no_impr = 0
        use_smooth = args.lambda_smooth > 0.0
        for epoch in range(args.epochs):
            model.train()
            train_loss = 0.0
            for xb, yb in train_loader:
                optimizer.zero_grad()
                pred = model(xb)                                  # (batch, horizon, 1)
                loss = criterion(pred, yb)
                if use_smooth:
                    smooth = torch.mean((pred[:, 1:] - pred[:, :-1]) ** 2)
                    loss = loss + args.lambda_smooth * smooth
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    pred = model(xb)
                    vl   = criterion(pred, yb)
                    if use_smooth:
                        smooth = torch.mean((pred[:, 1:] - pred[:, :-1]) ** 2)
                        vl     = vl + args.lambda_smooth * smooth
                    val_loss += vl.item()
            val_loss /= len(val_loader)

            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']

            in_warmup = epoch < args.warmup_epochs
            print(f'Epoch {epoch+1}/{args.epochs}  train={train_loss:.4f}  val={val_loss:.4f}  '
                  f'lr={current_lr:.2e}{"  [warmup]" if in_warmup else ""}')

            if in_warmup:
                continue

            if val_loss < best_val_loss:
                best_val_loss  = val_loss
                epochs_no_impr = 0
                torch.save(model.state_dict(),
                           os.path.join(args.output_dir, f'{args.model_prefix}_{tag}_initial.pth'))
            else:
                epochs_no_impr += 1
                if epochs_no_impr >= args.patience:
                    print(f'Early stopping at epoch {epoch+1} (no improvement for {args.patience} epochs)')
                    break

        print(f'Saved {args.model_prefix}_{tag}_initial.pth  (best val={best_val_loss:.4f})')

    # Persist architecture config so _load_models() can reconstruct the exact architecture.
    # sequence_length and forecast_horizon are recorded so downstream eval / adaptive
    # runs can rebuild the model with matching window dimensions instead of falling back
    # to module-level constants.
    arch_config = {
        'hidden_dim':       args.hidden_dim,
        'num_layers':       args.num_layers,
        'cnn_channels':     args.cnn_channels,
        'kernel_sizes':     kernel_sizes,
        'dropout':          args.dropout,
        'sequence_length':  args.sequence_length,
        'forecast_horizon': args.forecast_horizon,
        # Feature inventory persisted so evaluate / adaptive paths can rebuild
        # the same input stack without depending on LSTM_COLUMNS being stable.
        'input_size':       int(data_scaled.shape[1]),
        'features':         list(feature_cols),
        # Which target heads were actually trained in this run (may be a
        # subset of TARGET_COLUMNS when the caller passed --targets).
        'targets':          list(targets_to_train),
        # Pool mode for the decoder bottleneck. Persisted so eval/forecaster
        # can rebuild the model with the right FC input dim. Falls back to
        # 'attention' (legacy) when absent.
        'pool_mode':        args.pool_mode,
        # Which target columns were log1p-transformed before scaling. Inverse
        # path applies expm1 after scaler.inverse_transform on these columns.
        'log1p_targets':    log1p_targets,
    }
    arch_path = os.path.join(args.output_dir, f'{args.model_prefix}_arch.json')
    with open(arch_path, 'w') as f:
        json.dump(arch_config, f, indent=2)
    print(f'Saved architecture config -> {arch_path}')

    print(f'\nAll {len(targets_to_train)} models saved to {args.output_dir}')


if __name__ == '__main__':
    main()
