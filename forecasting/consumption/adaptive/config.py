"""Configuration for adaptive consumption forecasting."""

import importlib
import json
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from forecasting.consumption.constants import SEQUENCE_LENGTH
from forecasting.constants import (
    FORECAST_HORIZON,
    DEFAULT_LSTM_THERMAL_WEIGHT, DEFAULT_GBDT_THERMAL_WEIGHT,
    DEFAULT_LSTM_ELECTRIC_WEIGHT, DEFAULT_GBDT_ELECTRIC_WEIGHT,
    DEFAULT_MAE_THRESHOLD, DEFAULT_RMSE_THRESHOLD, DEFAULT_MASE_THRESHOLD,
    DEFAULT_EVALUATION_FREQUENCY, DEFAULT_MAX_RETRAINS, RETRAIN_WINDOW,
    TARGET_THERMAL_HEATING, TARGET_THERMAL_COOLING, TARGET_DHW, TARGET_ELECTRIC,
)


# ---------------------------------------------------------------------------
# Evaluation protocol
# ---------------------------------------------------------------------------
# 'legacy' (default) keeps every per-dataset override declared in
# forecasting/data/<NAME>/constants.py. 'clean' ignores the overrides that were
# set by hand after looking at forecast results (retrain thresholds, GBDT floor,
# naive-anchor weights, per-target MASE gates, per-target loss weights, retrain
# boosting settings) and replaces them by rules fitted on the validation slice
# of the training partition. Structural descriptors (loss objective by target
# type, activity filters, inactivity / seasonal gates) are kept in both modes.
PROTOCOL_ENV = 'FORECAST_PROTOCOL'
_CLEAN_DROP_KEYS = ('_retrain', 'over_weight', 'under_weight')


def protocol_is_clean() -> bool:
    return os.environ.get(PROTOCOL_ENV, '').strip().lower() == 'clean'


def clean_training_hyperparams(hp: dict) -> dict:
    """Copy of TRAINING_HYPERPARAMS without the hand-tuned per-target keys."""
    out: dict = {}
    for family, per_target in (hp or {}).items():
        if not isinstance(per_target, dict):
            out[family] = per_target
            continue
        out[family] = {}
        for target, cfg in per_target.items():
            if isinstance(cfg, dict):
                cfg = {k: v for k, v in cfg.items() if k not in _CLEAN_DROP_KEYS}
            out[family][target] = cfg
    return out


def create_default_args():
    class DefaultArgs:
        def __init__(self):
            self.lstm_model_name   = 'lstm_cnn_consumption'
            self.gbdt_model_prefix = 'xgb_consumption'
            self.tft_model_prefix  = 'tft_consumption'

            # Root directory for a specific training run, expected layout:
            #   <model_base_dir>/lstm_cnn/*.pth, *.sav
            #   <model_base_dir>/gbdt/*.json, *.sav
            #   <model_base_dir>/transformer/*.pth, *.sav (optional)
            #   <model_base_dir>/weights/weights_per_horizon.json (optional)
            # When None, the forecaster falls back to the legacy source-tree
            # locations (adaptive/../lstm_cnn, adaptive/../gbdt).
            self.model_base_dir = None

            # Per-target ensemble weights, used when per-horizon weights are
            # unavailable. Tuples can be 2-element (lstm, gbdt) for the legacy
            # LSTM+GBDT blend, or 3-element (lstm, gbdt, tft) when a TFT model
            # is loaded. ``combine_predictions`` picks the right arity based on
            # whether ``tft_preds`` is passed.
            self.weights = {
                TARGET_THERMAL_HEATING: (DEFAULT_LSTM_THERMAL_WEIGHT,  DEFAULT_GBDT_THERMAL_WEIGHT,  0.0),
                TARGET_THERMAL_COOLING: (DEFAULT_LSTM_THERMAL_WEIGHT,  DEFAULT_GBDT_THERMAL_WEIGHT,  0.0),
                TARGET_DHW:             (DEFAULT_LSTM_THERMAL_WEIGHT,  DEFAULT_GBDT_THERMAL_WEIGHT,  0.0),
                TARGET_ELECTRIC:        (DEFAULT_LSTM_ELECTRIC_WEIGHT, DEFAULT_GBDT_ELECTRIC_WEIGHT, 0.0),
            }

            self.use_horizon_weights = True   # load weights_per_horizon.json when present
            self.sequence_length  = SEQUENCE_LENGTH
            self.forecast_horizon = FORECAST_HORIZON
            self.start_index      = SEQUENCE_LENGTH
            self.non_adaptive     = False
            self.plot_interval    = 25
            self.save_plots       = False
            self.plots_dir        = 'plots'
            self.mae_threshold    = DEFAULT_MAE_THRESHOLD
            self.rmse_threshold   = DEFAULT_RMSE_THRESHOLD
            self.mase_threshold   = DEFAULT_MASE_THRESHOLD
            self.min_retrain_samples   = 500
            self.retrain_window        = RETRAIN_WINDOW
            self.evaluation_frequency  = DEFAULT_EVALUATION_FREQUENCY
            self.max_retrains          = DEFAULT_MAX_RETRAINS

            # Peak-aware loss knobs, consumed by ``_finetune_lstm`` and
            # ``_finetune_tft``. The triple (peak_weight_alpha=0.5,
            # peak_threshold=0.3, loss_type='huber', huber_delta=0.15)
            # turns plain MSE into a continuous peak-emphasis Huber: each
            # sample is weighted by 1 + alpha * y/y_max for peaks above
            # 30 % of target max, and the regression loss itself uses
            # Huber with delta=0.15 to be more robust to the calibration
            # outliers that show up on heating/cooling shoulder weeks.
            # Defaults were chosen so the loss matches the initial-
            # training behaviour for KSRP (training_hyperparams overrides
            # still take precedence per target).
            self.peak_weight_alpha    = 0.5
            self.peak_threshold       = 0.3
            self.loss_type            = 'huber'
            self.huber_delta          = 0.15
            self.under_weight_heating = 1.0
            self.under_weight_other   = 1.0
            self.min_target_fraction  = 0.02

            # Ensemble sanity gate — disables a base learner's contribution
            # for a given (target, window) when its rolling-mean MAE over the
            # last ``base_health_window`` windows exceeds
            # ``base_health_ratio`` × the median of the other learners' MAEs.
            # Ratio 6.0 only trips on catastrophic miscalibration (e.g. the
            # TFT issue from runs/20260520-081105-* where TFT MAE was ~6×
            # GBDT on electric); a tighter 4× threshold was firing on KSRP
            # LSTM-electric where the model was off but not broken, masking
            # its (real) under-prediction signal as a fully-dropped stream.
            self.base_health_window = 50
            self.base_health_ratio  = 6.0

            # ── Per-target post-combine weekly-naive anchor ───────────────
            # Blends each target's ensemble output with the actual value 168 h
            # earlier (same hour-of-day, one week prior). A defensive band-aid
            # for targets where the trained model lost to a trivial seasonal
            # baseline because the test-era diurnal/weekly shape no longer
            # matches the training-era shape (KSRP DHW: ~12 kW model MAE vs
            # ~5 kW for weekly-naive after the campus-wide schedule change
            # around 2016). Default empty dict → no blending; dataset
            # constants opt in via ``ADAPTIVE_NAIVE_ANCHOR_WEIGHTS``.
            self.naive_anchor_weights: dict[str, float] = {}

            # ── Per-target MASE retrain trigger ───────────────────────────
            # Augments the overall MAE/RMSE/MASE gates with per-target MASE
            # thresholds. Needed when one target's scale is so much smaller
            # than the others (KSRP: DHW MAE ~12 kW vs heating ~150 kW) that
            # its drift never moves the overall average enough to trip the
            # global gate. Dataset constants opt in via
            # ``ADAPTIVE_PER_TARGET_MASE_THRESHOLD``; default empty dict
            # leaves legacy overall-only behaviour.
            self.per_target_mase_threshold: dict[str, float] = {}
    return DefaultArgs()


def validate_args(args):
    for target, ws in args.weights.items():
        if sum(ws) == 0:
            raise ValueError(f'Weights for {target} must not all be zero.')
    if args.mae_threshold <= 0 or args.rmse_threshold <= 0:
        raise ValueError('MAE/RMSE thresholds must be positive.')
    if args.mase_threshold <= 0:
        raise ValueError('MASE threshold must be positive.')


def apply_model_meta(args) -> tuple[int, int]:
    """Override args.sequence_length / args.forecast_horizon from the trained
    model's persisted metadata (train_meta.json, with _arch.json as fallback).

    Returns the effective (sequence_length, forecast_horizon). When no
    model_base_dir is set or no metadata is found, leaves args unchanged.
    """
    base = getattr(args, 'model_base_dir', None)
    if not base:
        return args.sequence_length, args.forecast_horizon

    meta_path = os.path.join(base, 'train_meta.json')
    seq_len, horizon = None, None
    if os.path.exists(meta_path):
        try:
            with open(meta_path) as fh:
                payload = json.load(fh)
            seq_len = payload.get('sequence_length')
            horizon = payload.get('forecast_horizon')
        except Exception:
            pass

    if seq_len is None or horizon is None:
        # Fallback: per-model arch.json written by lstm_cnn/train.py
        arch_path = os.path.join(base, 'lstm_cnn', f'{args.lstm_model_name}_arch.json')
        if os.path.exists(arch_path):
            try:
                with open(arch_path) as fh:
                    arch = json.load(fh)
                seq_len = seq_len if seq_len is not None else arch.get('sequence_length')
                horizon = horizon if horizon is not None else arch.get('forecast_horizon')
            except Exception:
                pass

    if seq_len is not None:
        args.sequence_length = int(seq_len)
        args.start_index     = int(seq_len)
    if horizon is not None:
        args.forecast_horizon = int(horizon)
    return args.sequence_length, args.forecast_horizon


# ---------------------------------------------------------------------------
# Per-dataset adaptive threshold loading
# ---------------------------------------------------------------------------

def _infer_dataset_name_from_csv(csv_path: str | None) -> str | None:
    """Walk up from a CSV path to find ``forecasting/data/<NAME>/``.

    Returns the dataset directory name (e.g. 'HouseZero', 'VelykaDymerka') or
    None when the path doesn't sit under the standard layout. Used to pick up
    that dataset's ``ADAPTIVE_*_THRESHOLD*`` constants automatically.
    """
    if not csv_path:
        return None
    p = os.path.abspath(csv_path)
    for _ in range(10):                      # bounded ascent
        parent = os.path.dirname(p)
        if parent == p:
            return None
        if (os.path.basename(parent) == 'data'
                and os.path.basename(os.path.dirname(parent)) == 'forecasting'):
            return os.path.basename(p)
        p = parent
    return None


def apply_dataset_thresholds(args, csv_path: str | None) -> str | None:
    """Override args' MAE / RMSE / MASE thresholds with dataset-specific values
    when the input CSV lives under ``forecasting/data/<NAME>/`` and that
    dataset's ``constants.py`` declares ``ADAPTIVE_MAE_THRESHOLD_KW`` etc.

    Caller responsibility: invoke after ``create_default_args()`` and before
    applying any explicit CLI overrides, so the precedence chain is

        global defaults  ←  dataset constants  ←  CLI overrides

    Returns the dataset name when overrides were applied (for logging), or
    None when nothing matched.
    """
    name = _infer_dataset_name_from_csv(csv_path)
    if not name:
        return None
    try:
        mod = importlib.import_module(f'forecasting.data.{name}.constants')
    except ModuleNotFoundError:
        return None

    applied: list[str] = []
    mae   = getattr(mod, 'ADAPTIVE_MAE_THRESHOLD_KW',  None)
    rmse  = getattr(mod, 'ADAPTIVE_RMSE_THRESHOLD_KW', None)
    mase  = getattr(mod, 'ADAPTIVE_MASE_THRESHOLD',    None)
    retrn = getattr(mod, 'ADAPTIVE_RETRAIN_WINDOW',    None)
    gfh   = getattr(mod, 'ADAPTIVE_GBDT_FLOOR_HORIZONS', None)
    gfw   = getattr(mod, 'ADAPTIVE_GBDT_FLOOR_WEIGHTS',  None)
    gates = getattr(mod, 'INACTIVITY_GATES',           None)
    seasg = getattr(mod, 'APPLY_SEASONAL_GATE',        None)
    hp    = getattr(mod, 'TRAINING_HYPERPARAMS',       None)
    anch  = getattr(mod, 'ADAPTIVE_NAIVE_ANCHOR_WEIGHTS',     None)
    ptm   = getattr(mod, 'ADAPTIVE_PER_TARGET_MASE_THRESHOLD', None)
    if protocol_is_clean():
        # Hand-set per-dataset values are not used; run.py / tune_weights.py
        # supply rule-based replacements (see PROTOCOL_ENV above).
        gfh = gfw = anch = ptm = retrn = None
        if isinstance(hp, dict):
            hp = clean_training_hyperparams(hp)
        applied.append('protocol=clean')
    if mae is not None:
        args.mae_threshold = float(mae)
        applied.append(f'mae={mae}')
    if rmse is not None:
        args.rmse_threshold = float(rmse)
        applied.append(f'rmse={rmse}')
    if mase is not None:
        args.mase_threshold = float(mase)
        applied.append(f'mase={mase}')
    if retrn is not None:
        args.retrain_window = int(retrn)
        applied.append(f'retrain_window={retrn}')
    if gfh is not None and gfw is not None and len(gfh) == len(gfw):
        # Stash as plain tuples so the forecaster can forward them to
        # tune_per_horizon_weights without depending on the dataset module.
        args.gbdt_floor_horizons = tuple(int(x) for x in gfh)
        args.gbdt_floor_weights  = tuple(float(x) for x in gfw)
        applied.append(f'gbdt_floor={list(zip(args.gbdt_floor_horizons, args.gbdt_floor_weights))}')
    if gates:
        # Stash on args so apply_physical_constraints can read it without an
        # extra argument through the forecaster's prediction path.
        args.inactivity_gates = dict(gates)
        applied.append(f'inactivity_gates={list(gates.keys())}')
    if seasg is not None:
        # Per-dataset toggle for the legacy calendar seasonal gate in
        # apply_physical_constraints. Defaults to True (gate applies) when the
        # dataset omits the constant; cooling-dominated/year-round sites set it
        # False so real out-of-"season" demand is not zeroed.
        args.apply_seasonal_gate = bool(seasg)
        applied.append(f'apply_seasonal_gate={bool(seasg)}')
    if isinstance(hp, dict) and hp:
        # Stash so the forecaster can apply the same per-target objectives /
        # asymmetric-loss weights at adaptive-retrain time as the initial
        # training script used. Schema documented on
        # ``load_dataset_training_hyperparams`` below.
        args.training_hyperparams = hp
        applied.append('training_hyperparams=set')
    if isinstance(anch, dict) and anch:
        # Per-target weekly-naive (lag-168) anchor weights, consumed by
        # apply_naive_anchor() in the forecaster's predict path. Schema:
        # {target: alpha in [0, 1]}; alpha=0 means "no anchor" (default).
        args.naive_anchor_weights = {str(k): float(v) for k, v in anch.items()}
        applied.append(f'naive_anchor={args.naive_anchor_weights}')
    if isinstance(ptm, dict) and ptm:
        # Per-target MASE retrain trigger. Fires retrain when target's MASE
        # over the last window exceeds threshold, even if overall metrics
        # are within bounds. Schema: {target: mase_threshold}.
        args.per_target_mase_threshold = {str(k): float(v) for k, v in ptm.items()}
        applied.append(f'per_target_mase={args.per_target_mase_threshold}')
    if applied:
        print(f'[adaptive] using {name} thresholds: {", ".join(applied)}')
        return name
    return None


def load_dataset_training_hyperparams(csv_path: str | None) -> dict:
    """Load the dataset's optional ``TRAINING_HYPERPARAMS`` dict.

    Mirrors ``apply_dataset_thresholds`` but for *training* knobs that the
    train scripts read at startup. The expected schema:

        TRAINING_HYPERPARAMS = {
            'lstm_cnn': {
                'dhw_kw': {'over_weight': 3.0, 'min_nonzero_fraction': 0.1},
                ...
            },
            'gbdt': {
                'dhw_kw': {'objective': 'reg:tweedie',
                           'tweedie_variance_power': 1.5},
                ...
            },
        }

    Returns the dict as-is (or an empty dict if the dataset has none / the
    CSV path doesn't sit under ``forecasting/data/<NAME>/``). The caller
    decides which sub-keys it cares about.
    """
    name = _infer_dataset_name_from_csv(csv_path)
    if not name:
        return {}
    try:
        mod = importlib.import_module(f'forecasting.data.{name}.constants')
    except ModuleNotFoundError:
        return {}
    hp = getattr(mod, 'TRAINING_HYPERPARAMS', None)
    if not isinstance(hp, dict):
        return {}
    if protocol_is_clean():
        return clean_training_hyperparams(hp)
    return hp


def load_dataset_seasonal_gate(csv_path: str | None) -> bool:
    """Return the dataset's ``APPLY_SEASONAL_GATE`` flag (default ``True``).

    Mirrors ``load_dataset_training_hyperparams`` but for the legacy calendar
    seasonal split. When ``True`` (or absent), heating/cooling are trained as
    separate regime-restricted models (heating-season-only / cooling-season-
    only) and the inference gate zeros out-of-season demand. When a dataset
    sets it ``False`` (e.g. HEEW — a desert, cooling-dominated campus where both
    thermal channels run year-round), heating/cooling train all-year and the
    inference gate is skipped, so the two stay consistent.

    Defaults to ``True`` when the CSV path doesn't sit under
    ``forecasting/data/<NAME>/`` or the dataset omits the constant, preserving
    the historical HouseZero behaviour for every other dataset.
    """
    name = _infer_dataset_name_from_csv(csv_path)
    if not name:
        return True
    try:
        mod = importlib.import_module(f'forecasting.data.{name}.constants')
    except ModuleNotFoundError:
        return True
    return bool(getattr(mod, 'APPLY_SEASONAL_GATE', True))
