"""
AdaptiveForecaster for consumption forecasting.

Targets: thermal_heating_kw, thermal_cooling_kw, dhw_kw, electric_demand_kw

Three base learners per target:
  * LSTM-CNN  — required
  * GBDT      — required
  * TFT       — optional; loaded when transformer/ checkpoints are present

The three model families are blended per-horizon by weights produced by
tune_weights.py. Heating/cooling models are seasonally filtered during
fine-tuning to match the data distribution they were trained on.
"""

import json
import os
import pickle
import sys
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from forecasting.constants import (
    FORECAST_HORIZON,
    TARGET_THERMAL_HEATING, TARGET_THERMAL_COOLING, TARGET_DHW, TARGET_ELECTRIC,
)
from forecasting.consumption.constants import (
    LSTM_COLUMNS, TARGET_COLUMNS, SEQUENCE_LENGTH, LAG_STEPS,
    OPTIONAL_GBDT_FEATURE_COLUMNS, ACTIVITY_LAG_STEPS,
)
from forecasting.consumption.lstm_cnn.model import LSTM_CNN_Model, POOL_MODE_ATTENTION
from forecasting.consumption.lstm_cnn.train import (
    _TAG, _IS_HEATING_IDX, _SEASON_FILTER, _asymmetric_mse,
)
from forecasting.consumption.transformer.model import TFTModel
from forecasting.consumption.adaptive.utils import (
    prepare_lstm_input, forecast_lstm_cnn_single, forecast_transformer_single,
    forecast_gbdt_iterative, combine_predictions, combine_with_horizon_weights,
    apply_naive_anchor, apply_physical_constraints,
    calculate_metrics_adaptive, check_threshold_exceeded,
    create_sequences_for_finetuning,
    tune_per_horizon_weights,
)

warnings.filterwarnings('ignore')

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))


def _weekly_naive(data: pd.DataFrame, target: str, start_idx: int, horizon: int) -> np.ndarray:
    """Weekly-seasonal-naive forecast issued at ``start_idx`` (NaN if no history)."""
    out = np.full(horizon, np.nan, dtype=float)
    if start_idx < 168 or target not in data.columns:
        return out
    col = data[target].to_numpy(dtype=float, copy=False)
    for h in range(horizon):
        out[h] = col[start_idx - 168 + (h % 168)]
    return out


class AdaptiveForecaster:
    def __init__(self, args, device: str = 'cpu'):
        self.args    = args
        self.device  = device
        self.retrain_count       = 0
        self.performance_history = []
        self.non_adaptive_mode   = getattr(args, 'non_adaptive', False)
        self._load_models()
        if self.non_adaptive_mode:
            print('[AdaptiveForecaster] non-adaptive mode (frozen initial models)')

    # ------------------------------------------------------------------
    def _load_models(self):
        sfx      = '_initial'
        pfx      = self.args.gbdt_model_prefix
        lnm      = self.args.lstm_model_name
        base     = getattr(self.args, 'model_base_dir', None) or os.path.join(_MODULE_DIR, '..')
        lstm_dir = os.path.join(base, 'lstm_cnn')

        # Shared scaler (fitted on all LSTM_COLUMNS)
        with open(os.path.join(lstm_dir, f'scaler_{lnm}{sfx}.sav'), 'rb') as f:
            self.lstm_scaler = pickle.load(f)

        # One LSTM-CNN model per target (num_features=1 each).
        # Architecture is inferred directly from the checkpoint state dict so that
        # the instantiated model always matches the saved weights, regardless of
        # what args or the _arch.json say.
        # Read dropout + input_size + pool_mode + log1p_targets from arch.json
        # (not recoverable from weights alone); fall back to args/default.
        # input_size matters when training used a custom --features list
        # (n_feat != len(LSTM_COLUMNS)).
        arch_path = os.path.join(lstm_dir, f'{lnm}_arch.json')
        if os.path.exists(arch_path):
            with open(arch_path) as _f:
                _saved = json.load(_f)
            _dropout    = _saved.get('dropout', getattr(self.args, 'dropout', 0.2))
            n_feat      = int(_saved.get('input_size', len(LSTM_COLUMNS)))
            _pool_mode  = _saved.get('pool_mode')   # may be None for legacy
            self.lstm_log1p_targets = set(_saved.get('log1p_targets') or [])
        else:
            _dropout    = getattr(self.args, 'dropout', 0.2)
            n_feat      = len(LSTM_COLUMNS)
            _pool_mode  = None
            self.lstm_log1p_targets = set()

        # Resolve which targets this run actually trained. Datasets without a
        # given channel (e.g. HEEW has no DHW) never produce that head's
        # checkpoint, so we load only the targets whose .pth exists and drive
        # every per-target loop below off self.targets instead of the global
        # TARGET_COLUMNS. Canonical TARGET_COLUMNS order is preserved.
        self.lstm_models = {}
        self.targets     = []
        for t in TARGET_COLUMNS:
            tag      = _TAG[t]
            pth_path = os.path.join(lstm_dir, f'{lnm}_{tag}{sfx}.pth')
            if not os.path.exists(pth_path):
                continue
            self.targets.append(t)
            sd       = torch.load(pth_path, map_location=self.device, weights_only=False)
            # Infer structural hyperparams from weight tensor shapes
            hidden_dim = sd['lstm.weight_hh_l0'].shape[1]   # hh: (4*H, H)
            num_layers = sum(1 for k in sd if k.startswith('lstm.weight_ih_l'))
            if any(k.startswith('conv_branches.') for k in sd):
                # Multi-scale: one branch per entry in conv_branches.
                # conv weight shape is (out_channels, in_channels, kernel_size);
                # in_channels is the LSTM-CNN input feature count.
                n_branches   = len([k for k in sd if k.startswith('conv_branches.')
                                    and k.endswith('.weight')])
                cnn_channels = sd['conv_branches.0.weight'].shape[0]   # per-branch
                kernel_sizes = [sd[f'conv_branches.{i}.weight'].shape[2]
                                for i in range(n_branches)]
                ckpt_n_feat  = sd['conv_branches.0.weight'].shape[1]
            else:
                # Legacy single-scale checkpoint
                cnn_channels = sd['conv1.weight'].shape[0]
                kernel_sizes = [sd['conv1.weight'].shape[2]]
                ckpt_n_feat  = sd['conv1.weight'].shape[1]
            # Trust the checkpoint over arch.json: if --features changed between
            # training and tune/forecast, the saved weights are the ground truth.
            if ckpt_n_feat != n_feat:
                n_feat = ckpt_n_feat
            # Resolve pool_mode: arch.json wins; fall back to inferring from
            # the FC input dim (hidden_dim => attention-only;
            # 3*hidden_dim => attention+max+last concat).
            pool_mode = _pool_mode
            if not pool_mode:
                fc_in = sd['fc.weight'].shape[1] if 'fc.weight' in sd else hidden_dim
                pool_mode = ('attention_max_last' if fc_in == 3 * hidden_dim
                             else POOL_MODE_ATTENTION)
            model = LSTM_CNN_Model(
                n_feat, int(self.args.forecast_horizon), num_features=1,
                hidden_dim=hidden_dim, num_layers=num_layers,
                cnn_channels=cnn_channels, kernel_sizes=kernel_sizes,
                dropout=_dropout, pool_mode=pool_mode,
            )
            model.load_state_dict(sd)
            self.lstm_models[t] = model.to(self.device)
            self.lstm_models[t].eval()

        if not self.targets:
            raise FileNotFoundError(
                f'No LSTM-CNN checkpoints found in {lstm_dir} for any of '
                f'{TARGET_COLUMNS} (expected {lnm}_<tag>{sfx}.pth).'
            )
        if set(self.targets) != set(TARGET_COLUMNS):
            _missing = [t for t in TARGET_COLUMNS if t not in self.targets]
            print(f'[AdaptiveForecaster] {len(self.targets)} target head(s) present '
                  f'{[_TAG[t] for t in self.targets]}; missing {[_TAG[t] for t in _missing]} '
                  f'(channel absent from this dataset).')

        # Per-horizon ensemble weights — produced by tune_weights.py.
        # On-disk schemas supported:
        #   legacy flat:      {target: [w_lstm_h1, ...]}
        #   2-way wrapped:    {"weights": {target: [w_lstm_h1, ...]}}
        #   3-way wrapped:    {"weights": {target: {"lstm": [...], "gbdt": [...], "tft": [...]}}}
        # Falls back to fixed default weights when the file is absent.
        self.horizon_weights = {}
        if getattr(self.args, 'use_horizon_weights', True):
            weights_path = (
                os.path.join(base, getattr(self.args, 'weights_dir_name', 'weights'),
                             'weights_per_horizon.json')
                if getattr(self.args, 'model_base_dir', None)
                else os.path.join(_MODULE_DIR, 'weights_per_horizon.json')
            )
            if os.path.exists(weights_path):
                with open(weights_path) as _f:
                    raw = json.load(_f)
                if isinstance(raw, dict) and isinstance(raw.get('weights'), dict):
                    w_raw = raw['weights']
                else:
                    w_raw = raw       # legacy flat layout
                for t in w_raw:
                    if t not in TARGET_COLUMNS:
                        continue
                    entry = w_raw[t]
                    if isinstance(entry, dict):
                        self.horizon_weights[t] = {k: np.asarray(v, dtype=float)
                                                   for k, v in entry.items()}
                    else:
                        self.horizon_weights[t] = np.asarray(entry, dtype=float)
                _kinds = {t: ('3-way' if isinstance(v, dict) else '2-way')
                          for t, v in self.horizon_weights.items()}
                print(f'[AdaptiveForecaster] Loaded per-horizon weights: '
                      + ', '.join(f'{_TAG[t]}({k})' for t, k in _kinds.items()))
            else:
                print('[AdaptiveForecaster] No weights_per_horizon.json — using fixed ensemble weights')
        else:
            print('[AdaptiveForecaster] Per-horizon weights disabled — using fixed ensemble weights')

        # Sidecar written by tune_weights.py: validation-selected GBDT floor
        # per target and validation error levels (used by the clean protocol).
        self.weights_meta = {}
        if getattr(self.args, 'model_base_dir', None):
            _wm = os.path.join(base, getattr(self.args, 'weights_dir_name', 'weights'),
                               'weights_meta.json')
            if os.path.exists(_wm):
                with open(_wm) as _f:
                    self.weights_meta = json.load(_f)

        # XGBoost — one model + two scalers per target
        gbdt_dir = os.path.join(base, 'gbdt')
        self.gbdt_models       = {}
        self.gbdt_scalers_feat = {}
        self.gbdt_scalers_tgt  = {}
        for t in self.targets:
            tag = _TAG[t]
            m = xgb.Booster()
            m.load_model(os.path.join(gbdt_dir, f'{pfx}_{tag}{sfx}.json'))
            self.gbdt_models[t] = m
            with open(os.path.join(gbdt_dir, f'scaler_features_{tag}_{pfx}{sfx}.sav'), 'rb') as f:
                self.gbdt_scalers_feat[t] = pickle.load(f)
            with open(os.path.join(gbdt_dir, f'scaler_{tag}_{pfx}{sfx}.sav'), 'rb') as f:
                self.gbdt_scalers_tgt[t] = pickle.load(f)

        # TFT — optional third base learner. Present only when the run dir
        # contains a transformer/ subdirectory with checkpoints.
        self.tft_models = {}
        self.tft_scaler = None
        tft_dir  = os.path.join(base, 'transformer')
        tft_pref = getattr(self.args, 'tft_model_prefix', 'tft_consumption')
        tft_scaler_path = os.path.join(tft_dir, f'scaler_{tft_pref}{sfx}.sav')
        if ('tft' in self._members() and os.path.isdir(tft_dir)
                and os.path.exists(tft_scaler_path)):
            with open(tft_scaler_path, 'rb') as f:
                self.tft_scaler = pickle.load(f)
            for t in self.targets:
                tag      = _TAG[t]
                pth_path = os.path.join(tft_dir, f'{tft_pref}_{tag}{sfx}.pth')
                if not os.path.exists(pth_path):
                    continue
                sd = torch.load(pth_path, map_location=self.device, weights_only=False)
                hidden_dim  = sd['lstm_encoder.weight_hh_l0'].shape[1]
                lstm_layers = sum(1 for k in sd if k.startswith('lstm_encoder.weight_ih_l'))
                # MultiheadAttention head count isn't recoverable from weights;
                # rely on the persisted arch.json, fall back to a divisor of hidden_dim.
                tft_arch_path = os.path.join(tft_dir, f'{tft_pref}_arch.json')
                n_heads = 4
                tft_dropout = 0.1
                ckpt_n_feat = None
                if os.path.exists(tft_arch_path):
                    with open(tft_arch_path) as _af:
                        _ta = json.load(_af)
                    n_heads      = int(_ta.get('n_heads', n_heads))
                    tft_dropout  = float(_ta.get('dropout', tft_dropout))
                    ckpt_n_feat  = int(_ta.get('input_size', 0)) or None
                # Number of features the VSN was built for = number of feature_grns entries
                vsn_keys = [k for k in sd if k.startswith('vsn.feature_grns.')]
                if vsn_keys:
                    idxs = {int(k.split('.')[2]) for k in vsn_keys}
                    ckpt_n_feat = (max(idxs) + 1) if idxs else ckpt_n_feat
                if ckpt_n_feat is None:
                    ckpt_n_feat = len(LSTM_COLUMNS)
                model = TFTModel(
                    input_size=ckpt_n_feat,
                    output_size=int(self.args.forecast_horizon),
                    num_features=1,
                    hidden_dim=hidden_dim,
                    n_heads=n_heads,
                    lstm_layers=lstm_layers,
                    dropout=tft_dropout,
                )
                model.load_state_dict(sd)
                model.eval()
                self.tft_models[t] = model.to(self.device)
            if self.tft_models:
                print(f'[AdaptiveForecaster] Loaded TFT models for: '
                      f'{[_TAG[t] for t in self.tft_models]}')

    # ------------------------------------------------------------------
    def _members(self) -> set:
        """Base learners taking part in the blend (default: all three).

        ``args.members`` is a comma list such as ``'lstm,gbdt'`` (two-head run
        on a three-head checkpoint) or ``'gbdt'`` (single-model baseline).
        """
        m = getattr(self.args, 'members', None)
        if not m:
            return {'lstm', 'gbdt', 'tft'}
        if isinstance(m, str):
            m = [x.strip().lower() for x in m.split(',') if x.strip()]
        return set(m) or {'lstm', 'gbdt', 'tft'}

    def _update_causal_anchor(self, actuals: dict) -> None:
        """Choose each target's weekly-naive anchor weight from recent windows.

        Used when ``args.anchor_mode == 'causal'``. After every scored window
        the pre-anchor blend, the weekly-naive forecast and the actuals are
        buffered (last ``anchor_window`` windows). The weight for the NEXT
        window is the grid value in {0, 0.1, ..., 0.9} that minimises MAE of
        ``(1 - a) * blend + a * naive`` over the buffer; it stays 0 until
        ``anchor_min_windows`` windows are available. Only past data is used.
        """
        if getattr(self.args, 'anchor_mode', 'fixed') != 'causal':
            return
        st = getattr(self, 'last_stages', None)
        if not st or 'naive' not in st:
            return
        if not hasattr(self, '_anchor_buf'):
            self._anchor_buf, self._causal_alpha = {}, {}
        keep  = int(getattr(self.args, 'anchor_window', 28))
        min_n = int(getattr(self.args, 'anchor_min_windows', 14))
        grid  = np.round(np.arange(0.0, 0.91, 0.1), 2)
        for t, act in actuals.items():
            b, nv = st['blend'].get(t), st['naive'].get(t)
            if b is None or nv is None or np.any(np.isnan(nv)):
                continue
            buf = self._anchor_buf.setdefault(t, [])
            buf.append((np.asarray(b, dtype=float), np.asarray(nv, dtype=float),
                        np.asarray(act, dtype=float)))
            if len(buf) > keep:
                del buf[:len(buf) - keep]
            if len(buf) < min_n:
                self._causal_alpha[t] = 0.0
                continue
            B = np.stack([x[0] for x in buf])
            N = np.stack([x[1] for x in buf])
            A = np.stack([x[2] for x in buf])
            maes = [float(np.mean(np.abs(A - ((1.0 - a) * B + a * N)))) for a in grid]
            self._causal_alpha[t] = float(grid[int(np.argmin(maes))])

    def _combine(self, lstm_preds: dict, gbdt_preds: dict,
                 tft_preds: dict | None = None) -> dict:
        """Use per-horizon weights when available, else fixed weights.

        Per-target sanity gate: if a base learner's RECENT (last
        ``base_health_window`` windows) MAE on this target is more than
        ``base_health_ratio`` × the median of the other base learners' MAEs,
        DROP that learner from the combine for this target — its weight is
        zeroed AND the surviving streams' weights are renormalised so they
        sum to 1 (or, in the per-horizon case, sum to 1 per step). Without
        the renormalisation step, the gated stream's weight would simply
        leak into a sub-unity convex combination, pulling the combined
        output below every surviving base learner. Default thresholds are
        conservative — they only fire when a learner is catastrophically
        off, so a healthy ensemble sees no behaviour change.
        """
        # Single-model baselines: the "blend" is that model's own forecast.
        _mem = self._members()
        if _mem == {'gbdt'}:
            return {t: np.array(gbdt_preds[t], dtype=float, copy=True) for t in lstm_preds}
        if _mem == {'lstm'}:
            return {t: np.array(lstm_preds[t], dtype=float, copy=True) for t in lstm_preds}
        if _mem == {'tft'} and tft_preds:
            return {t: np.array(tft_preds[t], dtype=float, copy=True) for t in lstm_preds}
        gated_streams = self._gating_mask(lstm_preds, gbdt_preds, tft_preds)
        if self.horizon_weights:
            weights = (self._adjust_horizon_weights(gated_streams)
                       if gated_streams else self.horizon_weights)
            out = combine_with_horizon_weights(
                lstm_preds, gbdt_preds, weights, tft_preds=tft_preds,
            )
        else:
            fixed = (self._adjust_fixed_weights(gated_streams)
                     if gated_streams else self.args.weights)
            out = combine_predictions(
                lstm_preds, gbdt_preds, fixed, tft_preds=tft_preds,
            )
        return self._apply_rolling_weights(out, lstm_preds, gbdt_preds, tft_preds)

    # ------------------------------------------------------------------
    # Causal rolling blend weights (``args.weights_mode == 'rolling'``)
    # ------------------------------------------------------------------
    @staticmethod
    def _simplex_grid(k: int, step: float = 0.05) -> np.ndarray:
        """All weight vectors of length k on the simplex with the given step."""
        import itertools
        m = int(round(1.0 / step))
        return np.array([c for c in itertools.product(range(m + 1), repeat=k)
                         if sum(c) == m], dtype=float) / m

    def _apply_rolling_weights(self, out: dict, lstm_preds: dict, gbdt_preds: dict,
                               tft_preds: dict | None) -> dict:
        """Replace the static blend by the rolling-weight blend where weights exist.

        Targets without rolling weights yet (fewer than ``roll_min_windows``
        scored windows) keep the validation-tuned blend in ``out``.
        """
        if getattr(self.args, 'weights_mode', 'static') != 'rolling':
            return out
        rw = getattr(self, '_roll_w', None) or {}
        streams = {'lstm': lstm_preds, 'gbdt': gbdt_preds, 'tft': tft_preds or {}}
        for t in list(out):
            w = rw.get(t)
            if not w or any(t not in streams[n] for n in w):
                continue
            out[t] = sum(np.asarray(w[n], dtype=float) * np.asarray(streams[n][t], dtype=float)
                         for n in w)
        return out

    def _update_rolling_weights(self, lstm_preds: dict, gbdt_preds: dict,
                                tft_preds: dict | None, actuals: dict) -> None:
        """Choose each target's blend weights for the NEXT window.

        After every scored window the member forecasts and the actuals are
        buffered (last ``roll_window`` windows). For each of ``roll_blocks``
        equal blocks of the horizon (0 = one weight vector per horizon step)
        the weights are the simplex grid point (step 0.05) with the lowest
        absolute error over the buffer. No weights are set until
        ``roll_min_windows`` windows are available. Only past data is used.
        Called by the evaluation runner with the member forecasts as issued
        (before the runtime gates), for every window.
        """
        if getattr(self.args, 'weights_mode', 'static') != 'rolling':
            return
        mem = self._members()
        if not ({'lstm', 'gbdt'} <= mem):
            return
        names = ['lstm', 'gbdt'] + (['tft'] if (tft_preds and 'tft' in mem) else [])
        keep  = int(getattr(self.args, 'roll_window', 42))
        min_n = min(int(getattr(self.args, 'roll_min_windows', 14)), keep)
        nblk  = int(getattr(self.args, 'roll_blocks', 0) or 0)
        if not hasattr(self, '_roll_buf'):
            self._roll_buf, self._roll_w, self._roll_grid = {}, {}, {}
        if len(names) not in self._roll_grid:
            self._roll_grid[len(names)] = self._simplex_grid(len(names))
        grid = self._roll_grid[len(names)]
        streams = {'lstm': lstm_preds, 'gbdt': gbdt_preds, 'tft': tft_preds or {}}
        for t, act in actuals.items():
            if any(t not in streams[n] for n in names):
                continue
            M = np.stack([np.asarray(streams[n][t], dtype=float) for n in names])
            A = np.asarray(act, dtype=float)
            if M.shape[1] != A.shape[0] or np.any(np.isnan(A)) or np.any(np.isnan(M)):
                continue
            buf = self._roll_buf.setdefault(t, [])
            if buf and buf[-1][0].shape != M.shape:
                del buf[:]                       # member set changed: start again
            buf.append((M, A))
            if len(buf) > keep:
                del buf[:len(buf) - keep]
            if len(buf) < min_n:
                self._roll_w[t] = None
                continue
            MM = np.stack([b[0] for b in buf])                      # (n, k, H)
            AA = np.stack([b[1] for b in buf])                      # (n, H)
            err = np.abs(np.einsum('ck,nkh->cnh', grid, MM) - AA[None]).sum(axis=1)   # (C, H)
            H = AA.shape[1]
            step = max(H // (nblk if nblk > 0 else H), 1)
            w = np.zeros((len(names), H))
            for lo in range(0, H, step):
                c = int(np.argmin(err[:, lo:lo + step].sum(axis=1)))
                w[:, lo:lo + step] = grid[c][:, None]
            self._roll_w[t] = {n: w[i] for i, n in enumerate(names)}

    # ------------------------------------------------------------------
    # Forecast-weather stress test (``args.weather_noise > 0``)
    # ------------------------------------------------------------------
    _CALENDAR_PREFIXES = ('hour', 'month', 'dow', 'doy', 'day_', 'week', 'is_', 'season', 'year',
                          'terrestrial_radiation', 'solar_obstruction')

    def _perturb_future_weather(self, data: pd.DataFrame, start: int, horizon: int):
        """Add lead-time-dependent noise to the weather columns of the forecast
        window, in place, and return what is needed to undo it.

        The loads use measured weather as if it were a perfect forecast. This
        stress test replaces the weather of rows ``start .. start+horizon`` by
        ``x + level * sd(x) * sqrt(h / 24) * z_h`` where ``z`` is an AR(1)
        series (rho = 0.9, unit variance), independent per variable, and the
        result is clipped to the observed range of the variable. Calendar and
        astronomical columns, the targets and their lags are left alone, and
        so is everything before ``start`` (the observed past).
        """
        level = float(getattr(self.args, 'weather_noise', 0.0) or 0.0)
        end = min(start + horizon, len(data))
        if level <= 0 or end <= start:
            return None
        key = tuple(data.columns)
        cache = getattr(self, '_wn_cache', None)
        if cache is None or cache[0] != key:
            skip = set(self.targets)
            pos, sd, lo, hi = [], [], [], []
            for j, c in enumerate(data.columns):
                cs = str(c)
                if (c in skip or '_lag_' in cs or cs.startswith('Unnamed')
                        or cs.startswith(self._CALENDAR_PREFIXES)
                        or not pd.api.types.is_numeric_dtype(data[c])
                        or pd.api.types.is_bool_dtype(data[c])):
                    continue
                v = data[c].to_numpy(dtype=float)
                s = float(np.nanstd(v))
                if not np.isfinite(s) or s <= 0:
                    continue
                if not pd.api.types.is_float_dtype(data[c]):
                    data[c] = data[c].astype(float)
                pos.append(j); sd.append(s)
                lo.append(float(np.nanmin(v))); hi.append(float(np.nanmax(v)))
            cache = (key, np.array(pos, dtype=int), np.array(sd), np.array(lo), np.array(hi))
            self._wn_cache = cache
            print(f'[weather-noise] level={level}: {len(pos)} weather columns perturbed in the '
                  f'forecast window')
        _, pos, sd, lo, hi = cache
        if len(pos) == 0:
            return None
        n = end - start
        try:
            stamp = int(pd.Timestamp(data.index[start]).value // 10 ** 9)
        except Exception:
            stamp = int(start)
        rng = np.random.default_rng((stamp + 7919 * int(getattr(self.args, 'seed', 0) or 0)) % (2 ** 32))
        rho = 0.9
        z = np.empty((n, len(pos)))
        z[0] = rng.standard_normal(len(pos))
        eps = rng.standard_normal((n, len(pos))) * np.sqrt(1.0 - rho ** 2)
        for h in range(1, n):
            z[h] = rho * z[h - 1] + eps[h]
        lead = np.sqrt(np.arange(1, n + 1) / 24.0)[:, None]
        orig = data.iloc[start:end, pos].to_numpy(dtype=float, copy=True)
        noisy = np.clip(orig + level * sd[None, :] * lead * z, lo[None, :], hi[None, :])
        noisy = np.where(np.isnan(orig), orig, noisy)
        # Zero-floored variables (radiation at night, no rain): an exact zero stays zero.
        noisy = np.where((orig == 0.0) & (lo[None, :] == 0.0), orig, noisy)
        data.iloc[start:end, pos] = noisy
        return (start, end, pos, orig)

    @staticmethod
    def _restore_future_weather(data: pd.DataFrame, token) -> None:
        if token is not None:
            start, end, pos, orig = token
            data.iloc[start:end, pos] = orig

    def _gating_mask(self, lstm_preds: dict, gbdt_preds: dict,
                     tft_preds: dict | None) -> dict:
        """Return ``{target: set[stream_name]}`` listing which streams should
        be dropped from the combine for each target.

        ``self._recent_per_model_mae`` is a sliding ``{model_name: {target:
        [mae_w1, mae_w2, ...]}}`` map populated by ``observe_window`` after
        each rolling forecast. The gate compares the running mean over the
        last ``base_health_window`` entries; before enough history is
        collected (cold start) the mask is empty and the combine uses the
        original weights.
        """
        # Args-driven knobs with safe defaults so legacy runs see no change.
        win   = int(getattr(self.args, 'base_health_window', 50) or 0)
        ratio = float(getattr(self.args, 'base_health_ratio', 4.0) or 0.0)
        if win <= 0 or ratio <= 1.0:
            return {}
        history = getattr(self, '_recent_per_model_mae', None)
        if not history:
            return {}

        streams: dict[str, dict] = {'lstm': lstm_preds, 'gbdt': gbdt_preds}
        if tft_preds:
            streams['tft'] = tft_preds

        # Per-target recent-MAE table (only non-empty streams contribute).
        per_model_recent: dict[str, dict[str, float]] = {}
        for name in streams:
            target_to_recent = {}
            for t, hist in (history.get(name) or {}).items():
                if not hist:
                    continue
                tail = hist[-win:]
                if tail:
                    target_to_recent[t] = float(np.nanmean(tail))
            if target_to_recent:
                per_model_recent[name] = target_to_recent
        if len(per_model_recent) < 2:
            return {}

        mask: dict[str, set] = {}
        for name in streams:
            for t in list(streams[name]):
                others = [per_model_recent[other].get(t)
                          for other in per_model_recent if other != name
                          and per_model_recent[other].get(t) is not None]
                if not others:
                    continue
                ref = float(np.median(others))
                if ref <= 0:
                    continue
                own = per_model_recent.get(name, {}).get(t)
                if own is None or own <= ratio * ref:
                    continue
                mask.setdefault(t, set()).add(name)
                print(f'  [combine] gated {name.upper()} on {_TAG.get(t, t)}: '
                      f'recent MAE {own:.2f} > {ratio:.1f}x median {ref:.2f}')
        return mask

    @staticmethod
    def _stream_order(n: int) -> list:
        """Stream-name positions for fixed-weight tuples: 2-tuples → lstm/gbdt;
        3-tuples → lstm/gbdt/tft."""
        return ['lstm', 'gbdt', 'tft'][:n]

    def _adjust_horizon_weights(self, gated_streams: dict) -> dict:
        """Build ``self.horizon_weights`` with gated streams zeroed and the
        surviving streams' weights renormalised to sum to 1 per horizon step.

        Edge case: if every stream is gated for a target, leave that target's
        weights untouched — produces the same (low-trust) combined that the
        pre-fix behaviour did, which is still better than emitting all-zero
        predictions on a target where nobody is performing well enough.
        """
        adj: dict = {}
        for t, w in self.horizon_weights.items():
            gs = gated_streams.get(t)
            if not gs:
                adj[t] = w
                continue
            if isinstance(w, dict):
                wl = np.asarray(w['lstm'], dtype=float).copy()
                wg = np.asarray(w['gbdt'], dtype=float).copy()
                wt = np.asarray(w.get('tft', np.zeros_like(wl)), dtype=float).copy()
                if 'lstm' in gs: wl[:] = 0.0
                if 'gbdt' in gs: wg[:] = 0.0
                if 'tft'  in gs: wt[:] = 0.0
                total = wl + wg + wt
                safe  = total > 0
                wl[safe] /= total[safe]
                wg[safe] /= total[safe]
                wt[safe] /= total[safe]
                adj[t] = {'lstm': wl, 'gbdt': wg, 'tft': wt}
            else:
                wl = np.asarray(w, dtype=float).copy()
                if 'lstm' in gs and 'gbdt' in gs:
                    adj[t] = wl                       # both gated → leave as-is
                elif 'lstm' in gs:
                    adj[t] = np.zeros_like(wl)        # GBDT takes all weight
                elif 'gbdt' in gs:
                    adj[t] = np.ones_like(wl)         # LSTM takes all weight
                else:
                    adj[t] = wl
        return adj

    def _adjust_fixed_weights(self, gated_streams: dict) -> dict:
        """Build ``self.args.weights``-shaped dict with gated streams zeroed
        and the surviving streams' weights renormalised to sum to 1."""
        adj: dict = {}
        for t, w in self.args.weights.items():
            gs = gated_streams.get(t)
            if not gs:
                adj[t] = w
                continue
            names = self._stream_order(len(w))
            vals  = [0.0 if n in gs else float(w[i]) for i, n in enumerate(names)]
            total = sum(vals)
            if total > 0:
                vals = [v / total for v in vals]
            else:
                vals = list(w)                        # all gated → fall back
            adj[t] = tuple(vals)
        return adj

    def observe_window(self, lstm_preds: dict, gbdt_preds: dict,
                       tft_preds: dict | None, actuals: dict) -> None:
        """Update the per-model sliding-window MAE table.

        Called once per forecast window by ``run_scenario`` after the
        prediction is finalised. Uses ``self._recent_per_model_mae`` as a
        ``{model_name: {target: deque-as-list}}`` map; trimmed to twice the
        gating window so memory stays bounded.
        """
        self._update_causal_anchor(actuals)
        win = int(getattr(self.args, 'base_health_window', 50) or 0)
        if win <= 0:
            return
        cap = max(win * 2, 100)
        if not hasattr(self, '_recent_per_model_mae') or self._recent_per_model_mae is None:
            self._recent_per_model_mae = {}
        streams = {'lstm': lstm_preds, 'gbdt': gbdt_preds}
        if tft_preds:
            streams['tft'] = tft_preds
        for name, preds in streams.items():
            bucket = self._recent_per_model_mae.setdefault(name, {})
            for t, pred in preds.items():
                act = actuals.get(t)
                if act is None or pred is None:
                    continue
                pred = np.asarray(pred, dtype=float)
                act  = np.asarray(act,  dtype=float)
                n = min(pred.size, act.size)
                if n == 0:
                    continue
                mae = float(np.nanmean(np.abs(pred[:n] - act[:n])))
                lst = bucket.setdefault(t, [])
                lst.append(mae)
                if len(lst) > cap:
                    del lst[:len(lst) - cap]

    def _predict_tft(self, data: pd.DataFrame, lstm_start: int) -> dict:
        """Run all loaded TFT models on the current context window.

        Returns an empty dict when no TFT checkpoints are loaded — the rest of
        the pipeline then degrades to the 2-way LSTM+GBDT blend.

        The TFT head is a raw nn.Linear with no sigmoid, so its raw outputs
        can drift outside [0, 1] in scaled space. Combined with the
        training-time ``target_clip_quantile`` lowering each target's
        ``data_max_``, that drift gets multiplied by (data_max - data_min)
        on inverse-transform and shows up as 3–4× overpredictions in raw
        kW. Two guardrails:
          * ``clamp_scaled_to_unit=True`` clips the scaled output to [0, 1]
            before the inverse — caps the inverse at exactly ``data_max_``.
          * ``max_value_kw`` further caps to a target-specific absolute
            ceiling derived from the scaler's own ``data_max_`` plus a
            small overshoot margin, so a single broken horizon step can't
            poison the whole window.
        """
        if not self.tft_models:
            return {}
        scaler = self.tft_scaler or self.lstm_scaler
        # TFT was trained on its own scaler (separate path); the log1p list
        # only applies when we're reusing the LSTM-CNN scaler as a fallback.
        log1p_targets = self.lstm_log1p_targets if scaler is self.lstm_scaler else None
        tft_input = prepare_lstm_input(
            data, lstm_start, self.args.sequence_length, scaler,
            log1p_targets=log1p_targets,
        )

        cols = (list(scaler.feature_names_in_)
                if hasattr(scaler, 'feature_names_in_')
                else list(LSTM_COLUMNS))
        # Absolute kW ceiling per target = scaler's training-time max * 1.5,
        # giving a small overshoot margin while still preventing the wild
        # blow-ups seen on heating/cooling/electric.
        max_caps = {}
        for t in self.tft_models:
            if t in cols:
                t_idx = cols.index(t)
                _mx = float(scaler.data_max_[t_idx])
                if log1p_targets and t in log1p_targets:
                    _mx = float(np.expm1(_mx))     # log-space maximum -> kW
                max_caps[t] = _mx * 1.5
            else:
                max_caps[t] = None

        return {
            t: forecast_transformer_single(
                self.tft_models[t], tft_input, scaler, t, self.device,
                log1p_targets=log1p_targets,
                clamp_scaled_to_unit=True,
                max_value_kw=max_caps[t],
            )
            for t in self.tft_models
        }

    def _lstm_max_caps(self) -> dict:
        """Per-target absolute kW ceiling for the LSTM inverse-transform.

        Mirrors the TFT guardrail in ``_predict_tft``: caps each target at the
        scaler's training-time ``data_max_`` plus a 50% overshoot margin. A
        fine-tuned LSTM head can diverge (catastrophic forgetting across
        retrains), and on a log1p target like ``electric`` the ``expm1`` inverse
        then turns a drifted scaled output into an astronomically large kW
        (1e20+ MAE). This cap is the last-resort safety net that keeps a broken
        head from poisoning the combine/metrics; the gating logic still drops
        it, but on a sane scale.
        """
        scaler = self.lstm_scaler
        cols = (list(scaler.feature_names_in_)
                if hasattr(scaler, 'feature_names_in_') else list(LSTM_COLUMNS))
        # The scaler is fitted on log1p(kW) for the targets listed in
        # ``lstm_log1p_targets``; its ``data_max_`` is then a log-space value
        # and must be converted back before it can serve as a kW ceiling.
        # (Using it directly capped e.g. a 20 MW load at 16.5 kW.)
        log1p = getattr(self, 'lstm_log1p_targets', None) or set()
        caps = {}
        for t in self.targets:
            if t not in cols:
                caps[t] = None
                continue
            mx = float(scaler.data_max_[cols.index(t)])
            if t in log1p:
                mx = float(np.expm1(mx))
            caps[t] = mx * 1.5
        return caps

    def predict(self, data: pd.DataFrame, forecast_start_idx: int) -> dict:
        args       = self.args
        lstm_start = forecast_start_idx - args.sequence_length

        lstm_input = prepare_lstm_input(
            data, lstm_start, args.sequence_length, self.lstm_scaler,
            log1p_targets=self.lstm_log1p_targets,
        )
        lstm_caps = self._lstm_max_caps()
        lstm_preds = {
            t: forecast_lstm_cnn_single(
                self.lstm_models[t], lstm_input,
                self.lstm_scaler, t, self.device,
                log1p_targets=self.lstm_log1p_targets,
                max_value_kw=lstm_caps[t],
            )
            for t in self.targets
        }
        gbdt_preds = forecast_gbdt_iterative(
            self.gbdt_models, self.gbdt_scalers_feat, self.gbdt_scalers_tgt,
            self.targets, data, forecast_start_idx, args.forecast_horizon,
        )
        tft_preds = self._predict_tft(data, lstm_start)
        combined  = self._combine(lstm_preds, gbdt_preds, tft_preds=tft_preds or None)
        combined  = apply_naive_anchor(
            combined, data, forecast_start_idx, args.forecast_horizon,
            getattr(args, 'naive_anchor_weights', None),
        )
        combined  = apply_physical_constraints(
            combined, data, forecast_start_idx, args.forecast_horizon,
            inactivity_gates=getattr(args, 'inactivity_gates', None),
            apply_seasonal_gate=getattr(args, 'apply_seasonal_gate', True),
        )
        return combined

    def predict_components(self, data: pd.DataFrame, forecast_start_idx: int) -> tuple:
        """
        Same as predict() but returns all base-model signals separately.

        Returns
        -------
        (lstm_preds, gbdt_preds, tft_preds, combined_preds)
        each a dict[target -> np.ndarray(horizon,)]. ``tft_preds`` is empty
        when no TFT models are loaded.
        """
        args       = self.args
        lstm_start = forecast_start_idx - args.sequence_length

        lstm_input = prepare_lstm_input(
            data, lstm_start, args.sequence_length, self.lstm_scaler,
            log1p_targets=self.lstm_log1p_targets,
        )
        lstm_caps = self._lstm_max_caps()
        lstm_preds = {
            t: forecast_lstm_cnn_single(
                self.lstm_models[t], lstm_input,
                self.lstm_scaler, t, self.device,
                log1p_targets=self.lstm_log1p_targets,
                max_value_kw=lstm_caps[t],
            )
            for t in self.targets
        }
        _wn_token = self._perturb_future_weather(data, forecast_start_idx, args.forecast_horizon)
        try:
            gbdt_preds = forecast_gbdt_iterative(
                self.gbdt_models, self.gbdt_scalers_feat, self.gbdt_scalers_tgt,
                self.targets, data, forecast_start_idx, args.forecast_horizon,
            )
            tft_preds = self._predict_tft(data, lstm_start)
        finally:
            self._restore_future_weather(data, _wn_token)
        combined  = self._combine(lstm_preds, gbdt_preds, tft_preds=tft_preds or None)
        # Snapshot the intermediate stages so the evaluation runner can store
        # them: the convex blend before the weekly-naive anchor, and after it
        # (before the physical / inactivity gates). Copies, so later in-place
        # edits by the caller cannot change what was recorded.
        _blend = {t: np.array(v, dtype=float, copy=True) for t, v in combined.items()}
        _naive = {t: _weekly_naive(data, t, forecast_start_idx, args.forecast_horizon)
                  for t in combined}
        _mode = getattr(args, 'anchor_mode', 'fixed')
        if _mode == 'causal':
            _aw = dict(getattr(self, '_causal_alpha', None) or {})
        elif _mode == 'off':
            _aw = None
        else:
            _aw = getattr(args, 'naive_anchor_weights', None)
        combined  = apply_naive_anchor(
            combined, data, forecast_start_idx, args.forecast_horizon, _aw,
        )
        _anchored = {t: np.array(v, dtype=float, copy=True) for t, v in combined.items()}
        combined  = apply_physical_constraints(
            combined, data, forecast_start_idx, args.forecast_horizon,
            inactivity_gates=getattr(args, 'inactivity_gates', None),
            apply_seasonal_gate=getattr(args, 'apply_seasonal_gate', True),
        )
        self.last_stages = {'blend': _blend, 'anchored': _anchored, 'naive': _naive,
                            'anchor_weights': dict(_aw or {})}
        return lstm_preds, gbdt_preds, tft_preds, combined

    # ------------------------------------------------------------------
    def _should_retrain(self, metrics: dict) -> tuple:
        if self.non_adaptive_mode:
            return False, []
        return check_threshold_exceeded(
            metrics,
            self.args.mae_threshold,
            self.args.rmse_threshold,
            mase_thr=getattr(self.args, 'mase_threshold', None),
            per_target_mase_thr=getattr(self.args, 'per_target_mase_threshold', None),
            targets=self.targets,
        )

    def _finetune_lstm(self, data_window: pd.DataFrame):
        """Fine-tune each LSTM model on recent data, respecting seasonal filters."""
        # Use the column order the scaler was fit on so a --features override at
        # training time (e.g. NEST_UMAR, which lacks HouseZero-only zone columns)
        # doesn't blow up the slice here.
        cols = (list(self.lstm_scaler.feature_names_in_)
                if hasattr(self.lstm_scaler, 'feature_names_in_') else list(LSTM_COLUMNS))
        subset = data_window[cols].dropna().copy()
        if len(subset) == 0:
            return
        # Apply log1p on target columns to match the scaler's fit space when
        # the model was trained with --target_log_transform.
        if self.lstm_log1p_targets:
            for t in self.lstm_log1p_targets:
                if t in subset.columns:
                    subset[t] = np.log1p(np.clip(subset[t].values, a_min=0.0, a_max=None))
        scaled = self.lstm_scaler.transform(subset)
        X_all, y_all = create_sequences_for_finetuning(
            scaled, self.args.sequence_length, self.args.forecast_horizon
        )
        if len(X_all) < 10:
            return

        from forecasting.consumption.lstm_cnn.train import _season_sequences

        # Build season-pure sequence sets from contiguous blocks — mirrors training.
        _hs_raw   = subset['is_heating_season'].values.astype(bool) if 'is_heating_season' in subset.columns else np.ones(len(scaled), dtype=bool)
        X_heating, y_heating = _season_sequences(scaled, _hs_raw,  self.args.sequence_length, self.args.forecast_horizon)
        X_cooling, y_cooling = _season_sequences(scaled, ~_hs_raw, self.args.sequence_length, self.args.forecast_horizon)

        # Per-target asymmetric-MSE weight, sourced from the dataset's
        # TRAINING_HYPERPARAMS (set on args by apply_dataset_thresholds).
        # Falls back to 1.0 (= ordinary MSE) when the dataset doesn't ship a
        # value for this target.
        lstm_hp = ((getattr(self.args, 'training_hyperparams', {}) or {})
                   .get('lstm_cnn') or {})

        # Mirror training-time peak-aware loss settings so fine-tuning doesn't
        # silently regress to plain MSE and undo the peak-tracking behaviour
        # the initial training learned. All knobs default to off (1.0/0.0/mse)
        # so legacy callers see no behaviour change.
        peak_alpha   = float(getattr(self.args, 'peak_weight_alpha', 0.0))
        peak_thresh  = float(getattr(self.args, 'peak_threshold',    0.3))
        loss_type    = str  (getattr(self.args, 'loss_type',         'mse'))
        huber_delta  = float(getattr(self.args, 'huber_delta',       0.1))
        uw_heating   = float(getattr(self.args, 'under_weight_heating', 1.0))
        uw_other     = float(getattr(self.args, 'under_weight_other',   1.0))
        min_target_f = float(getattr(self.args, 'min_target_fraction', 0.02))

        for t in self.targets:
            t_idx  = cols.index(t)
            season = _SEASON_FILTER[t]

            if season is True:
                X_s = X_heating
                y_s = y_heating[:, :, t_idx : t_idx + 1]
            elif season is False:
                X_s = X_cooling
                y_s = y_cooling[:, :, t_idx : t_idx + 1]
            else:
                X_s = X_all
                y_s = y_all[:, :, t_idx : t_idx + 1]   # (n, horizon, 1)
            if len(X_s) < 10:
                continue

            X_t = torch.tensor(X_s, dtype=torch.float32).to(self.device)
            y_t = torch.tensor(y_s, dtype=torch.float32).to(self.device)
            # batch_size=16 + 6 epochs. The retrain window is ~1500 h (~94
            # batches/epoch on KSRP) — 3 epochs (~280 steps) under-shoots when
            # the recent window contains a regime shift the base model wasn't
            # trained on (KSRP DHW post-2016 schedule inversion). Bumping to
            # 6 epochs (~560 steps) roughly doubles the gradient budget while
            # staying well under the count where pure-SGD-without-val-set
            # starts to overfit a 62-day window. lr=1e-4 unchanged: raising
            # the learning rate is the higher-risk lever (catastrophic
            # forgetting in the lower CNN layers); more passes at the same
            # LR is the safer way to dial up aggressiveness. Dataset can
            # override via TRAINING_HYPERPARAMS['lstm_cnn'][target]['_retrain'].
            ft_cfg     = (lstm_hp.get(t) or {}).get('_retrain', {}) or {}
            batch_size = int(ft_cfg.get('batch_size', 16))
            epochs     = int(ft_cfg.get('epochs',     6))
            lr         = float(ft_cfg.get('learning_rate', 1e-4))
            loader = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True)

            ow = float((lstm_hp.get(t) or {}).get('over_weight', 1.0))
            uw = uw_heating if t == 'thermal_heating_kw' else uw_other

            model = self.lstm_models[t]
            model.train()
            opt = torch.optim.Adam(model.parameters(), lr=lr)
            for _ in range(epochs):
                for xb, yb in loader:
                    opt.zero_grad()
                    loss = _asymmetric_mse(
                        model(xb), yb,
                        over_weight=ow,
                        eps=min_target_f,
                        under_weight=uw,
                        peak_threshold=peak_thresh,
                        peak_weight_alpha=peak_alpha,
                        loss_type=loss_type,
                        huber_delta=huber_delta,
                    )
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
            model.eval()

    def _finetune_gbdt(self, data_window: pd.DataFrame,
                        num_boost_round: int = 50):
        """Continue training each per-target GBDT booster on the recent window.

        Lag columns are constructed from the window's own actual history (no
        forecast leakage). Pairs (features_at_t, target_at_t) — the booster's
        original objective, which forecast_gbdt_iterative later rolls across
        the horizon.

        Per-target retrain knobs (``num_boost_round``, ``learning_rate``)
        can be overridden via ``TRAINING_HYPERPARAMS['gbdt'][target]['_retrain']``
        in the dataset's constants.py. Without overrides the caller's
        ``num_boost_round`` is used (default 50 trees, bumped from 20). The
        boosters' initial training round count is usually 200-500, so the
        old 20-tree default barely shifted the prediction surface on a
        large retrain window; 50 trees at lr=0.05 is roughly equivalent
        to 100 trees at lr=0.025 and gives the adapter meaningful
        capacity to track regime shifts without overwriting the prior.
        """
        df = data_window.copy()
        max_lag = max(LAG_STEPS) if LAG_STEPS else 0
        for t in self.targets:
            if t not in df.columns:
                continue
            for lag in LAG_STEPS:
                col = f'{t}_lag_{lag}'
                if col not in df.columns:
                    df[col] = df[t].shift(lag)
        # Lagged activity signals (used by checkpoints trained with them; older
        # checkpoints read the original column named in their feature scaler).
        for c in OPTIONAL_GBDT_FEATURE_COLUMNS:
            if c in df.columns:
                for lag in ACTIVITY_LAG_STEPS:
                    col = f'{c}_lag_{lag}'
                    if col not in df.columns:
                        df[col] = df[c].shift(lag)
        df = df.iloc[max_lag:].copy().ffill().fillna(0.0)
        if len(df) < 50:
            return

        gbdt_hp = ((getattr(self.args, 'training_hyperparams', {}) or {})
                   .get('gbdt') or {})
        for t in self.targets:
            if t not in data_window.columns:
                continue
            # Per-target objective so retraining mirrors what initial training
            # used. e.g. NEST_UMAR DHW uses reg:tweedie + variance_power=1.5.
            t_cfg = gbdt_hp.get(t) or {}
            ft_cfg = t_cfg.get('_retrain', {}) or {}
            params = {
                'objective':     str(t_cfg.get('objective', 'reg:squarederror')),
                'learning_rate': float(ft_cfg.get('learning_rate', 0.05)),
                'max_depth':     int(ft_cfg.get('max_depth', 6)),
                'seed':          int(getattr(self.args, 'seed', 0) or 0),
            }
            for k, v in t_cfg.items():
                if k in ('objective', '_retrain'):
                    continue
                params[k] = v
            n_extra = int(ft_cfg.get('num_boost_round', num_boost_round))
            # 'refit' baseline: a fresh booster on the retrain window instead of
            # extra trees on top of the existing one.
            _refit = getattr(self.args, 'gbdt_retrain_mode', 'boost') == 'refit'
            if _refit:
                n_extra = int(getattr(self.args, 'gbdt_refit_rounds', 400))
            scaler_feat = self.gbdt_scalers_feat[t]
            scaler_tgt  = self.gbdt_scalers_tgt[t]
            if hasattr(scaler_feat, 'feature_names_in_'):
                cols = list(scaler_feat.feature_names_in_)
            else:
                exclude = {t, 'Unnamed: 0'}
                cols = [c for c in df.columns if c not in exclude]
            feat_data = df.copy()
            for c in cols:
                if c not in feat_data.columns:
                    feat_data[c] = 0.0
            feat_df = feat_data[cols].fillna(0.0)
            y_raw   = df[t].fillna(0.0).values

            X = scaler_feat.transform(feat_df.values)
            y = scaler_tgt.transform(y_raw.reshape(-1, 1)).ravel()

            dtrain = xgb.DMatrix(X, label=y)
            self.gbdt_models[t] = xgb.train(
                params, dtrain,
                num_boost_round=n_extra,
                xgb_model=(None if _refit else self.gbdt_models[t]),
            )

    def _replay(self, data_window: pd.DataFrame, start: int):
        """Rolling base-model forecasts over ``data_window`` from row ``start``.

        Returns ``{'lstm','gbdt','tft','act': {target: [arrays]}, 'start': [ts]}``
        or ``None`` when a target column is missing.
        """
        args    = self.args
        horizon = args.forecast_horizon
        seq_len = args.sequence_length
        if any(t not in data_window.columns for t in self.targets):
            return None
        wins = {k: {t: [] for t in self.targets} for k in ('lstm', 'gbdt', 'tft', 'act')}
        wins['start'] = []
        n = len(data_window)
        i = max(seq_len, int(start))
        while i + horizon <= n:
            actuals = {t: data_window[t].iloc[i:i + horizon].values for t in self.targets}
            if any(np.any(np.isnan(a)) for a in actuals.values()):
                i += horizon
                continue
            lstm_p, gbdt_p, tft_p, _ = self.predict_components(data_window, i)
            for t in self.targets:
                wins['lstm'][t].append(np.asarray(lstm_p[t], dtype=float))
                wins['gbdt'][t].append(np.asarray(gbdt_p[t], dtype=float))
                if t in tft_p:
                    wins['tft'][t].append(np.asarray(tft_p[t], dtype=float))
                wins['act'][t].append(np.asarray(actuals[t], dtype=float))
            wins['start'].append(data_window.index[i])
            i += horizon
        return wins

    def _tune_from_wins(self, wins) -> bool:
        """Grid-search the per-horizon weights on replayed windows.

        Each target is tuned on its own so a per-target GBDT floor can apply:
        the validation-selected floor from weights_meta.json when
        ``args.floor_source == 'validation'``, else the dataset / default one.
        """
        args    = self.args
        targets = self.targets
        if not wins or not targets:
            return False
        n = len(wins['act'][targets[0]])
        if n < 3 or not ({'lstm', 'gbdt'} <= self._members()):
            return False
        have_tft = all(len(wins['tft'][t]) == n for t in targets)
        floors = {}
        if getattr(args, 'floor_source', 'dataset') == 'validation':
            floors = (getattr(self, 'weights_meta', None) or {}).get('floor') or {}
        new = {}
        for t in targets:
            kw = {}
            f = floors.get(t)
            if f:
                kw = {'gbdt_floor_horizons': tuple(f['horizons']),
                      'gbdt_floor_weights':  tuple(f['weights'])}
            elif hasattr(args, 'gbdt_floor_horizons') and hasattr(args, 'gbdt_floor_weights'):
                kw = {'gbdt_floor_horizons': args.gbdt_floor_horizons,
                      'gbdt_floor_weights':  args.gbdt_floor_weights}
            w = tune_per_horizon_weights(
                {t: wins['lstm'][t]}, {t: wins['gbdt'][t]}, {t: wins['act'][t]},
                [t], args.forecast_horizon,
                tft_wins=({t: wins['tft'][t]} if have_tft else None), **kw,
            )
            new[t] = w[t]
        self.horizon_weights = new
        return True

    def _retune_weights(self, data_window: pd.DataFrame):
        """Legacy entry point: replay the whole window, then tune (in-sample)."""
        wins = self._replay(data_window, self.args.sequence_length)
        if wins:
            self._tune_from_wins(wins)

    def _snapshot_models(self) -> dict:
        import copy
        return {
            'lstm': {t: copy.deepcopy(m.state_dict()) for t, m in self.lstm_models.items()},
            'gbdt': {t: b.copy() for t, b in self.gbdt_models.items()},
            'tft':  {t: copy.deepcopy(m.state_dict()) for t, m in self.tft_models.items()},
        }

    def _restore_model(self, snap: dict, model: str, target: str) -> None:
        if model == 'gbdt':
            self.gbdt_models[target] = snap['gbdt'][target]
        elif model == 'lstm':
            self.lstm_models[target].load_state_dict(snap['lstm'][target])
            self.lstm_models[target].eval()
        elif model == 'tft' and target in self.tft_models:
            self.tft_models[target].load_state_dict(snap['tft'][target])
            self.tft_models[target].eval()

    def _finetune_tft(self, data_window: pd.DataFrame):
        """Fine-tune each loaded TFT model on the recent window.

        Mirrors ``_finetune_lstm`` but uses the TFT-specific scaler (when
        loaded) and skips the LSTM-only season-pure filtering — TFT's VSN +
        attention is robust to mixed-season inputs and the heating /cooling
        models were trained on season-pure sequences anyway, so filtering
        again here would over-constrain the recent window. Without this,
        TFT stays frozen across retrains, drifting further from the
        adapted LSTM/GBDT pair (which is what the run20260520-* logs show).
        """
        if not self.tft_models:
            return
        scaler = self.tft_scaler or self.lstm_scaler
        cols = (list(scaler.feature_names_in_)
                if hasattr(scaler, 'feature_names_in_') else list(LSTM_COLUMNS))
        log1p_targets = (self.lstm_log1p_targets
                          if scaler is self.lstm_scaler else None)
        subset = data_window[[c for c in cols if c in data_window.columns]].dropna().copy()
        if len(subset) == 0:
            return
        # Add any columns missing in the recent window so scaler.transform
        # doesn't blow up on a single-column drop in the upstream CSV.
        for c in cols:
            if c not in subset.columns:
                subset[c] = 0.0
        subset = subset[cols]
        if log1p_targets:
            for t in log1p_targets:
                if t in subset.columns:
                    subset[t] = np.log1p(np.clip(subset[t].values, a_min=0.0, a_max=None))
        scaled = scaler.transform(subset)
        X_all, y_all = create_sequences_for_finetuning(
            scaled, self.args.sequence_length, self.args.forecast_horizon
        )
        if len(X_all) < 10:
            return

        # Re-use the LSTM peak-aware loss knobs — TFT was trained with the
        # same asymmetric_mse + smoothness penalty in transformer/train.py.
        peak_alpha   = float(getattr(self.args, 'peak_weight_alpha', 0.0))
        peak_thresh  = float(getattr(self.args, 'peak_threshold',    0.3))
        loss_type    = str  (getattr(self.args, 'loss_type',         'mse'))
        huber_delta  = float(getattr(self.args, 'huber_delta',       0.1))
        uw_heating   = float(getattr(self.args, 'under_weight_heating', 1.0))
        uw_other     = float(getattr(self.args, 'under_weight_other',   1.0))
        min_target_f = float(getattr(self.args, 'min_target_fraction', 0.02))
        tft_hp = ((getattr(self.args, 'training_hyperparams', {}) or {})
                  .get('tft') or {})

        for t, model in self.tft_models.items():
            if t not in cols:
                continue
            t_idx = cols.index(t)
            y_s = y_all[:, :, t_idx : t_idx + 1]
            if len(X_all) < 10:
                continue
            t_hp   = tft_hp.get(t) or {}
            ft_cfg = t_hp.get('_retrain', {}) or {}
            # Defaults reflect the post-clamp regime: 8 epochs at lr=5e-5
            # are more conservative on LR than the LSTM-CNN's 6 epochs at
            # 1e-4 because attention layers re-route easily under aggressive
            # updates — half the LR, 8/6 the passes preserves a similar
            # gradient budget without destabilising the heads. Bumped from
            # the earlier 5-epoch default for the same reason as the LSTM:
            # the previous 5 epochs were too few to adapt to a real regime
            # shift inside the retrain window.
            batch_size = int(ft_cfg.get('batch_size', 16))
            epochs     = int(ft_cfg.get('epochs',     8))
            lr         = float(ft_cfg.get('learning_rate', 5e-5))
            ow = float(t_hp.get('over_weight', 1.0))
            uw = uw_heating if t == 'thermal_heating_kw' else uw_other

            X_t = torch.tensor(X_all, dtype=torch.float32).to(self.device)
            y_t = torch.tensor(y_s,   dtype=torch.float32).to(self.device)
            loader = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True)

            model.train()
            opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
            for _ in range(epochs):
                for xb, yb in loader:
                    opt.zero_grad()
                    pred = model(xb)
                    loss = _asymmetric_mse(
                        pred, yb,
                        over_weight=ow,
                        eps=min_target_f,
                        under_weight=uw,
                        peak_threshold=peak_thresh,
                        peak_weight_alpha=peak_alpha,
                        loss_type=loss_type,
                        huber_delta=huber_delta,
                    )
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
            model.eval()

    def _retrain(self, data_window: pd.DataFrame):
        """Retraining step: fine-tune the member models, then re-tune weights.

        Options on ``self.args`` (defaults reproduce the original behaviour):

        * ``retune_holdout_frac`` (0): when > 0 the last fraction of the window
          is held out. Models are fine-tuned on the earlier part only and the
          weights are tuned on forecasts for the held-out part, so the weight
          search no longer sees data the models were just fitted to.
        * ``retrain_accept`` (False): with a hold-out, each fine-tuned model is
          kept for a target only if its hold-out MAE is not worse than that of
          the model it replaces; otherwise the previous model is restored.
        * ``gbdt_retrain_mode == 'auto'`` (needs ``retrain_accept``): the GBDT of
          each target is the best of three on the hold-out: the previous model,
          the previous model with extra trees ('boost') and a new model fitted
          on the retrain window ('refit').
        * ``weight_retune`` (True): set False to keep the weights fixed.
        * ``members``: only member models are fine-tuned.

        ``self.last_retrain`` records what happened (for the run artefacts).
        """
        args    = self.args
        members = self._members()
        horizon = args.forecast_horizon
        seq_len = args.sequence_length
        n       = len(data_window)
        frac    = float(getattr(args, 'retune_holdout_frac', 0.0) or 0.0)
        split   = int(n * (1.0 - frac)) if 0.0 < frac < 1.0 else n
        hold    = split < n and split >= seq_len and (n - split) >= 3 * horizon
        fit_win = data_window.iloc[:split] if hold else data_window
        accept  = bool(getattr(args, 'retrain_accept', False)) and hold

        snap = old = None
        if accept:
            snap = self._snapshot_models()
            old  = self._replay(data_window, split)

        if 'lstm' in members:
            self._finetune_lstm(fit_win)
        gbdt_auto = ('gbdt' in members and accept
                     and getattr(args, 'gbdt_retrain_mode', 'boost') == 'auto')
        if 'gbdt' in members:
            if gbdt_auto:
                args.gbdt_retrain_mode = 'boost'
                try:
                    self._finetune_gbdt(fit_win)
                finally:
                    args.gbdt_retrain_mode = 'auto'
            else:
                self._finetune_gbdt(fit_win)
        if 'tft' in members:
            self._finetune_tft(fit_win)

        wins = self._replay(data_window, split if hold else seq_len)

        # 'auto': also fit the refit candidate and replay it on the hold-out.
        refit_alt = None
        if gbdt_auto and wins and wins['start']:
            boost_models = {t: self.gbdt_models[t] for t in self.targets}
            for t in self.targets:
                self._restore_model(snap, 'gbdt', t)
            args.gbdt_retrain_mode = 'refit'
            try:
                self._finetune_gbdt(fit_win)
            finally:
                args.gbdt_retrain_mode = 'auto'
            alt = self._replay(data_window, split)
            refit_models = {t: self.gbdt_models[t] for t in self.targets}
            for t in self.targets:
                self.gbdt_models[t] = boost_models[t]
            if alt and len(alt['start']) == len(wins['start']):
                refit_alt = (refit_models, alt['gbdt'])

        decisions = {}
        if accept and old and wins and wins['start'] and len(old['start']) == len(wins['start']):
            for m in ('lstm', 'gbdt', 'tft'):
                if m not in members:
                    continue
                for t in self.targets:
                    if not wins[m][t] or len(old[m][t]) != len(wins[m][t]):
                        continue
                    A     = np.stack(wins['act'][t])
                    e_new = float(np.mean(np.abs(A - np.stack(wins[m][t]))))
                    e_old = float(np.mean(np.abs(A - np.stack(old[m][t]))))
                    if (m == 'gbdt' and refit_alt is not None and refit_alt[1].get(t)
                            and len(refit_alt[1][t]) == len(wins[m][t])):
                        e_ref = float(np.mean(np.abs(A - np.stack(refit_alt[1][t]))))
                        # ties: previous model first, then boost, then refit
                        best = min((e_old, 0, 'old'), (e_new, 1, 'boost'), (e_ref, 2, 'refit'))[2]
                        decisions[f'{m}:{t}'] = {'old_mae': e_old, 'new_mae': e_new,
                                                 'refit_mae': e_ref, 'chosen': best,
                                                 'kept_new': bool(best != 'old')}
                        if best == 'old':
                            self._restore_model(snap, m, t)
                            wins[m][t] = old[m][t]
                        elif best == 'refit':
                            self.gbdt_models[t] = refit_alt[0][t]
                            wins[m][t] = refit_alt[1][t]
                        continue
                    keep  = e_new <= e_old
                    decisions[f'{m}:{t}'] = {'old_mae': e_old, 'new_mae': e_new,
                                             'kept_new': bool(keep)}
                    if not keep:
                        self._restore_model(snap, m, t)
                        wins[m][t] = old[m][t]
            rejected = [k for k, v in decisions.items() if not v['kept_new']]
            if rejected:
                print(f'  [retrain] previous model restored for: {rejected}')

        tuned = False
        if bool(getattr(args, 'weight_retune', True)) and wins:
            tuned = self._tune_from_wins(wins)
        self.last_retrain = {
            'holdout':         bool(hold),
            'fit_rows':        int(len(fit_win)),
            'holdout_rows':    int(n - split) if hold else 0,
            'weights_retuned': bool(tuned),
            'accept':          decisions,
            'wins':            wins,
        }

    def adaptive_forecast(self, data: pd.DataFrame) -> list:
        """Run rolling adaptive forecasts over the full dataset."""
        args    = self.args
        n       = len(data)
        history = []
        i       = args.start_index

        while i + args.forecast_horizon <= n:
            result = self.predict(data, i)

            end     = i + args.forecast_horizon
            actuals = {t: data[t].iloc[i:end].values for t in self.targets}
            metrics = calculate_metrics_adaptive(actuals, result, data=data, start_idx=i)
            history.append(metrics)

            if len(history) % args.evaluation_frequency == 0:
                should, reasons = self._should_retrain(metrics)
                if should and self.retrain_count < args.max_retrains:
                    win_start = max(0, i - args.retrain_window)
                    self._retrain(data.iloc[win_start:i])
                    self.retrain_count += 1
                    print(f'[adaptive] retrain #{self.retrain_count} at step {i}: {reasons}')

            i += args.forecast_horizon

        self.performance_history = history
        return history
