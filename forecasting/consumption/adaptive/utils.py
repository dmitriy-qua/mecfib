"""Utility functions for adaptive consumption forecasting."""

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from forecasting.constants import (
    TARGET_THERMAL_HEATING, TARGET_THERMAL_COOLING, TARGET_DHW, TARGET_ELECTRIC,
)
from forecasting.consumption.constants import (
    LSTM_COLUMNS, TARGET_COLUMNS, LAG_STEPS, OPTIONAL_GBDT_FEATURE_COLUMNS, ACTIVITY_LAG_STEPS,
)
from forecasting.utils import create_sequences, calculate_mase, SEASONAL_NAIVE_PERIOD


def _apply_log1p_inplace(df, log1p_targets):
    """Apply log1p to listed target columns of ``df`` in place. No-op when the
    list is empty (legacy / linear-scaled runs).

    The training pipeline applies log1p BEFORE scaler.fit on these columns; any
    code that calls scaler.transform on raw data must do the same first or the
    transform output will be in the wrong space.
    """
    if not log1p_targets:
        return df
    for t in log1p_targets:
        if t in df.columns:
            df[t] = np.log1p(np.clip(df[t].values, a_min=0.0, a_max=None))
    return df


def prepare_lstm_input(data, start_idx, sequence_length, scaler,
                       log1p_targets=None):
    end_idx = start_idx + sequence_length
    # Use the column order the scaler was actually fit on to avoid sklearn
    # feature-name warnings and to handle any column subset correctly.
    cols = list(scaler.feature_names_in_) if hasattr(scaler, 'feature_names_in_') else list(LSTM_COLUMNS)
    seq = data.iloc[start_idx:end_idx][cols].ffill().fillna(0).copy()
    _apply_log1p_inplace(seq, log1p_targets)
    scaled = scaler.transform(seq)   # pass DataFrame so names are validated
    return torch.tensor(scaled, dtype=torch.float32).unsqueeze(0)


def _align_features(row, target_col, scaler_features):
    if hasattr(scaler_features, 'feature_names_in_'):
        cols = list(scaler_features.feature_names_in_)
        feat = {c: row[c].values if c in row.columns else [0] for c in cols}
        return pd.DataFrame(feat)
    exclude = {target_col, 'Unnamed: 0'}
    return row[[c for c in row.columns if c not in exclude]]


def prepare_gbdt_features(data, idx, target_col, scaler_features):
    row = data.iloc[idx:idx+1]
    feats = _align_features(row, target_col, scaler_features)
    return scaler_features.transform(feats)


def forecast_lstm_cnn_single(model, input_tensor, scaler, target: str, device,
                              log1p_targets=None,
                              clamp_scaled_to_unit: bool = False,
                              max_value_kw: float | None = None):
    """
    Run inference for a single-target LSTM-CNN model.

    Parameters
    ----------
    model         : LSTM_CNN_Model trained with num_features=1
    input_tensor  : (1, seq_len, n_features) tensor (already in scaled space —
                    callers must pass through ``prepare_lstm_input`` so any
                    log1p transform is already applied to target columns).
    scaler        : shared MinMaxScaler fitted on all LSTM_COLUMNS
    target        : name of the target column (e.g. 'thermal_heating_kw')
    device        : torch device string
    log1p_targets : iterable of target columns whose scaler was fit in log1p
                    space. When ``target`` is in this set, expm1 is applied
                    after the MinMax inverse to return raw kW.
    clamp_scaled_to_unit : when True, clip the raw scaled output to [0, 1]
                    before inverse-transform. The TFT head is a raw Linear
                    layer with no sigmoid, so it can leak outside [0, 1];
                    combined with the training-time ``target_clip_quantile``
                    lowering ``data_max_``, even a small overshoot becomes a
                    multi-x overprediction after the MinMax inverse. Default
                    off so existing callers see no behaviour change.
    max_value_kw  : optional absolute cap on the returned values, applied
                    after inverse transform. Acts as a last-resort safety
                    net against architecture/scaler mismatches; default None
                    leaves the upper bound unconstrained.

    Returns
    -------
    np.ndarray of shape (horizon,) in original kW units
    """
    input_tensor = input_tensor.to(device)
    with torch.no_grad():
        pred_scaled = model(input_tensor).cpu().numpy().flatten()   # (horizon,)

    if clamp_scaled_to_unit:
        pred_scaled = np.clip(pred_scaled, 0.0, 1.0)

    # Resolve the column index from the scaler's own column list so the index
    # is always consistent with what the scaler was actually fit on.
    cols  = list(scaler.feature_names_in_) if hasattr(scaler, 'feature_names_in_') else list(LSTM_COLUMNS)
    t_idx = cols.index(target)

    # Inverse-transform a single column directly — avoids building a dummy
    # full-width array and eliminates sklearn feature-name warnings.
    t_min = float(scaler.data_min_[t_idx])
    t_max = float(scaler.data_max_[t_idx])
    inv   = pred_scaled * (t_max - t_min) + t_min
    if log1p_targets and target in log1p_targets:
        inv = np.expm1(np.clip(inv, a_min=0.0, a_max=None))
    inv = np.clip(inv, 0.0, None)
    if max_value_kw is not None:
        inv = np.minimum(inv, float(max_value_kw))
    return inv


# Same input/output contract as forecast_lstm_cnn_single — the TFT was trained
# on the same scaler and emits (1, horizon, 1) outputs in the same scaled space.
# The TFT-specific path in AdaptiveForecaster._predict_tft enables
# ``clamp_scaled_to_unit`` so the raw-Linear head can't leak outside [0, 1].
forecast_transformer_single = forecast_lstm_cnn_single


def forecast_gbdt_iterative(models, scalers_feat, scalers_tgt, targets,
                             data, start_idx, horizon):
    """
    Iterative GBDT forecast for all consumption targets.

    Lag features ({target}_lag_{lag}) are built on-the-fly: for lags that
    land before the forecast window, actual historical values are used; for
    lags that land inside the forecast window, already-predicted values are
    used.  Contemporaneous target values are never used as features, removing
    the train/test leakage that existed in the previous implementation.

    Parameters
    ----------
    models       : dict[target_name -> xgb.Booster]
    scalers_feat : dict[target_name -> fitted feature scaler]
    scalers_tgt  : dict[target_name -> fitted target scaler]
    targets      : list of target column names (ordered)
    data         : full DataFrame with datetime index
    start_idx    : row index of first forecast step
    horizon      : number of steps to predict

    Returns
    -------
    dict[target_name -> np.ndarray (horizon,)]
    """
    preds = {t: [] for t in targets}
    # pred_store[abs_idx] = {target: predicted_value}
    pred_store: dict = {}

    for step in range(horizon):
        abs_idx = start_idx + step
        row = data.iloc[abs_idx:abs_idx + 1].copy()

        # Build lag features, mixing actuals (past) and predictions (forecast window)
        for lag in LAG_STEPS:
            lag_idx = abs_idx - lag
            for t in targets:
                col = f'{t}_lag_{lag}'
                if lag_idx < 0:
                    row[col] = 0.0
                elif lag_idx >= start_idx and lag_idx in pred_store:
                    row[col] = pred_store[lag_idx][t]
                else:
                    row[col] = float(data[t].iloc[lag_idx])

        # Lagged activity signals: only values observed before the forecast was
        # issued (never a row inside the forecast window, whatever the horizon).
        for c in OPTIONAL_GBDT_FEATURE_COLUMNS:
            if c in data.columns:
                for lag in ACTIVITY_LAG_STEPS:
                    j = min(abs_idx - lag, start_idx - 1)
                    row[f'{c}_lag_{lag}'] = float(data[c].iloc[j]) if j >= 0 else 0.0

        step_preds = {}
        for t in targets:
            feat        = _align_features(row, t, scalers_feat[t])
            feat_scaled = scalers_feat[t].transform(feat)
            pred_s = models[t].predict(xgb.DMatrix(feat_scaled))[0]
            val = max(0.0, scalers_tgt[t].inverse_transform([[pred_s]])[0][0])
            step_preds[t] = val
            preds[t].append(val)

        pred_store[abs_idx] = step_preds

    return {t: np.array(v) for t, v in preds.items()}


def combine_predictions(lstm_preds, gbdt_preds, weights, tft_preds=None):
    """
    Weighted ensemble of LSTM, GBDT, and optionally TFT outputs.

    Parameters
    ----------
    lstm_preds : dict[target -> np.ndarray (horizon,)]
    gbdt_preds : dict[target -> np.ndarray (horizon,)]
    weights    : dict[target -> tuple]. Each tuple is either
                 (lstm_weight, gbdt_weight) or (lstm_weight, gbdt_weight, tft_weight).
    tft_preds  : optional dict[target -> np.ndarray (horizon,)]

    Returns
    -------
    dict[target -> np.ndarray (horizon,)]
    """
    combined = {}
    for t in lstm_preds:
        w = weights[t]
        if tft_preds is not None and t in tft_preds and len(w) >= 3:
            lw, gw, tw = w[0], w[1], w[2]
            total = lw + gw + tw
            combined[t] = ((lw / total) * lstm_preds[t]
                           + (gw / total) * gbdt_preds[t]
                           + (tw / total) * tft_preds[t])
        else:
            lw, gw = w[0], w[1]
            total = lw + gw
            combined[t] = (lw / total) * lstm_preds[t] + (gw / total) * gbdt_preds[t]
    return combined


# ---------------------------------------------------------------------------
# Per-horizon weight tuning
# ---------------------------------------------------------------------------

def tune_per_horizon_weights(
    lstm_wins: dict,
    gbdt_wins: dict,
    act_wins: dict,
    targets: list,
    horizon: int,
    tft_wins: dict | None = None,
    grid_size: int = 21,
    lstm_ceil_horizons: tuple = (1, 12, 24),
    lstm_ceil_weights:  tuple = (0.10, 0.40, 0.60),
    tft_ceil_horizons:  tuple = (1, 12, 24),
    tft_ceil_weights:   tuple = (0.10, 0.30, 0.40),
    gbdt_floor_horizons: tuple = (1, 12, 24),
    gbdt_floor_weights:  tuple = (0.80, 0.45, 0.30),
) -> dict:
    """
    For each target and horizon step h, find weights minimising MAE on OOF windows.

    Two regimes:

    * **2-way** (``tft_wins`` is None): scan ``w_lstm ∈ [0, w_lstm_max(h)]`` along
      a 1-D grid, with a monotone non-decreasing constraint and a horizon-
      dependent ceiling encoding the prior that GBDT dominates early-horizon
      and LSTM takes over later. Returns ``{target: np.ndarray(horizon,)}``
      where each entry is the LSTM weight (GBDT weight = 1 − w_lstm).

    * **3-way** (``tft_wins`` provided): scan the simplex
      ``(w_lstm, w_tft)`` with ``w_lstm + w_tft ≤ 1``, with per-model
      horizon-dependent ceilings. The remaining mass goes to GBDT. Returns
      ``{target: {'lstm': arr, 'gbdt': arr, 'tft': arr}}``.

    **GBDT floor.** An explicit horizon-dependent lower bound on the GBDT
    weight, ``w_gbdt(h) ≥ floor(h)``, is now enforced on top of the LSTM /
    TFT ceilings. Previously, the 3-way ceilings only implied GBDT lower
    bounds of ``{0.80, 0.30, 0.00}`` at ``h ∈ {1, 12, 24}``, meaning the
    blend could legally drop GBDT to zero at the far end of the horizon —
    a structural mismatch with the empirical finding (KSRP run-`20260520-*`)
    that GBDT is the per-target-best learner on 3 of 4 heads even at
    ``h+24``. The default floor ``(0.80, 0.45, 0.30)`` tightens the
    simplex slice at the long end of the horizon, where the deep-learning
    models' confidence interval is widest. Sites where GBDT is more
    dominant (e.g. KSRP) can override the anchors via the per-dataset
    ``ADAPTIVE_GBDT_FLOOR_WEIGHTS`` constant; sites where the deep
    learners win at the long end can flatten the floor to (0.80, 0.30, 0.0)
    to recover the legacy permissive behaviour.

    Default 3-way ceilings + floor:
        h+1  → LSTM ≤ 0.10, TFT ≤ 0.10, GBDT ≥ 0.80
        h+12 → LSTM ≤ 0.40, TFT ≤ 0.30, GBDT ≥ 0.45
        h+24 → LSTM ≤ 0.60, TFT ≤ 0.40, GBDT ≥ 0.30

    Grid size 21 → step 0.05; 21² = 441 candidates per horizon step.

    Parameters
    ----------
    lstm_wins, gbdt_wins, act_wins : dict[target -> list[np.ndarray(horizon,)]]
    tft_wins                        : optional dict (same shape) — enables 3-way.
    targets                         : ordered list of target column names
    horizon                         : forecast horizon
    grid_size                       : candidate count per axis
    lstm_ceil_horizons / weights    : anchor curve for LSTM weight ceiling
    tft_ceil_horizons / weights     : anchor curve for TFT weight ceiling (3-way only)
    gbdt_floor_horizons / weights   : anchor curve for the GBDT lower bound
                                      (applies in both regimes).
    """
    three_way = tft_wins is not None
    grid      = np.linspace(0.0, 1.0, grid_size)
    weights: dict = {}

    for t in targets:
        n_windows = len(act_wins[t])

        if not three_way:
            w_per_h = np.zeros(horizon)
            w_prev  = 0.0
            for h in range(horizon):
                # LSTM ceiling + GBDT floor co-determine the LSTM max:
                # 1 - w_lstm = w_gbdt ≥ floor ⇒ w_lstm ≤ 1 - floor.
                w_max_ceil  = float(np.interp(h + 1, lstm_ceil_horizons, lstm_ceil_weights))
                w_gbdt_floor = float(np.interp(h + 1, gbdt_floor_horizons, gbdt_floor_weights))
                w_max = min(w_max_ceil, 1.0 - w_gbdt_floor)
                grid_h = grid[(grid >= w_prev - 1e-9) & (grid <= w_max + 1e-9)]
                if len(grid_h) == 0:
                    w_per_h[h] = w_prev
                    continue
                lstm_h = np.array([lstm_wins[t][i][h] for i in range(n_windows)])
                gbdt_h = np.array([gbdt_wins[t][i][h] for i in range(n_windows)])
                act_h  = np.array([act_wins[t][i][h]  for i in range(n_windows)])
                mae = np.array([
                    np.mean(np.abs(act_h - (w * lstm_h + (1.0 - w) * gbdt_h)))
                    for w in grid_h
                ])
                w_per_h[h] = grid_h[np.argmin(mae)]
                w_prev     = w_per_h[h]
            weights[t] = w_per_h
            continue

        # ── 3-way simplex search ────────────────────────────────────────────
        w_l = np.zeros(horizon)
        w_g = np.zeros(horizon)
        w_t = np.zeros(horizon)
        wl_prev, wt_prev = 0.0, 0.0

        for h in range(horizon):
            wl_max = float(np.interp(h + 1, lstm_ceil_horizons, lstm_ceil_weights))
            wt_max = float(np.interp(h + 1, tft_ceil_horizons,  tft_ceil_weights))
            # GBDT floor: w_gbdt ≥ floor(h) ⇒ w_lstm + w_tft ≤ 1 - floor(h).
            wg_floor = float(np.interp(h + 1, gbdt_floor_horizons, gbdt_floor_weights))
            sum_max  = 1.0 - wg_floor
            lstm_h = np.array([lstm_wins[t][i][h] for i in range(n_windows)])
            gbdt_h = np.array([gbdt_wins[t][i][h] for i in range(n_windows)])
            tft_h  = np.array([tft_wins[t][i][h]  for i in range(n_windows)])
            act_h  = np.array([act_wins[t][i][h]  for i in range(n_windows)])

            best_mae = np.inf
            best     = (wl_prev, wt_prev)
            for wl in grid:
                if wl < wl_prev - 1e-9 or wl > wl_max + 1e-9:
                    continue
                for wt in grid:
                    if wt < wt_prev - 1e-9 or wt > wt_max + 1e-9:
                        continue
                    if wl + wt > sum_max + 1e-9:
                        # GBDT floor violated.
                        continue
                    wg = 1.0 - wl - wt
                    if wg < -1e-9:
                        continue
                    wg = max(0.0, wg)
                    mae = float(np.mean(
                        np.abs(act_h - (wl * lstm_h + wg * gbdt_h + wt * tft_h))
                    ))
                    if mae < best_mae:
                        best_mae = mae
                        best     = (wl, wt)
            wl, wt = best
            wg     = max(0.0, 1.0 - wl - wt)
            w_l[h], w_g[h], w_t[h] = wl, wg, wt
            wl_prev, wt_prev       = wl, wt

        weights[t] = {'lstm': w_l, 'gbdt': w_g, 'tft': w_t}

    return weights


def combine_with_horizon_weights(
    lstm_preds: dict,
    gbdt_preds: dict,
    horizon_weights: dict,
    tft_preds: dict | None = None,
) -> dict:
    """
    Combine base-model forecasts using per-horizon weights.

    Schema support:
      * 2-way (legacy):  ``horizon_weights[t] = np.ndarray(horizon,)`` — LSTM
        weight per step; GBDT weight = 1 − w_lstm.
      * 3-way (new):     ``horizon_weights[t] = {'lstm': arr, 'gbdt': arr, 'tft': arr}``.
        ``tft_preds`` must be supplied for the TFT component to contribute.

    Falls back to a simple average for any target without weights.
    """
    combined = {}
    for t in lstm_preds:
        if t not in horizon_weights:
            if tft_preds is not None and t in tft_preds:
                combined[t] = (lstm_preds[t] + gbdt_preds[t] + tft_preds[t]) / 3.0
            else:
                combined[t] = 0.5 * lstm_preds[t] + 0.5 * gbdt_preds[t]
            continue

        w = horizon_weights[t]
        if isinstance(w, dict):
            wl = np.asarray(w['lstm'])
            wg = np.asarray(w['gbdt'])
            wt = np.asarray(w.get('tft', np.zeros_like(wl)))
            tft_p = (tft_preds[t] if (tft_preds is not None and t in tft_preds)
                     else np.zeros_like(lstm_preds[t]))
            combined[t] = wl * lstm_preds[t] + wg * gbdt_preds[t] + wt * tft_p
        else:
            wl = np.asarray(w)
            combined[t] = wl * lstm_preds[t] + (1.0 - wl) * gbdt_preds[t]
    return combined


def calculate_metrics_adaptive(actuals, preds, data=None, start_idx: int = None):
    """
    Parameters
    ----------
    actuals, preds : dict[target -> np.ndarray (horizon,)]
    data           : full DataFrame with target columns (optional).
                     When provided together with start_idx, MASE is computed
                     using a weekly seasonal naive (lag = SEASONAL_NAIVE_PERIOD).
    start_idx      : row index of the first forecast step in data.

    Returns
    -------
    dict with per-target mae / rmse / r2 / mase + overall averages.

    MASE accounting
    ---------------
    For each target the per-window dict carries three MASE-related entries:

      * ``<t>_mase``           — the per-window ratio MAE_model / MAE_naive
                                 (RelMAE on that window). Useful for diagnostics
                                 and for the per-target retrain trigger of §2.5,
                                 but NOT the right basis for cross-window
                                 aggregation: averaging ratios over windows is
                                 biased upward whenever the denominator (naive
                                 MAE on that window) varies, which it does
                                 heavily on seasonally-zero targets (heating in
                                 summer, cooling in winter) and on any target
                                 whose autocorrelation at lag 168 is irregular.
      * ``<t>_mase_num_sum``   — sum-over-horizon of ``|y - p|`` for this
                                 window. Becomes the numerator term of the
                                 pooled cross-window MASE.
      * ``<t>_mase_den_sum``   — sum-over-horizon of ``|y - y_naive|`` for this
                                 window. Becomes the denominator term of the
                                 pooled cross-window MASE.

    The aggregator (in run.py / tune_weights.py) sums ``<t>_mase_num_sum`` and
    ``<t>_mase_den_sum`` independently across all windows and computes the
    pooled per-target MASE as the ratio of the two sums — equivalent to
    weighting every horizon-step error equally regardless of which window it
    came from. This eliminates the mean-of-ratios bias that the legacy
    ``mean(<t>_mase)`` aggregation suffered from (DHW on KSRP run
    `20260523-082538`: 3.82 mean-of-ratios vs 2.46 pooled, ~55 % inflation).
    """
    use_mase = (data is not None and start_idx is not None
                and start_idx >= SEASONAL_NAIVE_PERIOD)

    result = {}
    for t in actuals:
        y = actuals[t]
        p = preds[t]
        result[f'{t}_mae']  = float(mean_absolute_error(y, p))
        result[f'{t}_rmse'] = float(np.sqrt(mean_squared_error(y, p)))
        # Per-window R² is intentionally NOT computed here. Forecast windows are
        # short (typically 24 h) and many windows have near-constant actuals
        # (off-season heating/cooling, zero-heavy DHW). When SS_tot is tiny,
        # R² explodes to extreme negatives and any mean across windows is then
        # dominated by those outliers. The aggregation layer computes a single
        # *pooled* R² per target from the concatenated y/p across all windows
        # instead — see pooled_r2_per_target() below.
        result[f'{t}_r2'] = float('nan')

        if use_mase:
            horizon   = len(y)
            naive_idx = start_idx - SEASONAL_NAIVE_PERIOD
            y_naive   = data[t].iloc[naive_idx : naive_idx + horizon].values
            # Per-window ratio (diagnostic + trigger), plus the two sums the
            # aggregator needs for the pooled cross-window MASE.
            num_sum = float(np.sum(np.abs(y - p)))
            den_sum = float(np.sum(np.abs(y - y_naive)))
            result[f'{t}_mase']         = calculate_mase(y, p, y_naive)
            result[f'{t}_mase_num_sum'] = num_sum
            result[f'{t}_mase_den_sum'] = den_sum
        else:
            result[f'{t}_mase']         = float('nan')
            result[f'{t}_mase_num_sum'] = float('nan')
            result[f'{t}_mase_den_sum'] = float('nan')

    result['overall_mae']  = float(np.mean([result[f'{t}_mae']  for t in actuals]))
    result['overall_rmse'] = float(np.mean([result[f'{t}_rmse'] for t in actuals]))
    result['overall_r2']   = float('nan')   # pooled at aggregation time
    # Per-window "overall MASE" stays as the mean-of-per-target-ratios (used by
    # the threshold trigger of §2.5 and by the rolling-MASE plot). The pooled
    # cross-window per-target MASE is computed at aggregation time from
    # <t>_mase_num_sum / <t>_mase_den_sum and replaces the legacy mean-of-ratios
    # number in the summary reports.
    mase_vals = [result[f'{t}_mase'] for t in actuals]
    result['overall_mase'] = float(np.nanmean(mase_vals)) if any(
        not np.isnan(v) for v in mase_vals) else float('nan')
    return result


def check_threshold_exceeded(metrics, mae_thr, rmse_thr, mase_thr=None,
                              per_target_mase_thr=None, targets=None):
    """
    Return ``(triggered, reasons)``. Fires when *any* of:

      * ``overall_mae`` exceeds ``mae_thr``
      * ``overall_rmse`` exceeds ``rmse_thr``
      * ``overall_mase`` exceeds ``mase_thr`` (when provided)
      * any target's per-target MASE exceeds its entry in
        ``per_target_mase_thr`` (when provided)

    Per-target gating is needed for sites where one target's scale is so much
    smaller than the others that its MAE never moves the overall average
    enough to clear the global threshold — KSRP DHW (mean ~55 kW vs heating
    ~150 kW) is the canonical case. Without it, a chronically-bad target sits
    unchecked because the global thresholds are tuned to the dominant scale.
    """
    reasons = []
    if metrics['overall_mae']  > mae_thr:
        reasons.append(f"MAE {metrics['overall_mae']:.4f} > {mae_thr}")
    if metrics['overall_rmse'] > rmse_thr:
        reasons.append(f"RMSE {metrics['overall_rmse']:.4f} > {rmse_thr}")
    if (mase_thr is not None and not np.isnan(metrics.get('overall_mase', float('nan')))
            and metrics['overall_mase'] > mase_thr):
        reasons.append(f"MASE {metrics['overall_mase']:.4f} > {mase_thr}")
    if per_target_mase_thr and targets:
        for t in targets:
            thr = per_target_mase_thr.get(t)
            if thr is None:
                continue
            val = metrics.get(f'{t}_mase')
            if val is None or np.isnan(val):
                continue
            if val > thr:
                reasons.append(f"{t}.MASE {val:.4f} > {thr}")
    return len(reasons) > 0, reasons


def apply_naive_anchor(combined: dict, data: pd.DataFrame, start_idx: int,
                        horizon: int, anchor_weights: dict | None) -> dict:
    """Blend combined ensemble predictions with a weekly-seasonal-naive baseline.

    For each target ``t`` with an anchor weight ``alpha = anchor_weights[t] > 0``:

        final[t][h] = (1 - alpha) * combined[t][h] + alpha * y[t][start_idx - 168 + h]

    The naive baseline is the actual value 168 hours earlier (one full week,
    same hour-of-day, same day-of-week). When the lookback would underrun the
    available history (``start_idx < 168``), the anchor degrades to a no-op for
    that window.

    Why this exists
    ---------------
    On KSRP, the campus DHW schedule inverted around 2016 (morning-peak →
    night-flat) but the model was trained on the 2002-2011 morning-peak
    regime. The post-2016 diurnal shape is unrecoverable from the training
    features alone, yet still highly autocorrelated week-over-week — so the
    weekly naive achieves MAE ~5 kW where the trained model lands at ~12 kW
    and an R² of ~-2. Anchoring the ensemble to last week's actuals gives the
    model credit for short-term weather/calendar signal while preventing the
    learned-prior diurnal shape from poisoning the prediction.

    A weight of 0 (or no entry) leaves the target untouched, so the default
    behaviour is a no-op for targets and datasets that don't opt in.
    """
    if not anchor_weights:
        return combined
    out = dict(combined)
    naive_start = start_idx - SEASONAL_NAIVE_PERIOD
    if naive_start < 0:
        return out
    naive_end = naive_start + horizon
    if naive_end > len(data):
        return out
    for t, alpha in anchor_weights.items():
        if not alpha or t not in out:
            continue
        a = float(alpha)
        if a <= 0.0:
            continue
        a = min(a, 1.0)
        if t not in data.columns:
            continue
        y_naive = data[t].iloc[naive_start:naive_end].to_numpy(dtype=float, copy=False)
        if y_naive.shape[0] != horizon:
            continue
        out[t] = (1.0 - a) * np.asarray(out[t], dtype=float) + a * y_naive
    return out



def apply_physical_constraints(
    preds: dict,
    data,
    start_idx: int,
    horizon: int,
    inactivity_gates: dict | None = None,
    apply_seasonal_gate: bool = True,
) -> dict:
    """
    Enforce physical validity on combined predictions:
      - All consumption values clipped to >= 0.
      - Heating demand zeroed outside heating season (when ``is_heating_season``
        column is present — legacy gate inherited from HouseZero).
      - Data-driven inactivity gate: any target whose actuals over the recent
        lookback window are below threshold gets its horizon forced to zero.
        This catches sensor outages or equipment-off periods the model has no
        feature to learn from (e.g. NEST_UMAR DHW dropping to zero for weeks
        at a time).

    Parameters
    ----------
    preds            : dict[target -> np.ndarray (horizon,)]
    data             : full DataFrame; reads ``is_heating_season`` (optional)
                       and any target columns referenced by ``inactivity_gates``.
    start_idx        : first forecast row index
    horizon          : forecast length
    inactivity_gates : optional ``{target: {'lookback_hours', 'threshold_kw'}}``
                       — populated from the dataset's constants.py and stashed
                       on args by ``apply_dataset_thresholds``.
    apply_seasonal_gate : when False, skip the legacy calendar seasonal gate
                       entirely (set per-dataset via ``APPLY_SEASONAL_GATE`` in
                       constants.py). Cooling-dominated / year-round sites such
                       as HEEW disable it so real out-of-"season" heating and
                       cooling demand is not zeroed; the data-driven inactivity
                       gate below still applies.
    """
    result = {t: np.maximum(v, 0.0) for t, v in preds.items()}

    # Legacy seasonal gate (HouseZero-style).
    if apply_seasonal_gate and 'is_heating_season' in data.columns:
        end_idx = min(start_idx + horizon, len(data))
        n_valid = end_idx - start_idx
        is_heating = data['is_heating_season'].iloc[start_idx:end_idx].values.astype(bool)

        h = result[TARGET_THERMAL_HEATING].copy()
        c = result[TARGET_THERMAL_COOLING].copy()
        h[:n_valid] = np.where(is_heating, h[:n_valid], 0.0)
        c[:n_valid] = np.where(~is_heating, c[:n_valid], 0.0)
        result[TARGET_THERMAL_HEATING] = h
        result[TARGET_THERMAL_COOLING] = c

    # Data-driven inactivity gate.
    if inactivity_gates:
        for target, cfg in inactivity_gates.items():
            if target not in result or target not in data.columns:
                continue
            lookback  = int(cfg.get('lookback_hours', 168))
            threshold = float(cfg.get('threshold_kw', 0.01))
            ctx_start = max(0, start_idx - lookback)
            recent    = data[target].iloc[ctx_start:start_idx]
            if len(recent) > 0 and float(recent.mean()) < threshold:
                result[target] = np.zeros_like(result[target])

    return result


def create_sequences_for_finetuning(data, seq_len, horizon):
    return create_sequences(data, seq_len, horizon)


def pooled_r2_per_target(series: dict, targets) -> dict:
    """Compute one pooled R² per target from the concatenated y/p across all
    forecast windows, plus an ``overall_r2`` that is the mean of the per-target
    pooled values.

    ``series`` is the same dict produced by ``run_scenario`` —
    ``series[target] = {'actual': ndarray, 'pred': ndarray, ...}`` where both
    arrays are already concatenated across windows.

    Pooled R² is robust to the per-window degeneracy (off-season windows with
    near-constant actuals blowing up to −1e5) that makes a mean-of-per-window-R²
    meaningless on zero-heavy targets like heating/cooling/DHW.
    """
    out: dict = {}
    per_target_vals: list = []
    for t in targets:
        if t not in series:
            out[f'{t}_r2'] = float('nan')
            continue
        y = np.asarray(series[t]['actual'], dtype=float)
        p = np.asarray(series[t]['pred'],   dtype=float)
        mask = ~(np.isnan(y) | np.isnan(p))
        y, p = y[mask], p[mask]
        if y.size < 2 or np.var(y) <= 0:
            out[f'{t}_r2'] = float('nan')
            continue
        val = float(r2_score(y, p))
        out[f'{t}_r2'] = val
        per_target_vals.append(val)
    out['overall_r2'] = (float(np.mean(per_target_vals))
                        if per_target_vals else float('nan'))
    return out


def pooled_error_metrics_per_target(series: dict, targets) -> dict:
    """Pooled MAE / RMSE / R² per target, plus their ``overall_*`` aggregates.

    Mean-of-per-window aggregation gives a different RMSE than the pooled
    RMSE the R² formula implicitly uses: per-window RMSE is the inner sqrt
    over the horizon, then averaged across windows, whereas pooled RMSE is
    the sqrt over the entire (window × horizon) residual stream. The two
    only coincide when window-level RMSEs are identical. On KSRP DHW the
    gap was big enough to make the reported R² look inconsistent with the
    reported RMSE (RMSE 9.55 implies R² ≈ +0.21 against the pooled DHW
    variance of 114.8, but the pooled RMSE is actually 10.91 → R² ≈
    -0.03). This helper computes all three in the same pooled space so
    they are mutually consistent in the summary tables.

    MAE is identical between the two schemes because |·| is linear, so
    the change is invisible there; only RMSE moves. R² is unchanged from
    ``pooled_r2_per_target``.

    Returns ``{target}_mae``, ``{target}_rmse``, ``{target}_r2`` plus
    ``overall_mae``, ``overall_rmse``, ``overall_r2`` (means across the
    per-target values).
    """
    out: dict = {}
    maes, rmses, r2s = [], [], []
    for t in targets:
        if t not in series:
            out[f'{t}_mae']  = float('nan')
            out[f'{t}_rmse'] = float('nan')
            out[f'{t}_r2']   = float('nan')
            continue
        y = np.asarray(series[t]['actual'], dtype=float)
        p = np.asarray(series[t]['pred'],   dtype=float)
        mask = ~(np.isnan(y) | np.isnan(p))
        y, p = y[mask], p[mask]
        if y.size == 0:
            out[f'{t}_mae']  = float('nan')
            out[f'{t}_rmse'] = float('nan')
            out[f'{t}_r2']   = float('nan')
            continue
        mae  = float(np.mean(np.abs(y - p)))
        rmse = float(np.sqrt(np.mean((y - p) ** 2)))
        out[f'{t}_mae']  = mae
        out[f'{t}_rmse'] = rmse
        maes.append(mae); rmses.append(rmse)
        if y.size >= 2 and np.var(y) > 0:
            r2 = float(r2_score(y, p))
            out[f'{t}_r2'] = r2
            r2s.append(r2)
        else:
            out[f'{t}_r2'] = float('nan')
    out['overall_mae']  = float(np.mean(maes))  if maes  else float('nan')
    out['overall_rmse'] = float(np.mean(rmses)) if rmses else float('nan')
    out['overall_r2']   = float(np.mean(r2s))   if r2s   else float('nan')
    return out


def pooled_mase_per_target(history: list, targets) -> dict:
    """Pooled cross-window MASE per target, plus the across-target overall.

    ``history`` is the per-window metrics list produced by
    ``calculate_metrics_adaptive`` — each entry already carries the per-window
    ``<t>_mase_num_sum`` (sum of |y - p| over the horizon) and
    ``<t>_mase_den_sum`` (sum of |y - y_naive| over the horizon). The pooled
    per-target MASE is the ratio of the two sums across every window where
    they are defined:

        MASE_pooled(t) = (Σ_w  Σ_h |y_t,w,h − ŷ_t,w,h|)
                       / (Σ_w  Σ_h |y_t,w,h − y_naive,t,w,h|)

    This is the standard "Relative MAE" aggregation form — equivalent to
    weighting every horizon-step error equally regardless of which window it
    came from. It eliminates the mean-of-per-window-ratios bias that the
    legacy aggregation suffered from (the per-window naive MAE in the
    denominator can be tiny on quiescent shoulder-season windows, blowing up
    the ratio; arithmetic mean of those ratios is then pulled upward by a
    handful of outliers, even though every one of them represents an honest
    small absolute error). KSRP run `20260523-082538` per-target gap:
    heating 1.11 → ~0.86 pooled, cooling 1.17 → ~0.91 pooled, DHW 3.82 →
    ~2.46 pooled, electric 1.01 → ~0.90 pooled — uniformly tighter and
    closer to the actual model performance.

    Returns ``{target}_mase`` per target plus an ``overall_mase`` aggregate
    (mean of the per-target pooled values, matching the convention used by
    ``pooled_r2_per_target`` / ``pooled_error_metrics_per_target``).
    """
    out: dict = {}
    per_target_vals: list = []
    for t in targets:
        nums = [h.get(f'{t}_mase_num_sum') for h in history]
        dens = [h.get(f'{t}_mase_den_sum') for h in history]
        nums_v = [v for v in nums if v is not None and not np.isnan(v)]
        dens_v = [v for v in dens if v is not None and not np.isnan(v)]
        if not nums_v or not dens_v:
            out[f'{t}_mase'] = float('nan')
            continue
        total_den = float(np.sum(dens_v))
        if total_den < 1e-8:
            out[f'{t}_mase'] = float('nan')
            continue
        val = float(np.sum(nums_v)) / total_den
        out[f'{t}_mase'] = val
        per_target_vals.append(val)
    out['overall_mase'] = (float(np.mean(per_target_vals))
                          if per_target_vals else float('nan'))
    return out
