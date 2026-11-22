"""Ensemble-vs-single-model diagnostics for adaptive consumption forecasting.

These plotters and tables answer a single question: *is the LSTM+GBDT(+TFT)
blend actually better than running one of the constituents alone?* The
adaptive runner stores per-base-learner per-window predictions on the
``series`` dict under the key ``per_model_pred_windows``; everything here
reads from there so the diagnostics see the exact same predictions the
combined output was built from.

Exposed entry points (all keyed by target → series dict):
    * ``save_metrics_table``       — per-horizon MAE/RMSE/MAPE/R²/skill/Wilcoxon CSV + bar chart
    * ``plot_winrate``             — fraction of windows where ensemble beats each base learner
    * ``plot_error_correlation``   — heatmap of base-learner residual correlations (low ↔ ensemble has room to gain)
    * ``plot_oracle_gap``          — ensemble MAE vs hypothetical "best per-window single" MAE
    * ``plot_regime_breakdown``    — MAE per regime (season / business-hour / peak vs off-peak)

All plotters degrade gracefully when ``per_model_pred_windows`` is missing
or empty, so they're safe to call on the non-adaptive series too (the bars
just collapse to the singletons that exist).
"""

from __future__ import annotations

import os
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from scipy.stats import wilcoxon
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

from forecasting.constants import (
    TARGET_THERMAL_HEATING, TARGET_THERMAL_COOLING, TARGET_DHW, TARGET_ELECTRIC,
)

_TAG = {
    TARGET_THERMAL_HEATING: 'heating',
    TARGET_THERMAL_COOLING: 'cooling',
    TARGET_DHW:             'dhw',
    TARGET_ELECTRIC:        'electric',
}
_ALL_TARGETS = list(_TAG.keys())

_MODEL_COLORS = {
    'lstm':     '#1f77b4',
    'gbdt':     '#ff7f0e',
    'tft':      '#2ca02c',
    'ensemble': '#d62728',
}

_DEFAULT_MODELS = ('lstm', 'gbdt', 'tft', 'ensemble')


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _save(fig, output_dir: str, name: str, dpi: int = 120) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, name)
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {name}')
    return path


def _targets_in(series: dict) -> list:
    return [t for t in _ALL_TARGETS if t in series]


def _per_model(series: dict, target: str) -> dict:
    """Per-base-learner pred-windows dict; empty if not plumbed.

    Always includes 'ensemble' (the combined pred_windows already on the
    series), so callers can iterate the same shape regardless of source.
    """
    block = series[target]
    aw = np.asarray(block.get('actual_windows', np.empty((0, 0))), dtype=float)
    if aw.size == 0:
        return {}
    out: dict[str, np.ndarray] = {}
    raw = block.get('per_model_pred_windows') or {}
    for name in ('lstm', 'gbdt', 'tft'):
        arr = raw.get(name)
        if arr is None:
            continue
        arr = np.asarray(arr, dtype=float)
        if arr.shape == aw.shape:
            out[name] = arr
    ens = np.asarray(block.get('pred_windows', np.empty((0, 0))), dtype=float)
    if ens.shape == aw.shape:
        out['ensemble'] = ens
    return out


def _mae(pred: np.ndarray, act: np.ndarray, axis=None) -> np.ndarray:
    return np.nanmean(np.abs(pred - act), axis=axis)


def _rmse(pred: np.ndarray, act: np.ndarray, axis=None) -> np.ndarray:
    return np.sqrt(np.nanmean((pred - act) ** 2, axis=axis))


def _mape(pred: np.ndarray, act: np.ndarray, eps: float = 1.0) -> float:
    """MAPE with a 1 kW absolute floor — keeps off-season zero-heavy targets
    from blowing up the metric while still penalising magnitude error.
    Still explodes for heating/cooling whose actuals sit at 0 for whole
    months; report WAPE alongside for those targets."""
    denom = np.maximum(np.abs(act), eps)
    return float(np.nanmean(np.abs(pred - act) / denom) * 100.0)


def _wape(pred: np.ndarray, act: np.ndarray) -> float:
    """Weighted absolute percent error: sum(|err|) / sum(|act|).

    Bounded, scale-free, and well-defined for targets with many zero
    actuals (HVAC off-season, DHW idle hours) — exactly where MAPE blows
    up. Returns NaN only when the actuals are all zero."""
    a = np.abs(act).ravel()
    e = np.abs(pred - act).ravel()
    m = ~(np.isnan(a) | np.isnan(e))
    a, e = a[m], e[m]
    s = float(np.sum(a))
    if s <= 0:
        return float('nan')
    return float(np.sum(e) / s * 100.0)


def _r2(pred: np.ndarray, act: np.ndarray) -> float:
    a = act.ravel()
    p = pred.ravel()
    m = ~(np.isnan(a) | np.isnan(p))
    a, p = a[m], p[m]
    if a.size < 2 or np.var(a) <= 0:
        return float('nan')
    ss_res = float(np.sum((a - p) ** 2))
    ss_tot = float(np.sum((a - np.mean(a)) ** 2))
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float('nan')


def _r2_active(pred: np.ndarray, act: np.ndarray,
                threshold_frac: float = 0.05) -> tuple[float, float]:
    """R² restricted to hours where the actual is above a fraction of the
    target's peak. Useful for zero-heavy targets (DHW idle hours, HVAC
    shoulder months) where the pooled R² is dominated by the trivial
    zero-zero match points and the mean-predictor baseline is barely
    distinguishable from a competent model.

    Returns ``(r2_active, active_fraction)`` so callers can see how much
    data the active-only R² is evaluated on (a 1% active fraction with
    R² = 0.4 is far less impressive than 50% active with R² = 0.4).
    """
    a = act.ravel()
    p = pred.ravel()
    m = ~(np.isnan(a) | np.isnan(p))
    a, p = a[m], p[m]
    if a.size == 0:
        return float('nan'), 0.0
    peak = float(np.nanmax(a))
    if not np.isfinite(peak) or peak <= 0:
        return float('nan'), 0.0
    active = a >= threshold_frac * peak
    n_active = int(active.sum())
    if n_active < 2:
        return float('nan'), float(n_active / a.size)
    a_act = a[active]
    p_act = p[active]
    if np.var(a_act) <= 0:
        return float('nan'), float(n_active / a.size)
    ss_res = float(np.sum((a_act - p_act) ** 2))
    ss_tot = float(np.sum((a_act - np.mean(a_act)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float('nan')
    return r2, float(n_active / a.size)


def _window_mae(pred: np.ndarray, act: np.ndarray) -> np.ndarray:
    """Per-window MAE: shape (n_windows,). NaN-safe across the horizon axis."""
    return np.nanmean(np.abs(pred - act), axis=1)


# ---------------------------------------------------------------------------
# 1. Per-horizon metrics table + bar chart
# ---------------------------------------------------------------------------

def save_metrics_table(series: dict, output_dir: str,
                       label: str = 'ensemble_compare') -> str | None:
    """Per-(target, model) MAE/RMSE/MAPE/R²/skill/Wilcoxon as a CSV.

    Skill score compares each model to the best non-ensemble base learner;
    Wilcoxon p-value tests the ensemble's per-window MAE against that same
    best base learner (alternative='less', i.e. ensemble is significantly
    lower). The companion grouped bar chart visualises MAE per model side
    by side, with one panel per target.
    """
    targets = _targets_in(series)
    if not targets:
        return None
    rows: list[dict] = []
    bar_payload: dict[str, dict[str, float]] = {}
    bar_payload_rmse: dict[str, dict[str, float]] = {}
    for t in targets:
        block = series[t]
        aw = np.asarray(block.get('actual_windows', np.empty((0, 0))), dtype=float)
        if aw.size == 0:
            continue
        models = _per_model(series, t)
        if not models:
            continue
        # Identify best non-ensemble base learner for skill / Wilcoxon
        base_names = [m for m in models if m != 'ensemble']
        base_mae = {m: float(_mae(models[m], aw)) for m in base_names}
        best_base = (min(base_mae, key=base_mae.get) if base_mae else None)
        best_base_window_mae = (_window_mae(models[best_base], aw)
                                if best_base else None)
        ens_window_mae = (_window_mae(models['ensemble'], aw)
                          if 'ensemble' in models else None)

        bar_payload[_TAG[t]] = {}
        bar_payload_rmse[_TAG[t]] = {}
        for m, pred in models.items():
            mae  = float(_mae(pred, aw))
            rmse = float(_rmse(pred, aw))
            mape = _mape(pred, aw)
            wape = _wape(pred, aw)
            r2   = _r2(pred, aw)
            r2_act, active_frac = _r2_active(pred, aw)
            bar_payload[_TAG[t]][m]      = mae
            bar_payload_rmse[_TAG[t]][m] = rmse
            row: dict = {
                'target': _TAG[t],
                'model':  m,
                'n_windows':         int(aw.shape[0]),
                'horizon_steps':     int(aw.shape[1]),
                'mae':               round(mae,  5),
                'rmse':              round(rmse, 5),
                'mape_pct':          round(mape, 4),
                'wape_pct':          round(wape, 4) if not np.isnan(wape) else None,
                'r2_pooled':         round(r2,   5) if not np.isnan(r2) else None,
                # r2_active = R² over hours where actual >= 5% of target peak.
                # For DHW the pooled R² hovers near zero because the model
                # is roughly as good as the long-run mean across all hours;
                # r2_active strips out the zero-zero "trivially correct"
                # matches and surfaces the model's skill on the active hours.
                'r2_active':         round(r2_act, 5) if not np.isnan(r2_act) else None,
                'active_fraction':   round(active_frac, 4),
            }
            # Skill vs best non-ensemble base; ensemble row keeps a skill > 0
            # when it does outperform the best constituent. Base rows show a
            # negative skill when they're worse than the leader (and 0.0 for
            # the leader itself).
            if best_base is not None and best_base in models:
                ref = base_mae[best_base]
                skill = (1.0 - mae / ref) if ref > 0 else float('nan')
                row['best_base_model'] = best_base
                row['skill_vs_best_base'] = (round(skill, 5)
                                              if not np.isnan(skill) else None)
            # Wilcoxon: ensemble vs best base, per-window MAE, alt='less'.
            if (_HAS_SCIPY and m == 'ensemble' and best_base
                    and ens_window_mae is not None
                    and best_base_window_mae is not None):
                diff = ens_window_mae - best_base_window_mae
                diff = diff[~np.isnan(diff)]
                if diff.size >= 10 and np.any(diff != 0):
                    try:
                        stat = wilcoxon(diff, alternative='less',
                                        zero_method='zsplit')
                        row['wilcoxon_p_ens_lt_best_base'] = float(stat.pvalue)
                    except Exception:
                        pass
            rows.append(row)

    if not rows:
        return None

    csv_path = os.path.join(output_dir, f'metrics_table_{label}.csv')
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f'  Saved metrics_table_{label}.csv')

    # Companion bar chart — MAE + RMSE side by side per target
    targets_in = list(bar_payload.keys())
    if targets_in:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for ax, payload, ylabel in (
            (axes[0], bar_payload,      'MAE (kW)'),
            (axes[1], bar_payload_rmse, 'RMSE (kW)'),
        ):
            model_names: list[str] = []
            for tag in targets_in:
                for m in payload[tag]:
                    if m not in model_names:
                        model_names.append(m)
            x = np.arange(len(targets_in))
            w = 0.8 / max(len(model_names), 1)
            for i, m in enumerate(model_names):
                vals = [payload[tag].get(m, np.nan) for tag in targets_in]
                ax.bar(x + (i - (len(model_names) - 1) / 2) * w, vals, w,
                       label=m.upper(),
                       color=_MODEL_COLORS.get(m, None),
                       alpha=0.9 if m == 'ensemble' else 0.75,
                       edgecolor='black' if m == 'ensemble' else None,
                       linewidth=0.8 if m == 'ensemble' else 0.0)
            ax.set_xticks(x)
            ax.set_xticklabels(targets_in)
            ax.set_ylabel(ylabel)
            ax.set_title(ylabel.split(' ')[0] + ' per target')
            ax.grid(True, axis='y', alpha=0.3)
            ax.legend(fontsize=8)
        plt.suptitle(f'Per-model errors — {label}', fontsize=13)
        plt.tight_layout()
        _save(fig, output_dir, f'metrics_table_{label}.png')

    return csv_path


# ---------------------------------------------------------------------------
# 2. Win-rate per horizon (ensemble vs each base learner)
# ---------------------------------------------------------------------------

def plot_winrate(series: dict, output_dir: str,
                 label: str = 'ensemble_compare') -> str | None:
    """Per-horizon fraction of windows where |err_ensemble| ≤ |err_base|.

    A line at 0.5 marks the "no advantage" boundary; bars above it mean the
    ensemble wins more often than the base learner at that horizon step.
    Reported per target, one panel each.
    """
    targets = _targets_in(series)
    if not targets:
        return None
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True, sharey=True)
    axes = axes.flatten()
    plotted = False
    for ax, t in zip(axes, targets):
        models = _per_model(series, t)
        if 'ensemble' not in models:
            ax.set_title(f'{_TAG[t]} — no ensemble pred-windows')
            continue
        aw = np.asarray(series[t]['actual_windows'], dtype=float)
        ens_err = np.abs(models['ensemble'] - aw)
        h = ens_err.shape[1]
        x = np.arange(1, h + 1)
        for m in ('lstm', 'gbdt', 'tft'):
            if m not in models:
                continue
            base_err = np.abs(models[m] - aw)
            with np.errstate(invalid='ignore'):
                wins = (ens_err <= base_err).astype(float)
            wins[np.isnan(ens_err) | np.isnan(base_err)] = np.nan
            rate = np.nanmean(wins, axis=0)
            ax.plot(x, rate, color=_MODEL_COLORS[m], linewidth=1.4,
                    label=f'vs {m.upper()}')
            plotted = True
        ax.axhline(0.5, color='grey', linewidth=0.8, linestyle='--')
        ax.set_ylim(0.0, 1.0)
        ax.set_title(f'{_TAG[t]} — ensemble win-rate vs base learners')
        ax.set_xlabel('Horizon step (h)')
        ax.set_ylabel('P(ensemble ≤ base)')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='lower right')
    if not plotted:
        plt.close(fig)
        return None
    plt.suptitle('Ensemble win-rate by horizon step', fontsize=13)
    plt.tight_layout()
    return _save(fig, output_dir, f'ensemble_winrate_{label}.png')


# ---------------------------------------------------------------------------
# 3. Error-correlation heatmap
# ---------------------------------------------------------------------------

def plot_error_correlation(series: dict, output_dir: str,
                           label: str = 'ensemble_compare') -> str | None:
    """Pearson correlation of base-learner residuals per target.

    Low off-diagonal correlation means the constituents disagree in
    different ways — that's exactly the diversity an ensemble exploits.
    High correlation means the base learners are largely making the same
    mistakes, so the ensemble has limited room to improve.
    """
    targets = _targets_in(series)
    if not targets:
        return None
    target_payload: list[tuple[str, np.ndarray, list[str]]] = []
    for t in targets:
        models = _per_model(series, t)
        base_names = [m for m in models if m != 'ensemble']
        if len(base_names) < 2:
            continue
        aw = np.asarray(series[t]['actual_windows'], dtype=float)
        residuals = []
        names = []
        for m in base_names:
            r = (models[m] - aw).ravel()
            residuals.append(r)
            names.append(m.upper())
        mat = np.vstack(residuals)
        finite = np.all(np.isfinite(mat), axis=0)
        mat = mat[:, finite]
        if mat.shape[1] < 5:
            continue
        corr = np.corrcoef(mat)
        target_payload.append((_TAG[t], corr, names))
    if not target_payload:
        return None
    n = len(target_payload)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.0), squeeze=False)
    for ax, (tag, corr, names) in zip(axes[0], target_payload):
        im = ax.imshow(corr, vmin=-1.0, vmax=1.0, cmap='RdBu_r')
        ax.set_xticks(range(len(names))); ax.set_xticklabels(names, fontsize=8)
        ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=8)
        for i in range(len(names)):
            for j in range(len(names)):
                ax.text(j, i, f'{corr[i, j]:+.2f}',
                        ha='center', va='center', fontsize=8,
                        color='white' if abs(corr[i, j]) > 0.5 else 'black')
        ax.set_title(tag, fontsize=10)
    fig.colorbar(im, ax=axes[0].tolist(), fraction=0.04, pad=0.04,
                 label='Pearson r')
    plt.suptitle('Base-learner residual correlation  (low ↔ diverse)',
                 fontsize=13)
    return _save(fig, output_dir, f'error_correlation_{label}.png')


# ---------------------------------------------------------------------------
# 4. Oracle gap (ensemble vs per-window-best single model)
# ---------------------------------------------------------------------------

def plot_oracle_gap(series: dict, output_dir: str,
                    label: str = 'ensemble_compare') -> str | None:
    """Ensemble per-window MAE vs the oracle "pick the best base per window".

    The oracle is an unrealisable upper bound: for each window it selects
    whichever base learner happened to have the lowest MAE. Plotting the
    ensemble against it shows how much the adaptive blend recovers of the
    diversity available, vs the worst base learner (cheap baseline floor).
    """
    targets = _targets_in(series)
    if not targets:
        return None
    rows: list[dict] = []
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=False)
    axes = axes.flatten()
    plotted = False
    for ax, t in zip(axes, targets):
        models = _per_model(series, t)
        base_names = [m for m in models if m != 'ensemble']
        if 'ensemble' not in models or len(base_names) < 2:
            ax.set_title(f'{_TAG[t]} — need ensemble + ≥2 base learners')
            continue
        aw = np.asarray(series[t]['actual_windows'], dtype=float)

        # Per-window per-model MAE → oracle = min over models per window.
        base_window_mae = np.stack(
            [_window_mae(models[m], aw) for m in base_names], axis=1
        )                                              # (n_windows, n_base)
        oracle_mae = np.nanmin(base_window_mae, axis=1)
        worst_mae  = np.nanmax(base_window_mae, axis=1)
        ens_mae    = _window_mae(models['ensemble'], aw)

        # Sort windows by oracle for a clean monotonic-ish baseline curve.
        order = np.argsort(oracle_mae)
        idx = np.arange(1, len(order) + 1)
        ax.plot(idx, np.sort(oracle_mae), color='black', linewidth=1.2,
                label='Oracle (best base / window)')
        ax.plot(idx, ens_mae[order], color=_MODEL_COLORS['ensemble'],
                linewidth=1.2, label='Ensemble')
        ax.plot(idx, worst_mae[order], color='grey', linewidth=1.0,
                linestyle='--', alpha=0.7, label='Worst base / window')
        ax.set_title(f'{_TAG[t]} — oracle gap')
        ax.set_xlabel('Window rank (sorted by oracle MAE)')
        ax.set_ylabel('Window MAE (kW)')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='upper left')
        plotted = True

        mean_oracle = float(np.nanmean(oracle_mae))
        mean_ens    = float(np.nanmean(ens_mae))
        mean_worst  = float(np.nanmean(worst_mae))
        # Closure = (worst - ens) / (worst - oracle); 1.0 means the ensemble
        # matches the oracle, 0.0 means it matches the worst base learner.
        denom = (mean_worst - mean_oracle)
        closure = (mean_worst - mean_ens) / denom if denom > 0 else float('nan')
        rows.append({
            'target':                  _TAG[t],
            'mean_window_mae_ensemble': round(mean_ens,    5),
            'mean_window_mae_oracle':   round(mean_oracle, 5),
            'mean_window_mae_worst':    round(mean_worst,  5),
            'oracle_gap':               round(mean_ens - mean_oracle, 5),
            'closure_pct':              (round(closure * 100.0, 2)
                                          if not np.isnan(closure) else None),
        })
        ax.text(0.02, 0.98,
                f'mean ensemble: {mean_ens:.3f}\n'
                f'mean oracle:   {mean_oracle:.3f}\n'
                f'gap closed:    {closure * 100:.1f}%' if not np.isnan(closure)
                else f'mean ensemble: {mean_ens:.3f}\n'
                     f'mean oracle:   {mean_oracle:.3f}',
                transform=ax.transAxes, ha='left', va='top', fontsize=8,
                bbox=dict(facecolor='white', alpha=0.85,
                          edgecolor='lightgrey', boxstyle='round,pad=0.3'))
    if not plotted:
        plt.close(fig)
        return None
    plt.suptitle('Ensemble vs oracle "best base learner per window"',
                 fontsize=13)
    plt.tight_layout()
    path = _save(fig, output_dir, f'oracle_gap_{label}.png')

    # Companion CSV
    if rows:
        csv_path = os.path.join(output_dir, f'oracle_gap_{label}.csv')
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        print(f'  Saved oracle_gap_{label}.csv')

    return path


# ---------------------------------------------------------------------------
# 5. Per-regime breakdown
# ---------------------------------------------------------------------------

_BUSINESS_HOURS = range(8, 19)   # 08:00..18:00 inclusive
_PEAK_QUANTILE  = 0.85           # top 15 % of actuals counts as a "peak hour"


def _flatten_with_timestamps(series: dict, target: str) -> tuple:
    """Flatten per-window matrices and align with the per-window-start index.

    Returns ``(timestamps_per_step, per_model_pred_flat, actual_flat)``.
    Each ``per_model_pred_flat`` value has the same length as the actual
    vector; timestamps are reconstructed from ``window_starts`` extended by
    the horizon step. Returns ``(None, None, None)`` when data is missing.
    """
    block = series[target]
    aw = np.asarray(block.get('actual_windows', np.empty((0, 0))), dtype=float)
    if aw.size == 0:
        return None, None, None
    starts = block.get('window_starts')
    if not isinstance(starts, pd.DatetimeIndex) or len(starts) != aw.shape[0]:
        return None, None, None
    horizon = aw.shape[1]
    # Reconstruct (n_windows * horizon) timestamps row-major: start_w + k hours
    # for k = 0..horizon-1, ordered to match aw.ravel() (i.e. window-major).
    starts_np = starts.to_numpy()
    ts_flat = pd.DatetimeIndex(
        np.array([s + np.timedelta64(k, 'h')
                  for s in starts_np for k in range(horizon)])
    )
    actual_flat = aw.ravel()
    models = _per_model(series, target)
    per_model_flat = {m: pred.ravel() for m, pred in models.items()}
    return ts_flat, per_model_flat, actual_flat


def plot_regime_breakdown(series: dict, output_dir: str,
                          label: str = 'ensemble_compare') -> str | None:
    """Per-model MAE within regime bins:
       (a) season (heating vs cooling, derived from month),
       (b) business hours (08–18) vs off-hours,
       (c) peak (≥ 85th percentile of actuals) vs off-peak.

    Three panels per target; bars within each panel are grouped by model so
    you can spot regimes where the ensemble outperforms most.
    """
    targets = _targets_in(series)
    if not targets:
        return None
    fig, axes = plt.subplots(len(targets), 3,
                              figsize=(14, 3.5 * len(targets)),
                              squeeze=False)
    plotted = False
    for r, t in enumerate(targets):
        ts_flat, per_model_flat, actual_flat = _flatten_with_timestamps(series, t)
        if ts_flat is None or not per_model_flat:
            for c in range(3):
                axes[r, c].set_title(f'{_TAG[t]} — no per-model data')
            continue

        # Regime masks
        month   = pd.DatetimeIndex(ts_flat).month
        heating_mask = np.isin(month, [10, 11, 12, 1, 2, 3])
        cooling_mask = ~heating_mask
        hour    = pd.DatetimeIndex(ts_flat).hour
        biz_mask     = np.isin(hour, list(_BUSINESS_HOURS))
        offbiz_mask  = ~biz_mask
        # Per-target peak threshold from the actuals themselves
        finite_act = actual_flat[np.isfinite(actual_flat)]
        if finite_act.size == 0:
            for c in range(3):
                axes[r, c].set_title(f'{_TAG[t]} — no actuals')
            continue
        peak_thr = float(np.nanquantile(finite_act, _PEAK_QUANTILE))
        peak_mask    = actual_flat >= peak_thr
        offpeak_mask = ~peak_mask

        regimes = [
            ('Season',   [('heating', heating_mask), ('cooling', cooling_mask)]),
            ('Hours',    [('business', biz_mask),    ('off-hours', offbiz_mask)]),
            ('Demand',   [(f'peak (≥{peak_thr:.2f})', peak_mask),
                          ('off-peak', offpeak_mask)]),
        ]
        for c, (title, splits) in enumerate(regimes):
            ax = axes[r, c]
            split_names = [name for name, _ in splits]
            model_names = list(per_model_flat.keys())
            x = np.arange(len(split_names))
            w = 0.8 / max(len(model_names), 1)
            for i, m in enumerate(model_names):
                vals = []
                for _name, mask in splits:
                    sel = mask & np.isfinite(actual_flat) & np.isfinite(per_model_flat[m])
                    if sel.sum() == 0:
                        vals.append(np.nan)
                    else:
                        vals.append(float(np.nanmean(
                            np.abs(per_model_flat[m][sel] - actual_flat[sel]))))
                ax.bar(x + (i - (len(model_names) - 1) / 2) * w, vals, w,
                       label=m.upper(),
                       color=_MODEL_COLORS.get(m, None),
                       alpha=0.9 if m == 'ensemble' else 0.75,
                       edgecolor='black' if m == 'ensemble' else None,
                       linewidth=0.8 if m == 'ensemble' else 0.0)
            ax.set_xticks(x)
            ax.set_xticklabels(split_names, fontsize=8)
            ax.set_ylabel('MAE (kW)' if c == 0 else '')
            ax.set_title(f'{_TAG[t]} — {title}')
            ax.grid(True, axis='y', alpha=0.3)
            if r == 0 and c == 2:
                ax.legend(fontsize=8, loc='upper right')
        plotted = True
    if not plotted:
        plt.close(fig)
        return None
    plt.suptitle('Per-regime MAE — ensemble vs base learners', fontsize=13)
    plt.tight_layout()
    return _save(fig, output_dir, f'regime_breakdown_{label}.png')


# ---------------------------------------------------------------------------
# 6. Public façade — call all five at once
# ---------------------------------------------------------------------------

def run_all(series: dict, output_dir: str,
            label: str = 'ensemble_compare') -> dict:
    """Convenience entry point — runs all five diagnostics for one scenario.

    Returns a dict mapping diagnostic name → output path (or None on skip).
    """
    return {
        'metrics_table':     save_metrics_table(series, output_dir, label),
        'winrate':           plot_winrate(series, output_dir, label),
        'error_correlation': plot_error_correlation(series, output_dir, label),
        'oracle_gap':        plot_oracle_gap(series, output_dir, label),
        'regime_breakdown':  plot_regime_breakdown(series, output_dir, label),
    }


__all__ = [
    'save_metrics_table',
    'plot_winrate',
    'plot_error_correlation',
    'plot_oracle_gap',
    'plot_regime_breakdown',
    'run_all',
]
