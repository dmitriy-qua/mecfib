"""Visualization utilities for adaptive consumption forecasting.

This module ships two layers of plotters:

* **Legacy single-window plots** (``plot_predictions``, ``plot_performance_history``,
  ``plot_retrain_events``) — kept for backward compatibility with older notebooks.
* **Comparison and diagnostic plots** consumed by ``run.py`` — these operate on
  the canonical ``series`` dict produced by ``run_scenario`` and on the
  per-horizon weights dict loaded from ``weights_per_horizon.json``.

Series-dict contract (per target):
    series[target] = {
        'actual':         np.ndarray (n_windows * horizon,) flat concatenation,
        'pred':           np.ndarray (n_windows * horizon,) flat concatenation,
        'timestamps':     pd.DatetimeIndex (n_windows * horizon,),
        'pred_windows':   np.ndarray (n_windows, horizon),
        'actual_windows': np.ndarray (n_windows, horizon),
        'window_starts':  pd.DatetimeIndex (n_windows,),
    }

All new plotters accept ``series_fixed`` and an optional ``series_adaptive``;
they degrade gracefully when adaptive is None (non-adaptive-only runs).
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

from sklearn.metrics import r2_score

_TAG = {
    'thermal_heating_kw': 'heating',
    'thermal_cooling_kw': 'cooling',
    'dhw_kw':             'dhw',
    'electric_demand_kw': 'electric',
}
_ALL_TARGETS = list(_TAG.keys())

_COLOR_FIXED    = 'steelblue'
_COLOR_ADAPTIVE = 'tomato'
_COLOR_ACTUAL   = '#444444'
_COLOR_LSTM     = '#1f77b4'
_COLOR_GBDT     = '#ff7f0e'
_COLOR_TFT      = '#2ca02c'


# ---------------------------------------------------------------------------
# Legacy single-window plots (preserved for backward compatibility)
# ---------------------------------------------------------------------------

def plot_predictions(actual_thermal, actual_electric,
                     pred_thermal, pred_electric,
                     step, save_dir=None, show=False):
    """Plot actual vs predicted for one forecast window."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    t = np.arange(len(actual_thermal))
    axes[0].plot(t, actual_thermal, label='Actual',    color=_COLOR_FIXED)
    axes[0].plot(t, pred_thermal,   label='Predicted', color=_COLOR_ADAPTIVE, linestyle='--')
    axes[0].set_ylabel('Thermal Demand (kW)')
    axes[0].set_title(f'Step {step} - Thermal Demand')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(t, actual_electric, label='Actual',    color=_COLOR_FIXED)
    axes[1].plot(t, pred_electric,   label='Predicted', color=_COLOR_ADAPTIVE, linestyle='--')
    axes[1].set_ylabel('Electric Demand (kW)')
    axes[1].set_xlabel('Forecast Hour')
    axes[1].set_title(f'Step {step} - Electric Demand')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f'forecast_step_{step:05d}.png'), dpi=100)
    if show:
        plt.show()
    plt.close(fig)


def plot_performance_history(history, save_dir=None, show=False):
    """Plot MAE/RMSE/MASE over evaluation steps."""
    if not history:
        return
    keys = [
        ('overall_mae',            'Overall MAE'),
        ('overall_rmse',           'Overall RMSE'),
        ('overall_mase',           'Overall MASE'),
        ('thermal_heating_kw_mae', 'Heating MAE'),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.flatten()
    for ax, (key, title) in zip(axes, keys):
        vals = [h[key] for h in history if key in h]
        ax.plot(vals, color=_COLOR_FIXED, linewidth=0.8)
        ax.set_title(title)
        ax.set_xlabel('Evaluation Step')
        ax.grid(True, alpha=0.3)
    plt.suptitle('Adaptive Forecaster Performance History', fontsize=13)
    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'performance_history.png'), dpi=100)
    if show:
        plt.show()
    plt.close(fig)


def plot_retrain_events_legacy(history, retrain_steps, save_dir=None, show=False):
    """Overlay retrain events on the MAE time-series (legacy single-scenario)."""
    if not history:
        return
    vals = [h.get('overall_mae', float('nan')) for h in history]
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(vals, color=_COLOR_FIXED, linewidth=0.8, label='Overall MAE')
    for s in retrain_steps:
        ax.axvline(x=s, color=_COLOR_ADAPTIVE, linestyle='--', linewidth=0.8, alpha=0.7)
    ax.set_title('Overall MAE with Retrain Events')
    ax.set_xlabel('Evaluation Step')
    ax.set_ylabel('MAE (kW)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'retrain_events.png'), dpi=100)
    if show:
        plt.show()
    plt.close(fig)


plot_retrain_events = plot_retrain_events_legacy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _targets_in(series):
    return [t for t in _ALL_TARGETS if t in series]


def _windows(series, target):
    pw = series[target].get('pred_windows')
    aw = series[target].get('actual_windows')
    if pw is None or aw is None:
        return None, None
    pw = np.asarray(pw, dtype=float)
    aw = np.asarray(aw, dtype=float)
    if pw.size == 0 or aw.size == 0 or pw.shape != aw.shape:
        return None, None
    return pw, aw


def _flat(series, target):
    p = np.asarray(series[target].get('pred', []), dtype=float).ravel()
    a = np.asarray(series[target].get('actual', []), dtype=float).ravel()
    n = min(len(p), len(a))
    p, a = p[:n], a[:n]
    mask = ~(np.isnan(p) | np.isnan(a))
    return p[mask], a[mask]


def _save(fig, output_dir, name, dpi=120):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, name)
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 1. Per-horizon MAE / RMSE  +  skill score
# ---------------------------------------------------------------------------

def plot_horizon_error(series_fixed, output_dir, series_adaptive=None):
    """MAE and RMSE per horizon step, one panel per target."""
    targets = _targets_in(series_fixed)
    if not targets:
        return None
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
    axes = axes.flatten()
    for ax, t in zip(axes, targets):
        pf, af = _windows(series_fixed, t)
        if pf is None:
            ax.set_title(f'{_TAG[t]} - no per-window data')
            continue
        h = pf.shape[1]
        x = np.arange(1, h + 1)
        mae_f  = np.nanmean(np.abs(pf - af), axis=0)
        rmse_f = np.sqrt(np.nanmean((pf - af) ** 2, axis=0))
        ax.plot(x, mae_f,  color=_COLOR_FIXED, linewidth=1.4, label='Non-adaptive MAE')
        ax.plot(x, rmse_f, color=_COLOR_FIXED, linewidth=1.0, linestyle=':',
                alpha=0.7, label='Non-adaptive RMSE')
        if series_adaptive is not None and t in series_adaptive:
            pa, aa = _windows(series_adaptive, t)
            if pa is not None and pa.shape == pf.shape:
                mae_a  = np.nanmean(np.abs(pa - aa), axis=0)
                rmse_a = np.sqrt(np.nanmean((pa - aa) ** 2, axis=0))
                ax.plot(x, mae_a,  color=_COLOR_ADAPTIVE, linewidth=1.4, label='Adaptive MAE')
                ax.plot(x, rmse_a, color=_COLOR_ADAPTIVE, linewidth=1.0,
                        linestyle=':', alpha=0.7, label='Adaptive RMSE')
        ax.set_title(f'{_TAG[t]} - error vs horizon step')
        ax.set_ylabel('Error (kW)')
        ax.set_xlabel('Horizon step (h)')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='upper left')
    plt.suptitle('Per-horizon MAE and RMSE', fontsize=13)
    plt.tight_layout()
    return _save(fig, output_dir, 'horizon_error.png')


def plot_horizon_skill_score(series_fixed, series_adaptive, output_dir):
    """Skill score per horizon step: 1 - MAE_adaptive / MAE_non_adaptive."""
    if not series_adaptive:
        return None
    targets = _targets_in(series_fixed)
    if not targets:
        return None
    fig, ax = plt.subplots(figsize=(12, 5))
    skill_stack = []
    for t in targets:
        pf, af = _windows(series_fixed, t)
        pa, aa = _windows(series_adaptive, t) if t in series_adaptive else (None, None)
        if pf is None or pa is None or pf.shape != pa.shape:
            continue
        mae_f = np.nanmean(np.abs(pf - af), axis=0)
        mae_a = np.nanmean(np.abs(pa - aa), axis=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            skill = 1.0 - np.where(mae_f > 0, mae_a / mae_f, np.nan)
        x = np.arange(1, len(skill) + 1)
        ax.plot(x, skill, label=_TAG[t], linewidth=1.2)
        skill_stack.append(skill)
    if skill_stack:
        mean_skill = np.nanmean(np.vstack(skill_stack), axis=0)
        ax.plot(np.arange(1, len(mean_skill) + 1), mean_skill,
                color='black', linewidth=2.0, label='mean')
    ax.axhline(0.0, color='grey', linewidth=0.8, linestyle='--')
    ax.set_title('Adaptive skill score per horizon step (1 - MAE_adaptive / MAE_non-adaptive)')
    ax.set_xlabel('Horizon step (h)')
    ax.set_ylabel('Skill score')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, ncol=3)
    plt.tight_layout()
    return _save(fig, output_dir, 'horizon_skill_score.png')


# ---------------------------------------------------------------------------
# 2. Pooled scatter + load duration
# ---------------------------------------------------------------------------

def plot_pooled_scatter(series_fixed, output_dir, series_adaptive=None):
    """Hexbin of pooled predicted vs actual per target, with pooled R^2 annotated."""
    targets = _targets_in(series_fixed)
    if not targets:
        return None
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    for ax, t in zip(axes, targets):
        pf, af = _flat(series_fixed, t)
        if pf.size < 2 or np.var(af) <= 0:
            ax.set_title(f'{_TAG[t]} - insufficient data')
            continue
        lo = float(min(pf.min(), af.min()))
        hi = float(max(pf.max(), af.max(), 1e-6))
        ax.hexbin(af, pf, gridsize=40, cmap='Blues', mincnt=1,
                  extent=(lo, hi, lo, hi))
        r2_f = float(r2_score(af, pf))
        annot = f'Non-adaptive  R2={r2_f:.3f}  n={pf.size}'
        if series_adaptive is not None and t in series_adaptive:
            pa, aa = _flat(series_adaptive, t)
            if pa.size >= 2 and np.var(aa) > 0:
                ax.hexbin(aa, pa, gridsize=40, cmap='Reds', mincnt=1,
                          alpha=0.55, extent=(lo, hi, lo, hi))
                r2_a = float(r2_score(aa, pa))
                annot += f'\nAdaptive       R2={r2_a:.3f}  n={pa.size}'
        ax.plot([lo, hi], [lo, hi], color='black', linewidth=0.8, linestyle='--')
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('Actual (kW)')
        ax.set_ylabel('Predicted (kW)')
        ax.set_title(f'{_TAG[t]} - predicted vs actual (pooled)')
        ax.grid(True, alpha=0.3)
        ax.text(0.02, 0.98, annot, transform=ax.transAxes,
                ha='left', va='top', fontsize=9,
                bbox=dict(facecolor='white', alpha=0.85, edgecolor='lightgrey',
                          boxstyle='round,pad=0.3'))
    plt.suptitle('Pooled predicted-vs-actual scatter', fontsize=13)
    plt.tight_layout()
    return _save(fig, output_dir, 'pooled_scatter.png')


def plot_load_duration(series_fixed, output_dir, series_adaptive=None):
    """Load-duration curves: descending-sorted actual vs predicted per target."""
    targets = _targets_in(series_fixed)
    if not targets:
        return None
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    axes = axes.flatten()
    for ax, t in zip(axes, targets):
        pf, af = _flat(series_fixed, t)
        if pf.size == 0:
            ax.set_title(f'{_TAG[t]} - no data')
            continue
        x = np.arange(1, pf.size + 1) / pf.size * 100.0
        ax.plot(x, np.sort(af)[::-1], color=_COLOR_ACTUAL, linewidth=1.2, label='Actual')
        ax.plot(x, np.sort(pf)[::-1], color=_COLOR_FIXED,
                linewidth=1.1, linestyle='--', label='Non-adaptive')
        if series_adaptive is not None and t in series_adaptive:
            pa, _aa = _flat(series_adaptive, t)
            if pa.size:
                ax.plot(np.arange(1, pa.size + 1) / pa.size * 100.0,
                        np.sort(pa)[::-1], color=_COLOR_ADAPTIVE,
                        linewidth=1.1, linestyle='--', label='Adaptive')
        ax.set_title(f'{_TAG[t]} - load duration')
        ax.set_xlabel('Exceedance (%)')
        ax.set_ylabel('Demand (kW)')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    plt.suptitle('Load-duration curves - sorted actual vs forecast', fontsize=13)
    plt.tight_layout()
    return _save(fig, output_dir, 'load_duration.png')


# ---------------------------------------------------------------------------
# 3. Residual diagnostics + paired-difference
# ---------------------------------------------------------------------------

def plot_residual_violins(series_fixed, output_dir, series_adaptive=None):
    """Violin/box of residuals (pred - actual) per target, adaptive vs non-adaptive."""
    targets = _targets_in(series_fixed)
    if not targets:
        return None
    fig, ax = plt.subplots(figsize=(12, 6))
    data, labels, colors = [], [], []
    for t in targets:
        pf, af = _flat(series_fixed, t)
        if pf.size:
            data.append(pf - af)
            labels.append(f'{_TAG[t]}\nnon-adaptive')
            colors.append(_COLOR_FIXED)
        if series_adaptive is not None and t in series_adaptive:
            pa, aa = _flat(series_adaptive, t)
            if pa.size:
                data.append(pa - aa)
                labels.append(f'{_TAG[t]}\nadaptive')
                colors.append(_COLOR_ADAPTIVE)
    if not data:
        return None
    parts = ax.violinplot(data, showmeans=False, showmedians=True, widths=0.85)
    for body, color in zip(parts['bodies'], colors):
        body.set_facecolor(color)
        body.set_alpha(0.55)
        body.set_edgecolor('black')
    for key in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
        if key in parts:
            parts[key].set_color('black')
            parts[key].set_linewidth(0.8)
    ax.axhline(0.0, color='grey', linewidth=0.8, linestyle='--')
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('Residual (pred - actual, kW)')
    ax.set_title('Residual distributions per target')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    return _save(fig, output_dir, 'residual_violins.png')


def plot_window_mae_cdf(series_fixed, series_adaptive, output_dir):
    """Empirical CDF of (MAE_adaptive_window - MAE_non_adaptive_window) per target."""
    if not series_adaptive:
        return None
    targets = _targets_in(series_fixed)
    if not targets:
        return None
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    axes = axes.flatten()
    for ax, t in zip(axes, targets):
        pf, af = _windows(series_fixed, t)
        pa, aa = _windows(series_adaptive, t) if t in series_adaptive else (None, None)
        if pf is None or pa is None or pf.shape != pa.shape:
            ax.set_title(f'{_TAG[t]} - windows mismatch')
            continue
        mae_f = np.nanmean(np.abs(pf - af), axis=1)
        mae_a = np.nanmean(np.abs(pa - aa), axis=1)
        diff  = mae_a - mae_f
        diff  = diff[~np.isnan(diff)]
        if diff.size == 0:
            ax.set_title(f'{_TAG[t]} - no windows')
            continue
        x = np.sort(diff)
        y = np.arange(1, x.size + 1) / x.size
        ax.plot(x, y, color=_COLOR_ADAPTIVE, linewidth=1.4)
        ax.axvline(0.0, color='grey', linewidth=0.8, linestyle='--')
        ax.fill_between(x, 0, y, where=(x < 0), color=_COLOR_ADAPTIVE, alpha=0.18)
        wins   = int(np.sum(diff < 0))
        losses = int(np.sum(diff > 0))
        ties   = int(np.sum(diff == 0))
        annot  = f'adaptive better: {wins}/{diff.size}  worse: {losses}  tied: {ties}'
        if _HAS_SCIPY and diff.size >= 10 and np.any(diff != 0):
            try:
                stat = wilcoxon(diff, alternative='less', zero_method='zsplit')
                annot += f'\nWilcoxon p = {stat.pvalue:.3g}'
            except Exception:
                pass
        ax.set_title(f'{_TAG[t]} - per-window MAE difference')
        ax.set_xlabel('MAE_adaptive - MAE_non_adaptive (kW)')
        ax.set_ylabel('Empirical CDF')
        ax.grid(True, alpha=0.3)
        ax.text(0.02, 0.98, annot, transform=ax.transAxes,
                ha='left', va='top', fontsize=9,
                bbox=dict(facecolor='white', alpha=0.85, edgecolor='lightgrey',
                          boxstyle='round,pad=0.3'))
    plt.suptitle('Window-level paired MAE difference  (negative => adaptive wins)',
                 fontsize=13)
    plt.tight_layout()
    return _save(fig, output_dir, 'window_mae_cdf.png')


# ---------------------------------------------------------------------------
# 4. Ensemble-weight visualisations
# ---------------------------------------------------------------------------

def plot_ensemble_weights(weights_per_horizon, output_dir):
    """Stacked-area plot of LSTM / GBDT / TFT weights vs horizon step, per target."""
    if not weights_per_horizon:
        return None
    weights = weights_per_horizon.get('weights', weights_per_horizon)
    targets = [t for t in _ALL_TARGETS if t in weights]
    if not targets:
        return None
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    axes = axes.flatten()
    for ax, t in zip(axes, targets):
        w = weights[t]
        lstm = np.asarray(w.get('lstm', []), dtype=float)
        gbdt = np.asarray(w.get('gbdt', []), dtype=float)
        tft  = np.asarray(w.get('tft',  []), dtype=float)
        h = max(lstm.size, gbdt.size, tft.size)
        if h == 0:
            ax.set_title(f'{_TAG[t]} - no weights')
            continue
        x = np.arange(1, h + 1)

        def _pad(arr):
            if arr.size == 0:
                return np.zeros(h)
            if arr.size < h:
                return np.pad(arr, (0, h - arr.size), constant_values=arr[-1])
            return arr[:h]

        lstm = _pad(lstm)
        gbdt = _pad(gbdt)
        tft  = _pad(tft)
        ax.stackplot(x, gbdt, lstm, tft,
                     labels=['GBDT', 'LSTM', 'TFT'],
                     colors=[_COLOR_GBDT, _COLOR_LSTM, _COLOR_TFT],
                     alpha=0.85)
        ax.set_ylim(0, 1)
        ax.set_title(f'{_TAG[t]} - ensemble weights vs horizon')
        ax.set_xlabel('Horizon step (h)')
        ax.set_ylabel('Weight')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right', fontsize=8)
    plt.suptitle('Per-horizon ensemble weights', fontsize=13)
    plt.tight_layout()
    return _save(fig, output_dir, 'ensemble_weights.png')


def plot_weight_drift(weight_history, output_dir, retrain_steps=None, horizon_step=None):
    """LSTM / GBDT / TFT weight at a representative horizon step vs evaluation step."""
    if not weight_history:
        return None
    snapshots = [snap for snap in weight_history if 'weights' in snap]
    if not snapshots:
        return None
    sample = snapshots[0]['weights']
    targets = [t for t in _ALL_TARGETS if t in sample]
    if not targets:
        return None
    first_w = next(iter(sample.values()))
    h_default = max(len(first_w.get('lstm', [])), len(first_w.get('gbdt', [])),
                    len(first_w.get('tft',  [])))
    if h_default == 0:
        return None
    h_idx = int(horizon_step - 1) if horizon_step else h_default // 2
    h_idx = max(0, min(h_idx, h_default - 1))
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    axes = axes.flatten()
    for ax, t in zip(axes, targets):
        xs, lstm_v, gbdt_v, tft_v = [], [], [], []
        for snap in snapshots:
            w = snap['weights'].get(t)
            if not w:
                continue
            xs.append(snap.get('step', len(xs)))
            lstm_v.append(float(w['lstm'][h_idx]) if h_idx < len(w.get('lstm', [])) else np.nan)
            gbdt_v.append(float(w['gbdt'][h_idx]) if h_idx < len(w.get('gbdt', [])) else np.nan)
            tft_arr = w.get('tft', [])
            tft_v.append(float(tft_arr[h_idx]) if h_idx < len(tft_arr) else np.nan)
        if not xs:
            ax.set_title(f'{_TAG[t]} - no weight snapshots')
            continue
        ax.plot(xs, gbdt_v, label='GBDT', color=_COLOR_GBDT, linewidth=1.2)
        ax.plot(xs, lstm_v, label='LSTM', color=_COLOR_LSTM, linewidth=1.2)
        if not all(np.isnan(tft_v)):
            ax.plot(xs, tft_v, label='TFT', color=_COLOR_TFT, linewidth=1.2)
        for s in retrain_steps or []:
            ax.axvline(x=s, color='grey', linestyle=':', linewidth=0.7, alpha=0.7)
        ax.set_ylim(0, 1)
        ax.set_title(f'{_TAG[t]} - weight @ h+{h_idx + 1}')
        ax.set_xlabel('Evaluation step')
        ax.set_ylabel('Weight')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='upper right')
    plt.suptitle(f'Per-model weight drift over time (horizon step h+{h_idx + 1})',
                 fontsize=13)
    plt.tight_layout()
    return _save(fig, output_dir, 'weight_drift.png')


# ---------------------------------------------------------------------------
# 5. Condition-of-use diagnostics
# ---------------------------------------------------------------------------

def plot_mae_by_driver(series, data, output_dir, label='non_adaptive'):
    """Per-target MAE binned by selected drivers when present in ``data``."""
    targets = _targets_in(series)
    if not targets:
        return None
    driver_specs = []
    timestamps = series[targets[0]].get('timestamps')
    if isinstance(timestamps, pd.DatetimeIndex) and len(timestamps):
        driver_specs.append(('hour-of-day',
                             lambda t: pd.Series(timestamps[:len(series[t]['actual'])]).dt.hour.values))
    if 'temperature_2m' in data.columns and isinstance(timestamps, pd.DatetimeIndex):
        try:
            temp_series = data['temperature_2m'].reindex(timestamps)
            driver_specs.append(('temperature decile',
                                 lambda t: pd.qcut(temp_series.values[:len(series[t]['actual'])],
                                                   10, labels=False, duplicates='drop')))
        except Exception:
            pass
    driver_specs.append(('load quintile',
                         lambda t: pd.qcut(np.asarray(series[t]['actual']),
                                           5, labels=False, duplicates='drop')))
    n_rows = len(targets)
    n_cols = len(driver_specs)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3 * n_rows),
                              squeeze=False)
    for r, t in enumerate(targets):
        pf, af = _flat(series, t)
        if pf.size == 0:
            for c in range(n_cols):
                axes[r, c].set_title(f'{_TAG[t]} - no data')
            continue
        abs_err = np.abs(pf - af)
        for c, (name, builder) in enumerate(driver_specs):
            ax = axes[r, c]
            try:
                bins = builder(t)
            except Exception as exc:
                ax.set_title(f'{_TAG[t]} - {name} (n/a: {exc})')
                continue
            bins = np.asarray(bins)
            n = min(len(bins), abs_err.size)
            if n == 0:
                ax.set_title(f'{_TAG[t]} - {name} (no bins)')
                continue
            df = pd.DataFrame({'bin': bins[:n], 'err': abs_err[:n]}).dropna()
            if df.empty:
                ax.set_title(f'{_TAG[t]} - {name} (empty)')
                continue
            grouped = df.groupby('bin')['err'].mean().sort_index()
            ax.bar(grouped.index.astype(int), grouped.values,
                   color=_COLOR_FIXED if 'non' in label.lower() else _COLOR_ADAPTIVE,
                   alpha=0.85)
            ax.set_title(f'{_TAG[t]} - MAE by {name}')
            ax.set_xlabel(name)
            ax.set_ylabel('MAE (kW)')
            ax.grid(True, alpha=0.3, axis='y')
    plt.suptitle(f'Conditional MAE - {label}', fontsize=13)
    plt.tight_layout()
    fname = f'mae_by_driver_{label.lower().replace(" ", "_")}.png'
    return _save(fig, output_dir, fname)


def plot_error_calendar(series, output_dir, label='non_adaptive'):
    """Heatmap of |residual| with day-of-test rows x hour-of-day columns."""
    targets = _targets_in(series)
    if not targets:
        return None
    timestamps = series[targets[0]].get('timestamps')
    if not isinstance(timestamps, pd.DatetimeIndex) or len(timestamps) == 0:
        return None
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()
    for ax, t in zip(axes, targets):
        pf, af = _flat(series, t)
        if pf.size == 0:
            ax.set_title(f'{_TAG[t]} - no data')
            continue
        n = min(len(timestamps), pf.size)
        df = pd.DataFrame({
            'date': timestamps[:n].date,
            'hour': timestamps[:n].hour,
            'err':  np.abs(pf[:n] - af[:n]),
        })
        pivot = df.pivot_table(index='date', columns='hour', values='err', aggfunc='mean')
        im = ax.imshow(pivot.values, aspect='auto', cmap='inferno', interpolation='nearest')
        ax.set_title(f'{_TAG[t]} - |residual| heatmap')
        ax.set_xlabel('Hour of day')
        ax.set_ylabel('Day in test period')
        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label='|err| (kW)')
    plt.suptitle(f'Error calendar - {label}', fontsize=13)
    plt.tight_layout()
    fname = f'error_calendar_{label.lower().replace(" ", "_")}.png'
    return _save(fig, output_dir, fname)


def plot_peak_event_scatter(series_fixed, output_dir, series_adaptive=None):
    """Daily peak magnitude and peak-hour timing: forecast vs actual."""
    targets = _targets_in(series_fixed)
    if not targets:
        return None
    timestamps = series_fixed[targets[0]].get('timestamps')
    if not isinstance(timestamps, pd.DatetimeIndex) or len(timestamps) == 0:
        return None
    fig, axes = plt.subplots(len(targets), 2,
                              figsize=(12, 4 * len(targets)), squeeze=False)
    for r, t in enumerate(targets):
        pf, af = _flat(series_fixed, t)
        if pf.size == 0:
            continue
        n  = min(len(timestamps), pf.size)
        df = pd.DataFrame({
            'date': timestamps[:n].date,
            'hour': timestamps[:n].hour,
            'pred': pf[:n],
            'act':  af[:n],
        })
        daily_act_max  = df.groupby('date')['act'].max()
        daily_pred_max = df.groupby('date')['pred'].max()
        daily_act_hr   = df.loc[df.groupby('date')['act'].idxmax()].set_index('date')['hour']
        daily_pred_hr  = df.loc[df.groupby('date')['pred'].idxmax()].set_index('date')['hour']
        ax_mag = axes[r, 0]
        ax_hr  = axes[r, 1]
        lo = float(min(daily_act_max.min(), daily_pred_max.min()))
        hi = float(max(daily_act_max.max(), daily_pred_max.max(), 1e-6))
        ax_mag.plot([lo, hi], [lo, hi], color='black', linewidth=0.8, linestyle='--')
        ax_mag.scatter(daily_act_max, daily_pred_max, s=18, alpha=0.7,
                       color=_COLOR_FIXED, label='Non-adaptive')
        ax_hr.plot([0, 23], [0, 23], color='black', linewidth=0.8, linestyle='--')
        ax_hr.scatter(daily_act_hr, daily_pred_hr, s=18, alpha=0.7,
                      color=_COLOR_FIXED, label='Non-adaptive')
        if series_adaptive is not None and t in series_adaptive:
            pa, aa = _flat(series_adaptive, t)
            if pa.size:
                n2 = min(len(timestamps), pa.size)
                df2 = pd.DataFrame({
                    'date': timestamps[:n2].date,
                    'hour': timestamps[:n2].hour,
                    'pred': pa[:n2],
                    'act':  aa[:n2],
                })
                d_act_max  = df2.groupby('date')['act'].max()
                d_pred_max = df2.groupby('date')['pred'].max()
                d_act_hr   = df2.loc[df2.groupby('date')['act'].idxmax()].set_index('date')['hour']
                d_pred_hr  = df2.loc[df2.groupby('date')['pred'].idxmax()].set_index('date')['hour']
                ax_mag.scatter(d_act_max, d_pred_max, s=18, alpha=0.7,
                               color=_COLOR_ADAPTIVE, label='Adaptive')
                ax_hr.scatter(d_act_hr, d_pred_hr, s=18, alpha=0.7,
                              color=_COLOR_ADAPTIVE, label='Adaptive')
        ax_mag.set_xlim(lo, hi); ax_mag.set_ylim(lo, hi)
        ax_mag.set_aspect('equal', adjustable='box')
        ax_mag.set_xlabel('Actual peak (kW)')
        ax_mag.set_ylabel('Forecast peak (kW)')
        ax_mag.set_title(f'{_TAG[t]} - daily peak magnitude')
        ax_mag.grid(True, alpha=0.3)
        ax_mag.legend(fontsize=8, loc='upper left')
        ax_hr.set_xlim(-0.5, 23.5); ax_hr.set_ylim(-0.5, 23.5)
        ax_hr.set_xticks(range(0, 24, 3))
        ax_hr.set_yticks(range(0, 24, 3))
        ax_hr.set_aspect('equal', adjustable='box')
        ax_hr.set_xlabel('Actual peak hour')
        ax_hr.set_ylabel('Forecast peak hour')
        ax_hr.set_title(f'{_TAG[t]} - daily peak timing')
        ax_hr.grid(True, alpha=0.3)
        ax_hr.legend(fontsize=8, loc='upper left')
    plt.suptitle('Daily peak events - forecast vs actual', fontsize=13)
    plt.tight_layout()
    return _save(fig, output_dir, 'peak_event_scatter.png')


# ---------------------------------------------------------------------------
# 6. Operational integration plots
# ---------------------------------------------------------------------------

def plot_daily_energy_scatter(series_fixed, output_dir, series_adaptive=None):
    """Daily total kWh: forecast vs actual, one point per day per target."""
    targets = _targets_in(series_fixed)
    if not targets:
        return None
    timestamps = series_fixed[targets[0]].get('timestamps')
    if not isinstance(timestamps, pd.DatetimeIndex) or len(timestamps) == 0:
        return None
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    for ax, t in zip(axes, targets):
        pf, af = _flat(series_fixed, t)
        if pf.size == 0:
            ax.set_title(f'{_TAG[t]} - no data')
            continue
        n = min(len(timestamps), pf.size)
        df = pd.DataFrame({
            'date': timestamps[:n].date,
            'pred': pf[:n],
            'act':  af[:n],
        })
        daily = df.groupby('date').sum()
        lo = float(min(daily['act'].min(), daily['pred'].min()))
        hi = float(max(daily['act'].max(), daily['pred'].max(), 1e-6))
        ax.plot([lo, hi], [lo, hi], color='black', linewidth=0.8, linestyle='--')
        ax.scatter(daily['act'], daily['pred'], s=22, alpha=0.7,
                   color=_COLOR_FIXED, label='Non-adaptive')
        if series_adaptive is not None and t in series_adaptive:
            pa, aa = _flat(series_adaptive, t)
            if pa.size:
                n2 = min(len(timestamps), pa.size)
                df2 = pd.DataFrame({
                    'date': timestamps[:n2].date,
                    'pred': pa[:n2],
                    'act':  aa[:n2],
                })
                daily2 = df2.groupby('date').sum()
                ax.scatter(daily2['act'], daily2['pred'], s=22, alpha=0.7,
                           color=_COLOR_ADAPTIVE, label='Adaptive')
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('Actual daily energy (kWh)')
        ax.set_ylabel('Forecast daily energy (kWh)')
        ax.set_title(f'{_TAG[t]} - daily energy')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='upper left')
    plt.suptitle('Daily-energy scatter', fontsize=13)
    plt.tight_layout()
    return _save(fig, output_dir, 'daily_energy_scatter.png')


def plot_ramp_rate_errors(series_fixed, output_dir, series_adaptive=None):
    """Histogram of (Dp - Dy) per target - error in step-to-step ramps."""
    targets = _targets_in(series_fixed)
    if not targets:
        return None
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    for ax, t in zip(axes, targets):
        pf, af = _windows(series_fixed, t)
        if pf is None or pf.shape[1] < 2:
            ax.set_title(f'{_TAG[t]} - no per-window data')
            continue
        ramp_err_f = (np.diff(pf, axis=1) - np.diff(af, axis=1)).ravel()
        ramp_err_f = ramp_err_f[~np.isnan(ramp_err_f)]
        if ramp_err_f.size == 0:
            ax.set_title(f'{_TAG[t]} - no ramp data')
            continue
        ax.hist(ramp_err_f, bins=50, color=_COLOR_FIXED, alpha=0.6,
                label='Non-adaptive', density=True)
        if series_adaptive is not None and t in series_adaptive:
            pa, aa = _windows(series_adaptive, t)
            if pa is not None and pa.shape == pf.shape:
                ramp_err_a = (np.diff(pa, axis=1) - np.diff(aa, axis=1)).ravel()
                ramp_err_a = ramp_err_a[~np.isnan(ramp_err_a)]
                if ramp_err_a.size:
                    ax.hist(ramp_err_a, bins=50, color=_COLOR_ADAPTIVE, alpha=0.55,
                            label='Adaptive', density=True)
        ax.axvline(0.0, color='grey', linewidth=0.8, linestyle='--')
        ax.set_title(f'{_TAG[t]} - ramp-rate error (Dp - Dy)')
        ax.set_xlabel('Ramp-rate error (kW/h)')
        ax.set_ylabel('Density')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    plt.suptitle('Ramp-rate error distributions', fontsize=13)
    plt.tight_layout()
    return _save(fig, output_dir, 'ramp_rate_errors.png')


__all__ = [
    'plot_predictions', 'plot_performance_history',
    'plot_retrain_events', 'plot_retrain_events_legacy',
    'plot_horizon_error', 'plot_horizon_skill_score',
    'plot_pooled_scatter', 'plot_load_duration',
    'plot_residual_violins', 'plot_window_mae_cdf',
    'plot_ensemble_weights', 'plot_weight_drift',
    'plot_mae_by_driver', 'plot_error_calendar', 'plot_peak_event_scatter',
    'plot_daily_energy_scatter', 'plot_ramp_rate_errors',
]
