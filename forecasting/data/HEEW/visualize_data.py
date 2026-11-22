"""
HEEW Dataset Visualisations
===========================

Generates publication-quality charts from the processed 9-year hourly CSV.
Plots emphasise what is distinctive about HEEW / ASU Tempe Campus:
cooling-dominated load, multi-MW community scale, and the COVID-era
occupancy dip visible across 2020-2021.

Charts produced:

  1. targets_overview.png    -- 9-year time series of all load targets
  2. generation_overview.png -- 9-year PV time series + 30-day mean
  3. monthly_profiles.png    -- mean ± std by calendar month (grid subplots)
  4. daily_profiles.png      -- diurnal profile, weekday vs weekend
  5. seasonal_daily.png      -- hour-of-day profiles by season
  6. era_comparison.png      -- pre-COVID (2014-2019) vs post-COVID (2020-2022),
                                all targets
  7. pv_diurnal_heatmap.png  -- day-of-year × hour PV heatmap
  8. emissions_overview.png  -- GHG emission time series (publisher's column)
  9. missing_data.png        -- missingness heatmap (year × column)
  10. load_duration.png      -- sorted load-duration curves
  11. consumption_corr_*.png -- feature-impact heatmaps (KSRP style, per target)
  12. generation_corr_pv.png -- feature-impact heatmap (PV, KSRP style)

Usage
-----
    python forecasting/data/HEEW/visualize_data.py
    python forecasting/data/HEEW/visualize_data.py --csv processed/data_hourly.csv --out plots
"""

import argparse
import os
import sys
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

_DIR       = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_DIR)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from forecasting.constants import (
    TARGET_THERMAL_HEATING, TARGET_THERMAL_COOLING,
    TARGET_ELECTRIC, TARGET_PV,
)
from forecasting.data.HEEW.constants import PV_OP_START, PV_OP_END
from forecasting.data.HEEW._corr import plot_consumption_corr, plot_generation_corr

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
PALETTE = {
    'heating':  '#E05C5C',
    'cooling':  '#5B8DB8',
    'electric': '#6CBF8C',
    'pv':       '#F0A500',
    'emission': '#9B6B9F',
}
plt.rcParams.update({
    'figure.dpi': 120,
    'font.size': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
})
SEASON_LABELS = {1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Autumn'}
SEASON_COLORS = {1: '#5B8DB8', 2: '#6CBF8C', 3: '#F0A500', 4: '#E05C5C'}

TARGET_SPECS = [
    (TARGET_THERMAL_HEATING, 'Space Heating (kW)', PALETTE['heating']),
    (TARGET_THERMAL_COOLING, 'Space Cooling (kW)', PALETTE['cooling']),
    (TARGET_ELECTRIC,        'Electric Load (kW)', PALETTE['electric']),
    (TARGET_PV,              'PV Generation (kW)', PALETTE['pv']),
]

SEASON_MAP = {1: 1, 2: 1, 3: 2, 4: 2, 5: 2,
              6: 3, 7: 3, 8: 3, 9: 4, 10: 4, 11: 4, 12: 1}


def _save(fig: plt.Figure, path: str) -> None:
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {path}')


def _present(df, col):
    return col in df.columns and df[col].notna().any()


def _active_specs(df):
    return [(col, label, color) for col, label, color in TARGET_SPECS
            if _present(df, col)]


def _month_stats(df: pd.DataFrame, col: str) -> pd.DataFrame:
    g = df.groupby(df.index.month)[col]
    return pd.DataFrame({'mean': g.mean(), 'std': g.std(), 'median': g.median()})


# ---------------------------------------------------------------------------
# 1. Targets overview
# ---------------------------------------------------------------------------
def plot_targets_overview(df, out_dir):
    specs = _active_specs(df)
    if not specs:
        return

    fig, axes = plt.subplots(len(specs), 1, figsize=(16, 2.5 * len(specs)),
                             sharex=True)
    if len(specs) == 1:
        axes = [axes]

    for ax, (col, label, color) in zip(axes, specs):
        roll = df[col].rolling(window=24 * 30, center=True).mean()
        ax.fill_between(df.index, df[col], alpha=0.15, color=color)
        ax.plot(df.index, roll, color=color, linewidth=1.0, label='30-day mean')
        ax.set_ylabel(label, fontsize=9)
        ax.legend(loc='upper right', fontsize=8)

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    axes[-1].xaxis.set_major_locator(mdates.YearLocator())
    fig.suptitle('HEEW / ASU Tempe Campus — 9-Year Load Overview', fontsize=13)
    plt.tight_layout()
    _save(fig, os.path.join(out_dir, 'targets_overview.png'))


# ---------------------------------------------------------------------------
# 2. Generation overview
# ---------------------------------------------------------------------------
def plot_generation_overview(df, out_dir):
    if not _present(df, TARGET_PV):
        return
    pv = df.loc[(df.index >= PV_OP_START) & (df.index <= PV_OP_END), TARGET_PV]
    if pv.empty:
        return
    roll = pv.rolling(window=24 * 30, center=True).mean()
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.fill_between(pv.index, pv, alpha=0.15, color=PALETTE['pv'])
    ax.plot(pv.index, roll, color=PALETTE['pv'], linewidth=1.0, label='30-day mean')
    ax.set_ylabel('PV (kW)')
    ax.set_xlabel('Date')
    ax.set_title('HEEW — PV Generation (Community CN003)')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.legend(loc='upper right')
    plt.tight_layout()
    _save(fig, os.path.join(out_dir, 'generation_overview.png'))


# ---------------------------------------------------------------------------
# 3. Monthly profiles (mean ± std)
# ---------------------------------------------------------------------------
def plot_monthly_profiles(df, out_dir):
    months = np.arange(1, 13)
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    specs = _active_specs(df)
    if not specs:
        return
    n_cols = min(len(specs), 3)
    n_rows = (len(specs) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    for ax, (col, title, color) in zip(axes, specs):
        stats = _month_stats(df, col)
        ax.bar(months, stats['mean'], color=color, alpha=0.7,
               yerr=stats['std'], capsize=3, error_kw={'linewidth': 0.8})
        ax.plot(months, stats['median'], 'o--', color='black', markersize=4,
                linewidth=1, label='Median')
        ax.set_xticks(months)
        ax.set_xticklabels(month_names, fontsize=8)
        ax.set_title(title)
        ax.set_ylabel('kW')
        ax.legend(fontsize=8)
    for ax in axes[len(specs):]:
        ax.set_visible(False)

    fig.suptitle('Monthly Mean ± Std (hourly averages, 2014–2022)', fontsize=13)
    plt.tight_layout()
    _save(fig, os.path.join(out_dir, 'monthly_profiles.png'))


# ---------------------------------------------------------------------------
# 4. Diurnal profiles, weekday vs weekend
# ---------------------------------------------------------------------------
def plot_daily_profiles(df, out_dir):
    specs = _active_specs(df)
    if not specs:
        return
    hours = np.arange(24)
    is_wknd = df.index.dayofweek >= 5

    fig, axes = plt.subplots(1, len(specs), figsize=(5 * len(specs), 4))
    axes = np.atleast_1d(axes)

    for ax, (col, title, color) in zip(axes, specs):
        for is_we, label, ls in ((False, 'Weekday', '-'), (True, 'Weekend', '--')):
            sub = df.loc[is_wknd == is_we]
            mean = sub.groupby(sub.index.hour)[col].mean()
            std  = sub.groupby(sub.index.hour)[col].std()
            ax.plot(hours, mean, label=label, color=color, linestyle=ls)
            ax.fill_between(hours, mean - std, mean + std, alpha=0.15, color=color)
        ax.set_xlabel('Hour of day')
        ax.set_ylabel('kW')
        ax.set_title(title)
        ax.set_xticks(np.arange(0, 24, 3))
        ax.legend(fontsize=8)

    fig.suptitle('Diurnal Profiles (weekday vs weekend)', fontsize=13)
    plt.tight_layout()
    _save(fig, os.path.join(out_dir, 'daily_profiles.png'))


# ---------------------------------------------------------------------------
# 5. Seasonal hour-of-day
# ---------------------------------------------------------------------------
def plot_seasonal_daily(df, out_dir):
    specs = _active_specs(df)
    if not specs:
        return
    season = df.index.month.map(SEASON_MAP)
    hours = np.arange(24)

    fig, axes = plt.subplots(1, len(specs), figsize=(5 * len(specs), 4))
    axes = np.atleast_1d(axes)

    for ax, (col, title, _) in zip(axes, specs):
        for s_id, s_label in SEASON_LABELS.items():
            mask = season == s_id
            mean = df.loc[mask].groupby(df.loc[mask].index.hour)[col].mean()
            ax.plot(hours, mean, label=s_label, color=SEASON_COLORS[s_id], lw=2)
        ax.set_xlabel('Hour of day')
        ax.set_ylabel('kW')
        ax.set_title(title)
        ax.set_xticks(np.arange(0, 24, 3))
        ax.legend(fontsize=8)

    fig.suptitle('Diurnal Profiles by Season', fontsize=13)
    plt.tight_layout()
    _save(fig, os.path.join(out_dir, 'seasonal_daily.png'))


# ---------------------------------------------------------------------------
# 6. Era comparison: pre-COVID (2014–2019) vs post-COVID (2020–2022), all targets
# ---------------------------------------------------------------------------
def plot_era_comparison(df, out_dir):
    specs = _active_specs(df)
    if not specs:
        return

    pre  = df.loc['2014':'2019']
    post = df.loc['2020':'2022']

    n_cols = 2
    n_rows = (len(specs) + 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    axes = np.atleast_1d(axes).flatten()
    hours = np.arange(24)

    for ax, (col, title, color) in zip(axes, specs):
        for sub, era_label, era_color in (
            (pre,  '2014–2019 (pre-COVID)',  '#555555'),
            (post, '2020–2022 (post-COVID)', color),
        ):
            if sub.empty or not _present(sub, col):
                continue
            mean = sub.groupby(sub.index.hour)[col].mean()
            std  = sub.groupby(sub.index.hour)[col].std()
            ax.plot(hours, mean, label=era_label, color=era_color, linewidth=1.4)
            ax.fill_between(hours, mean - std, mean + std, alpha=0.2, color=era_color)
        ax.set_title(title)
        ax.set_xlabel('Hour of day')
        ax.set_ylabel('kW')
        ax.set_xticks(np.arange(0, 24, 3))
        ax.legend(fontsize=8)
    for ax in axes[len(specs):]:
        ax.set_visible(False)

    fig.suptitle('Pre-COVID vs Post-COVID — Mean Diurnal Patterns (All Targets)',
                 fontsize=13)
    plt.tight_layout()
    _save(fig, os.path.join(out_dir, 'era_comparison.png'))


# ---------------------------------------------------------------------------
# 7. PV diurnal heatmap
# ---------------------------------------------------------------------------
def plot_pv_heatmap(df, out_dir):
    if not _present(df, TARGET_PV):
        return
    pv = df.loc[(df.index >= PV_OP_START) & (df.index <= PV_OP_END), TARGET_PV]
    if pv.empty:
        return
    pivot = (pv.groupby([pv.index.dayofyear, pv.index.hour]).mean()
               .unstack(fill_value=0))
    fig, ax = plt.subplots(figsize=(11, 4.5))
    im = ax.imshow(pivot.T.values, aspect='auto', origin='lower',
                   cmap='YlOrRd', extent=[1, 366, 0, 24])
    ax.set_xlabel('Day of year')
    ax.set_ylabel('Hour of day')
    ax.set_title('HEEW — Mean PV (kW) by (day-of-year, hour)')
    fig.colorbar(im, ax=ax, label='PV (kW)')
    plt.tight_layout()
    _save(fig, os.path.join(out_dir, 'pv_diurnal_heatmap.png'))


# ---------------------------------------------------------------------------
# 8. Emissions overview
# ---------------------------------------------------------------------------
def plot_emissions_overview(df, out_dir):
    if not _present(df, 'emission_ton'):
        return
    roll = df['emission_ton'].rolling(window=24 * 30, center=True).mean()
    fig, ax = plt.subplots(figsize=(14, 3.5))
    ax.fill_between(df.index, df['emission_ton'], alpha=0.15, color=PALETTE['emission'])
    ax.plot(df.index, roll, color=PALETTE['emission'], linewidth=1.0,
            label='30-day mean')
    ax.set_ylabel('GHG emission (Ton CO₂-eq / hr)')
    ax.set_xlabel('Date')
    ax.set_title('HEEW — GHG Emission Time Series (Community CN003)')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.legend(loc='upper right')
    plt.tight_layout()
    _save(fig, os.path.join(out_dir, 'emissions_overview.png'))


# ---------------------------------------------------------------------------
# 9. Missingness heatmap
# ---------------------------------------------------------------------------
def plot_missing(df, out_dir):
    keep = [c for c in (
        TARGET_THERMAL_HEATING, TARGET_THERMAL_COOLING,
        TARGET_ELECTRIC, TARGET_PV, 'emission_ton',
        'local_air_temp_c', 'local_solar_rad_wm2',
    ) if c in df.columns]
    if not keep:
        return
    yearly = df[keep].isna().astype(int).resample('YE').mean()
    if yearly.empty:
        return
    fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(keep))))
    im = ax.imshow(yearly.T.values, aspect='auto', cmap='Reds',
                   extent=[yearly.index[0].year - 0.5,
                           yearly.index[-1].year + 0.5,
                           len(keep) - 0.5, -0.5])
    ax.set_yticks(np.arange(len(keep)))
    ax.set_yticklabels(keep, fontsize=8)
    ax.set_xticks(yearly.index.year)
    ax.set_xlabel('Year')
    ax.set_title('Annual Missing Fraction by Channel')
    plt.colorbar(im, ax=ax, label='Fraction missing')
    plt.xticks(rotation=45)
    plt.tight_layout()
    _save(fig, os.path.join(out_dir, 'missing_data.png'))


# ---------------------------------------------------------------------------
# 10. Load-duration curves
# ---------------------------------------------------------------------------
def plot_load_duration(df, out_dir):
    specs = _active_specs(df)
    if not specs:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    for col, label, color in specs:
        vals = df[col].dropna().sort_values(ascending=False).values
        pct = np.arange(len(vals)) / len(vals) * 100
        ax.plot(pct, vals, label=label, color=color, linewidth=1.4)
    ax.set_xlabel('% of hours')
    ax.set_ylabel('kW')
    ax.set_title('Load-Duration Curves (sorted hourly demand)')
    ax.legend(fontsize=8)
    plt.tight_layout()
    _save(fig, os.path.join(out_dir, 'load_duration.png'))


# ---------------------------------------------------------------------------
# 11 + 12. Correlation heatmaps — delegated to _corr.py (KSRP style)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run(csv_path: str, out_dir: str):
    print(f'[viz] Loading {csv_path} ...')
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    print(f'  {len(df):,} rows  ({df.index[0]} -> {df.index[-1]})')
    df.index.name = 'date'
    os.makedirs(out_dir, exist_ok=True)

    plot_targets_overview(df,    out_dir)
    plot_generation_overview(df, out_dir)
    plot_monthly_profiles(df,    out_dir)
    plot_daily_profiles(df,      out_dir)
    plot_seasonal_daily(df,      out_dir)
    plot_era_comparison(df,      out_dir)
    plot_pv_heatmap(df,          out_dir)
    plot_emissions_overview(df,  out_dir)
    plot_missing(df,             out_dir)
    plot_load_duration(df,       out_dir)

    plot_consumption_corr(df, out_dir, _DIR)
    plot_generation_corr(df,  out_dir, _DIR)

    print('[viz] Done.')


def _parse_args():
    p = argparse.ArgumentParser()
    base = _DIR
    p.add_argument('--csv', default=os.path.join(base, 'processed', 'data_hourly.csv'))
    p.add_argument('--out', default=os.path.join(base, 'plots'))
    return p.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    run(args.csv, args.out)
