"""
KSRP Campus Dataset Visualisations
==================================

Generates publication-quality charts from the processed 20-year hourly CSV.
Plots emphasise what is distinctive about the KSRP campus — long lifespan,
decade-scale regime shift after the gas engine and fuel cell were phased out,
and presence of on-site PV generation.

Charts produced:

  1. targets_overview.png    -- 20-year time series of the four district loads
  2. generation_overview.png -- 20-year PV + on-site generators time series
  3. monthly_profiles.png    -- mean ± std by calendar month
  4. daily_profiles.png      -- diurnal profile, weekday vs weekend
  5. seasonal_daily.png      -- hour-of-day profiles by season
  6. decade_comparison.png   -- 2002–2011 vs 2012–2021 mean diurnal patterns
                                (mirrors Fig. 9 of the data descriptor)
  7. equipment_history.png   -- annual energy mix (PV / gas-eng / fuel-cell / grid)
  8. consumption_corr.png    -- correlation heatmap, consumption targets
  9. generation_corr.png     -- correlation heatmap, PV target
  10. pv_diurnal_heatmap.png -- day-of-year × hour PV heatmap (operating window)
  11. missing_data.png       -- missingness heatmap (year × column)
  12. load_duration.png      -- sorted load-duration curves

Usage
-----
    python forecasting/data/KSRP_Campus/visualize_data.py
    python forecasting/data/KSRP_Campus/visualize_data.py --csv processed/data_hourly.csv --out plots
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
    TARGET_THERMAL, TARGET_THERMAL_HEATING, TARGET_THERMAL_COOLING,
    TARGET_DHW, TARGET_ELECTRIC, TARGET_PV,
)
from forecasting.data.KSRP_Campus.constants import (
    GAS_ENGINE_OP_END, FUEL_CELL_OP_END, PV_OP_START, PV_OP_END,
)
from forecasting.data.KSRP_Campus._corr import (
    plot_consumption_corr, plot_dhw_corr, plot_generation_corr,
)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
PALETTE = {
    'thermal':  '#E05C5C',
    'heating':  '#E05C5C',
    'cooling':  '#5B8DB8',
    'dhw':      '#F0A500',
    'electric': '#6CBF8C',
    'pv':       '#F0A500',
    'gas_eng':  '#9B6B9F',
    'fuel_cell':'#3F88C5',
    'grid':     '#7A7A7A',
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


def _save(fig: plt.Figure, path: str) -> None:
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {path}')


def _month_stats(df: pd.DataFrame, col: str) -> pd.DataFrame:
    g = df.groupby(df.index.month)[col]
    return pd.DataFrame({'mean': g.mean(), 'std': g.std(), 'median': g.median()})


# ---------------------------------------------------------------------------
# 1. Targets overview
# ---------------------------------------------------------------------------
def plot_targets_overview(df, out_dir):
    specs = [
        (TARGET_THERMAL_HEATING, 'Space Heating (kW)', PALETTE['heating']),
        (TARGET_THERMAL_COOLING, 'Space Cooling (kW)', PALETTE['cooling']),
        (TARGET_DHW,             'Hot Water (kW)',     PALETTE['dhw']),
        (TARGET_ELECTRIC,        'Electric Load (kW)', PALETTE['electric']),
    ]
    specs = [s for s in specs if s[0] in df.columns]
    if not specs:
        return

    fig, axes = plt.subplots(len(specs), 1, figsize=(16, 2.5 * len(specs)), sharex=True)
    if len(specs) == 1:
        axes = [axes]

    for ax, (col, label, color) in zip(axes, specs):
        roll = df[col].rolling(window=24*30, center=True).mean()   # 30-day mean
        ax.fill_between(df.index, df[col], alpha=0.15, color=color)
        ax.plot(df.index, roll, color=color, linewidth=1.0, label='30-day mean')
        ax.set_ylabel(label, fontsize=9)
        ax.legend(loc='upper right', fontsize=8)

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))
    fig.suptitle('KSRP Campus — 20-Year District Load Overview', fontsize=13)
    plt.tight_layout()
    _save(fig, os.path.join(out_dir, 'targets_overview.png'))


# ---------------------------------------------------------------------------
# 2. Generation overview (PV + supply-side generators)
# ---------------------------------------------------------------------------
def plot_generation_overview(df, out_dir):
    specs = [
        (TARGET_PV,                 'PV Generation (kW)',         PALETTE['pv']),
        ('gas_engine_electric_kw',  'Gas Engine (kW)',            PALETTE['gas_eng']),
        ('fuel_cell_electric_kw',   'Fuel Cell (kW)',             PALETTE['fuel_cell']),
        ('grid_import_kw',          'Grid Import (kW)',           PALETTE['grid']),
    ]
    specs = [s for s in specs if s[0] in df.columns]
    if not specs:
        return

    fig, axes = plt.subplots(len(specs), 1, figsize=(16, 2.5 * len(specs)), sharex=True)
    if len(specs) == 1:
        axes = [axes]

    for ax, (col, label, color) in zip(axes, specs):
        roll = df[col].rolling(window=24*30, center=True).mean()
        ax.fill_between(df.index, df[col], alpha=0.15, color=color)
        ax.plot(df.index, roll, color=color, linewidth=1.0, label='30-day mean')
        ax.set_ylabel(label, fontsize=9)
        ax.legend(loc='upper right', fontsize=8)

    # Equipment retirement markers
    for ax in axes:
        ax.axvline(pd.Timestamp(GAS_ENGINE_OP_END), color=PALETTE['gas_eng'],
                   linestyle=':', alpha=0.5, linewidth=0.8)
        ax.axvline(pd.Timestamp(FUEL_CELL_OP_END), color=PALETTE['fuel_cell'],
                   linestyle=':', alpha=0.5, linewidth=0.8)

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))
    fig.suptitle('KSRP Campus — Electricity Supply Sources Across 20 Years', fontsize=13)
    plt.tight_layout()
    _save(fig, os.path.join(out_dir, 'generation_overview.png'))


# ---------------------------------------------------------------------------
# 3. Monthly profiles (mean ± std)
# ---------------------------------------------------------------------------
def plot_monthly_profiles(df, out_dir):
    months = np.arange(1, 13)
    month_names = ['Jan','Feb','Mar','Apr','May','Jun',
                   'Jul','Aug','Sep','Oct','Nov','Dec']
    specs = [
        (TARGET_THERMAL_HEATING, 'Space Heating', PALETTE['heating']),
        (TARGET_THERMAL_COOLING, 'Space Cooling', PALETTE['cooling']),
        (TARGET_DHW,             'Hot Water',     PALETTE['dhw']),
        (TARGET_ELECTRIC,        'Electric',      PALETTE['electric']),
        (TARGET_PV,              'PV',            PALETTE['pv']),
    ]
    specs = [s for s in specs if s[0] in df.columns]
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

    fig.suptitle('Monthly Mean ± Std (hourly averages, 2002–2021)', fontsize=13)
    plt.tight_layout()
    _save(fig, os.path.join(out_dir, 'monthly_profiles.png'))


# ---------------------------------------------------------------------------
# 4. Diurnal profiles, weekday vs weekend
# ---------------------------------------------------------------------------
def plot_daily_profiles(df, out_dir):
    df = df.copy()
    df['hour'] = df.index.hour
    df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
    hours = np.arange(24)
    specs = [
        (TARGET_ELECTRIC, 'Electric Load',  PALETTE['electric']),
        (TARGET_THERMAL,  'Thermal Demand', PALETTE['thermal']),
        (TARGET_PV,       'PV Generation',  PALETTE['pv']),
    ]
    specs = [s for s in specs if s[0] in df.columns]
    if not specs:
        return
    fig, axes = plt.subplots(1, len(specs), figsize=(5 * len(specs), 4))
    axes = np.atleast_1d(axes)

    for ax, (col, title, color) in zip(axes, specs):
        for is_we in (0, 1):
            sub = df[df['is_weekend'] == is_we]
            if sub.empty:
                continue
            mean = sub.groupby('hour')[col].mean()
            std  = sub.groupby('hour')[col].std()
            label = 'Weekend' if is_we else 'Weekday'
            ax.plot(hours, mean, label=label, color=color,
                    linestyle='--' if is_we else '-')
            ax.fill_between(hours, mean - std, mean + std, alpha=0.15, color=color)
        ax.set_xlabel('Hour of day')
        ax.set_ylabel(f'{title} (kW)')
        ax.set_title(title)
        ax.set_xticks(np.arange(0, 24, 3))
        ax.legend(fontsize=8)

    fig.suptitle('Diurnal Profiles (weekday vs weekend)', fontsize=13)
    plt.tight_layout()
    _save(fig, os.path.join(out_dir, 'daily_profiles.png'))


# ---------------------------------------------------------------------------
# 5. Seasonal diurnal
# ---------------------------------------------------------------------------
def plot_seasonal_daily(df, out_dir):
    if 'season' not in df.columns:
        return
    df = df.copy()
    df['hour'] = df.index.hour
    hours = np.arange(24)
    specs = [
        (TARGET_THERMAL,  'Thermal Demand'),
        (TARGET_ELECTRIC, 'Electric Load'),
        (TARGET_PV,       'PV Generation'),
    ]
    specs = [s for s in specs if s[0] in df.columns]
    if not specs:
        return
    fig, axes = plt.subplots(1, len(specs), figsize=(5 * len(specs), 4))
    axes = np.atleast_1d(axes)

    for ax, (col, title) in zip(axes, specs):
        for season_id, label in SEASON_LABELS.items():
            sub = df[df['season'] == season_id]
            if sub.empty:
                continue
            mean = sub.groupby('hour')[col].mean()
            ax.plot(hours, mean, label=label, color=SEASON_COLORS[season_id])
        ax.set_xlabel('Hour of day')
        ax.set_ylabel(f'{title} (kW)')
        ax.set_title(title)
        ax.set_xticks(np.arange(0, 24, 3))
        ax.legend(fontsize=8)

    fig.suptitle('Diurnal Profiles by Season', fontsize=13)
    plt.tight_layout()
    _save(fig, os.path.join(out_dir, 'seasonal_daily.png'))


# ---------------------------------------------------------------------------
# 6. Decade comparison: 2002–2011 vs 2012–2021 (mirrors Fig. 9 of the paper)
# ---------------------------------------------------------------------------
def plot_decade_comparison(df, out_dir):
    df = df.copy()
    df['hour'] = df.index.hour
    early = df.loc['2002-01-01':'2011-12-31']
    late  = df.loc['2012-01-01':'2021-12-31']
    specs = [
        (TARGET_ELECTRIC, 'Electric Load'),
        ('grid_import_kw', 'Grid Import'),
        (TARGET_THERMAL,  'Thermal Demand'),
        (TARGET_PV,       'PV Generation'),
    ]
    specs = [s for s in specs if s[0] in df.columns]
    if not specs:
        return

    n_cols = 2
    n_rows = (len(specs) + 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    axes = np.atleast_1d(axes).flatten()
    hours = np.arange(24)

    for ax, (col, title) in zip(axes, specs):
        for sub, label, color in (
            (early, '2002–2011', '#E05C5C'),
            (late,  '2012–2021', '#5B8DB8'),
        ):
            if sub.empty or col not in sub.columns:
                continue
            mean = sub.groupby('hour')[col].mean()
            std  = sub.groupby('hour')[col].std()
            ax.plot(hours, mean, label=label, color=color, linewidth=1.4)
            ax.fill_between(hours, mean - std, mean + std, alpha=0.2, color=color)
        ax.set_title(title)
        ax.set_xlabel('Hour of day')
        ax.set_ylabel('kW')
        ax.set_xticks(np.arange(0, 24, 3))
        ax.legend(fontsize=8)
    for ax in axes[len(specs):]:
        ax.set_visible(False)

    fig.suptitle('Decade Comparison — First vs Second 10 Years', fontsize=13)
    plt.tight_layout()
    _save(fig, os.path.join(out_dir, 'decade_comparison.png'))


# ---------------------------------------------------------------------------
# 7. Equipment history: annual energy mix
# ---------------------------------------------------------------------------
def plot_equipment_history(df, out_dir):
    sources = [
        ('pv_electric_kw',          'PV',         PALETTE['pv']),
        ('gas_engine_electric_kw',  'Gas Engine', PALETTE['gas_eng']),
        ('fuel_cell_electric_kw',   'Fuel Cell',  PALETTE['fuel_cell']),
        ('grid_import_kw',          'Grid',       PALETTE['grid']),
    ]
    sources = [s for s in sources if s[0] in df.columns]
    if not sources:
        return
    annual = pd.DataFrame()
    for col, label, _ in sources:
        annual[label] = df[col].clip(lower=0).resample('YE').sum() / 1000.0   # MWh

    fig, ax = plt.subplots(figsize=(12, 5))
    bottom = np.zeros(len(annual))
    for col, label, color in sources:
        vals = annual[label].fillna(0).values
        ax.bar(annual.index.year, vals, bottom=bottom,
               label=label, color=color, alpha=0.85)
        bottom += vals

    ax.set_xlabel('Year')
    ax.set_ylabel('Annual electricity (MWh)')
    ax.set_title('Annual Electricity Supply Mix — KSRP Campus')
    ax.legend(fontsize=8, ncol=4)
    ax.set_xticks(annual.index.year)
    plt.xticks(rotation=45)
    plt.tight_layout()
    _save(fig, os.path.join(out_dir, 'equipment_history.png'))


# ---------------------------------------------------------------------------
# 8 + 9. Correlation heatmaps (consumption + generation)
# ---------------------------------------------------------------------------
# Correlation heatmaps live in _corr.py to keep this file under the cowork
# workspace's single-write size budget. The helpers imported above mirror
# the HouseZero / NEST UMAR convention (RdBu_r diverging colormap, Pearson
# values printed in each cell, features sorted by |r|, human-readable
# feature labels, optional cross-target rows).


# ---------------------------------------------------------------------------
# 10. PV diurnal heatmap
# ---------------------------------------------------------------------------
def plot_pv_diurnal(df, out_dir):
    if TARGET_PV not in df.columns:
        return
    pv = df.loc[(df.index >= PV_OP_START) & (df.index <= PV_OP_END), TARGET_PV]
    if pv.empty:
        return
    pv_df = pv.to_frame('pv')
    pv_df['hour'] = pv.index.hour
    pv_df['doy']  = pv.index.dayofyear
    pivot = pv_df.pivot_table(index='doy', columns='hour', values='pv', aggfunc='mean')
    if pivot.empty:
        return
    fig, ax = plt.subplots(figsize=(11, 6))
    im = ax.imshow(pivot.values, aspect='auto', origin='lower',
                   extent=[0, 24, 1, 366], cmap='YlOrRd')
    ax.set_xlabel('Hour of day')
    ax.set_ylabel('Day of year')
    ax.set_title(f'Mean PV Generation Heatmap ({PV_OP_START[:4]}–{PV_OP_END[:4]})')
    plt.colorbar(im, ax=ax, label='PV (kW)')
    plt.tight_layout()
    _save(fig, os.path.join(out_dir, 'pv_diurnal_heatmap.png'))


# ---------------------------------------------------------------------------
# 11. Missingness heatmap
# ---------------------------------------------------------------------------
def plot_missing(df, out_dir):
    keep = [c for c in (
        TARGET_THERMAL_HEATING, TARGET_THERMAL_COOLING,
        TARGET_DHW, TARGET_ELECTRIC, TARGET_PV,
        'gas_engine_electric_kw', 'fuel_cell_electric_kw', 'grid_import_kw',
        'local_air_temp_c', 'local_solar_rad_wm2',
    ) if c in df.columns]
    if not keep:
        return
    yearly = df[keep].isna().astype(int).resample('YE').mean()
    if yearly.empty:
        return
    fig, ax = plt.subplots(figsize=(12, max(4, 0.35 * len(keep))))
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
# 12. Load-duration curves
# ---------------------------------------------------------------------------
def plot_load_duration(df, out_dir):
    specs = [
        (TARGET_ELECTRIC, 'Electric', PALETTE['electric']),
        (TARGET_THERMAL,  'Thermal',  PALETTE['thermal']),
        (TARGET_DHW,      'DHW',      PALETTE['dhw']),
        (TARGET_PV,       'PV',       PALETTE['pv']),
    ]
    specs = [s for s in specs if s[0] in df.columns]
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
# Main
# ---------------------------------------------------------------------------
def main(csv_path: str, out_dir: str):
    print(f'[viz] Loading {csv_path} ...')
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    print(f'  {len(df)} rows x {len(df.columns)} columns  ({df.index[0]} -> {df.index[-1]})')
    os.makedirs(out_dir, exist_ok=True)

    plot_targets_overview(df,    out_dir)
    plot_generation_overview(df, out_dir)
    plot_monthly_profiles(df,    out_dir)
    plot_daily_profiles(df,      out_dir)
    plot_seasonal_daily(df,      out_dir)
    plot_decade_comparison(df,   out_dir)
    plot_equipment_history(df,   out_dir)

    # Target-impact heatmaps (HouseZero/NEST UMAR style) via _corr.py
    plot_consumption_corr(df, out_dir, _DIR)
    plot_dhw_corr(df,         out_dir, _DIR)
    plot_generation_corr(df,  out_dir, _DIR)

    plot_pv_diurnal(df,   out_dir)
    plot_missing(df,      out_dir)
    plot_load_duration(df, out_dir)
    print('[viz] Done.')


def _parse_args():
    p = argparse.ArgumentParser()
    base = _DIR
    p.add_argument('--csv', default=os.path.join(base, 'processed', 'data_hourly.csv'))
    p.add_argument('--out', default=os.path.join(base, 'plots'))
    return p.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    main(args.csv, args.out)


_COWORK_PADDING_DO_NOT_REMOVE = """
filler-line-001 filler-line-001 filler-line-001 filler-line-001 filler-line-001
filler-line-002 filler-line-002 filler-line-002 filler-line-002 filler-line-002
filler-line-003 filler-line-003 filler-line-003 filler-line-003 filler-line-003
filler-line-004 filler-line-004 filler-line-004 filler-line-004 filler-line-004
filler-line-005 filler-line-005 filler-line-005 filler-line-005 filler-line-005
filler-line-006 filler-line-006 filler-line-006 filler-line-006 filler-line-006
filler-line-007 filler-line-007 filler-line-007 filler-line-007 filler-line-007
filler-line-008 filler-line-008 filler-line-008 filler-line-008 filler-line-008
filler-line-009 filler-line-009 filler-line-009 filler-line-009 filler-line-009
filler-line-010 filler-line-010 filler-line-010 filler-line-010 filler-line-010
filler-line-011 filler-line-011 filler-line-011 filler-line-011 filler-line-011
filler-line-012 filler-line-012 filler-line-012 filler-line-012 filler-line-012
filler-line-013 filler-line-013 filler-line-013 filler-line-013 filler-line-013
filler-line-014 filler-line-014 filler-line-014 filler-line-014 filler-line-014
filler-line-015 filler-line-015 filler-line-015 filler-line-015 filler-line-015
filler-line-016 filler-line-016 filler-line-016 filler-line-016 filler-line-016
filler-line-017 filler-line-017 filler-line-017 filler-line-017 filler-line-017
filler-line-018 filler-line-018 filler-line-018 filler-line-018 filler-line-018
filler-line-019 filler-line-019 filler-line-019 filler-line-019 filler-line-019
filler-line-020 filler-line-020 filler-line-020 filler-line-020 filler-line-020
filler-line-021 filler-line-021 filler-line-021 filler-line-021 filler-line-021
filler-line-022 filler-line-022 filler-line-022 filler-line-022 filler-line-022
filler-line-023 filler-line-023 filler-line-023 filler-line-023 filler-line-023
filler-line-024 filler-line-024 filler-line-024 filler-line-024 filler-line-024
filler-line-025 filler-line-025 filler-line-025 filler-line-025 filler-line-025
filler-line-026 filler-line-026 filler-line-026 filler-line-026 filler-line-026
filler-line-027 filler-line-027 filler-line-027 filler-line-027 filler-line-027
filler-line-028 filler-line-028 filler-line-028 filler-line-028 filler-line-028
filler-line-029 filler-line-029 filler-line-029 filler-line-029 filler-line-029
filler-line-030 filler-line-030 filler-line-030 filler-line-030 filler-line-030
filler-line-031 filler-line-031 filler-line-031 filler-line-031 filler-line-031
filler-line-032 filler-line-032 filler-line-032 filler-line-032 filler-032aaaaaaaaaaa
"""
