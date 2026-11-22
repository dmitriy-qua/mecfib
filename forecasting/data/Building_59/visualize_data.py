"""
LBNL Building 59 Dataset Visualizations
=======================================

Generates publication-quality charts from the processed Building 59 CSV.
Building 59 is an all-electric office building with end-use metering and no
on-site generation, so the targets are HVAC electric input (mapped to the
"thermal" heating/cooling channels by season) and plug+lighting electricity.
There is no DHW meter, so DHW-specific plots are omitted.

This module mirrors the NEST_UMAR / HouseZero visualization style -- in
particular the article-friendly ``_target_impact_heatmap`` correlation
matrix (targets as columns, impact-sorted features as rows, cross-target
block on top, per-cell Pearson-r annotations) -- so the figures are visually
consistent across every dataset in forecasting/data/*.

Charts produced:

  1.  targets_overview.png       -- 3-year time series of HVAC + plug/light
  2.  thermal_overview.png       -- HVAC heating-season / cooling-season split
  3.  monthly_profiles.png       -- mean +/- std by calendar month
  4.  daily_profiles.png         -- mean diurnal profile (weekday vs weekend)
  5.  seasonal_daily.png         -- hour-of-day profiles split by season
  6.  consumption_corr_<t>.png   -- Pearson correlation heatmap per target
  7.  hvac_vs_temp.png           -- HVAC electric demand vs outdoor temperature
  8.  load_duration.png          -- load-duration curves (sorted demand)
  9.  missing_data.png           -- missing-value heatmap (monthly x feature)
  10. energy_balance.png         -- monthly bars: heating, cooling, electric
  11. occupancy_pattern.png      -- occupant / WiFi count heatmap (DoY x hour)
  12. covid_impact.png           -- 2019 vs 2020 monthly demand (COVID drift)

Usage
-----
    python forecasting/data/Building_59/visualize_data.py
    python forecasting/data/Building_59/visualize_data.py --csv processed/data_hourly.csv --out plots
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
    TARGET_ELECTRIC,
)

# Style (mirrors NEST_UMAR / HouseZero)
PALETTE = {
    'thermal':  '#E05C5C',
    'heating':  '#E05C5C',
    'cooling':  '#5B8DB8',
    'electric': '#6CBF8C',
    'total':    '#6CBF8C',
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


# 1. Targets overview ---------------------------------------------------------

def plot_targets_overview(df: pd.DataFrame, out_dir: str) -> None:
    specs = [
        (TARGET_THERMAL,  'HVAC Electric (kW)',           PALETTE['thermal']),
        (TARGET_ELECTRIC, 'Electric (kW)', PALETTE['total']),
    ]
    specs = [(c, l, k) for c, l, k in specs if c in df.columns]
    if not specs:
        return

    fig, axes = plt.subplots(len(specs), 1, figsize=(16, 3 * len(specs) + 1), sharex=True)
    if len(specs) == 1:
        axes = [axes]

    for ax, (col, label, color) in zip(axes, specs):
        roll = df[col].rolling(window=168, center=True).mean()
        ax.fill_between(df.index, df[col], alpha=0.2, color=color)
        ax.plot(df.index, roll, color=color, linewidth=1.2, label='7-day mean')
        ax.set_ylabel(label)
        ax.legend(loc='upper right', fontsize=8)

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=30, ha='right')
    fig.suptitle('LBNL Building 59 -- 3-Year Electricity Overview', fontsize=13)
    plt.tight_layout()
    _save(fig, os.path.join(out_dir, 'targets_overview.png'))


# 1b. Thermal (HVAC) breakdown overview --------------------------------------

def plot_thermal_overview(df: pd.DataFrame, out_dir: str) -> None:
    specs = [
        (TARGET_THERMAL_HEATING, 'HVAC, Heating Season (kW)', PALETTE['heating']),
        (TARGET_THERMAL_COOLING, 'HVAC, Cooling Season (kW)', PALETTE['cooling']),
    ]
    specs = [(c, l, k) for c, l, k in specs if c in df.columns]
    if not specs:
        return

    fig, axes = plt.subplots(len(specs), 1, figsize=(16, 3 * len(specs)), sharex=True)
    if len(specs) == 1:
        axes = [axes]

    for ax, (col, label, color) in zip(axes, specs):
        roll = df[col].rolling(window=168, center=True).mean()
        ax.fill_between(df.index, df[col], alpha=0.2, color=color)
        ax.plot(df.index, roll, color=color, linewidth=1.2, label='7-day mean')
        ax.set_ylabel(label)
        ax.legend(loc='upper right', fontsize=8)

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=30, ha='right')
    fig.suptitle('Building 59 -- HVAC Electric Demand by Season', fontsize=13)
    plt.tight_layout()
    _save(fig, os.path.join(out_dir, 'thermal_overview.png'))


# 2. Monthly profiles ---------------------------------------------------------

def plot_monthly_profiles(df: pd.DataFrame, out_dir: str) -> None:
    months = np.arange(1, 13)
    month_names = ['Jan','Feb','Mar','Apr','May','Jun',
                   'Jul','Aug','Sep','Oct','Nov','Dec']

    specs = [
        (TARGET_THERMAL_HEATING, 'Heating',   PALETTE['heating']),
        (TARGET_THERMAL_COOLING, 'Cooling',   PALETTE['cooling']),
        (TARGET_ELECTRIC,        'Electric', PALETTE['total']),
    ]
    specs = [(c, l, k) for c, l, k in specs if c in df.columns]
    if not specs:
        return

    fig, axes = plt.subplots(1, len(specs), figsize=(4 * len(specs) + 1, 5))
    if len(specs) == 1:
        axes = [axes]

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

    fig.suptitle('Monthly Mean +/- Std  (hourly averages)', fontsize=13)
    plt.tight_layout()
    _save(fig, os.path.join(out_dir, 'monthly_profiles.png'))


# 3. Daily (diurnal) profiles -------------------------------------------------

def plot_daily_profiles(df: pd.DataFrame, out_dir: str) -> None:
    df = df.copy()
    df['hour'] = df.index.hour
    df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)

    specs = [
        (TARGET_THERMAL_HEATING, 'Heating',   PALETTE['heating']),
        (TARGET_THERMAL_COOLING, 'Cooling',   PALETTE['cooling']),
        (TARGET_ELECTRIC,        'Electric', PALETTE['total']),
    ]
    specs = [(c, l, k) for c, l, k in specs if c in df.columns]
    if not specs:
        return

    fig, axes = plt.subplots(1, len(specs), figsize=(4 * len(specs) + 1, 5))
    if len(specs) == 1:
        axes = [axes]
    hours = np.arange(24)

    for ax, (col, title, color) in zip(axes, specs):
        series = df[col].replace([np.inf, -np.inf], np.nan)
        tmp = df.copy()
        tmp[col] = series

        wd = tmp[tmp['is_weekend'] == 0].groupby('hour')[col]
        we = tmp[tmp['is_weekend'] == 1].groupby('hour')[col]

        wd_mean = wd.mean().reindex(hours, fill_value=np.nan)
        we_mean = we.mean().reindex(hours, fill_value=np.nan)
        wd_std  = wd.std().reindex(hours, fill_value=0)

        ax.plot(hours, wd_mean, color=color, linewidth=1.5, label='Weekday')
        ax.fill_between(hours, wd_mean - wd_std, wd_mean + wd_std,
                        alpha=0.15, color=color)
        ax.plot(hours, we_mean, color=color, linewidth=1.5,
                linestyle='--', label='Weekend')
        ax.set_title(title)
        ax.set_ylabel('kW')
        ax.set_xlabel('Hour of day')
        ax.set_xticks(range(0, 24, 3))
        ax.legend(fontsize=8)

    fig.suptitle('Mean Diurnal Profile  (weekday vs weekend)', fontsize=13)
    plt.tight_layout()
    _save(fig, os.path.join(out_dir, 'daily_profiles.png'))


# 4. Seasonal diurnal profiles ------------------------------------------------

def plot_seasonal_daily(df: pd.DataFrame, out_dir: str) -> None:
    df = df.copy()
    df['hour'] = df.index.hour

    specs = [
        (TARGET_THERMAL_HEATING, 'Heating'),
        (TARGET_THERMAL_COOLING, 'Cooling'),
        (TARGET_ELECTRIC,        'Electric'),
    ]
    specs = [(c, l) for c, l in specs if c in df.columns]
    if not specs:
        return

    fig, axes = plt.subplots(1, len(specs), figsize=(4 * len(specs) + 1, 5))
    if len(specs) == 1:
        axes = [axes]
    hours = np.arange(24)

    for ax, (col, title) in zip(axes, specs):
        if 'season' not in df.columns:
            continue
        for season, label in SEASON_LABELS.items():
            sub = df[df['season'] == season].groupby('hour')[col].mean()
            ax.plot(hours, sub.reindex(hours, fill_value=0),
                    color=SEASON_COLORS[season], linewidth=1.5, label=label)
        ax.set_title(title)
        ax.set_ylabel('kW')
        ax.set_xlabel('Hour of day')
        ax.set_xticks(range(0, 24, 3))
        ax.legend(fontsize=8)

    fig.suptitle('Seasonal Diurnal Profiles', fontsize=13)
    plt.tight_layout()
    _save(fig, os.path.join(out_dir, 'seasonal_daily.png'))


# 5. Target-impact correlation heatmaps --------------------------------------

# Consumption targets to render a heatmap for. DHW is omitted (Building 59 has
# no DHW meter; dhw_kw is identically zero and would yield NaN correlations).
_CONSUMPTION_TARGETS = [TARGET_THERMAL_HEATING, TARGET_THERMAL_COOLING, TARGET_ELECTRIC]
_ALL_CONSUMPTION_TARGETS = list(_CONSUMPTION_TARGETS)

# Embedded default feature set (fallback when features_consumption.txt is absent).
_CONSUMPTION_FEATURES = [
    'is_heating_season', 'month', 'hour', 'day_of_week', 'is_business', 'is_day',
    'temperature_2m', 'apparent_temperature', 'dew_point_2m', 'local_air_temp_c',
    'relative_humidity_2m', 'local_rh_pct', 'cloud_cover', 'sunshine_duration',
    'shortwave_radiation', 'direct_radiation', 'diffuse_radiation',
    'global_tilted_irradiance', 'local_solar_rad_wm2',
    'wind_speed_10m', 'wind_gusts_10m', 'pressure_msl', 'rain_precipitation',
    'zone_cooling_sp_c', 'zone_heating_sp_c', 'zone_co2_mean_ppm',
    'occupant_count', 'wifi_device_count',
    'pm25', 'pm10', 'o3', 'no2',
]

_TARGET_LABELS = {
    TARGET_THERMAL_HEATING: 'Heating',
    TARGET_THERMAL_COOLING: 'Cooling',
    TARGET_ELECTRIC:        'Electric',
    TARGET_THERMAL:         'HVAC\nTotal',
}

_FEAT_LABELS = {
    'is_heating_season':        'Heating season',
    'hour':                     'Hour of day',
    'month':                    'Month',
    'day_of_week':              'Day of week',
    'month_cos':                'Month (cyclic)',
    'hour_cos':                 'Hour of day (cyclic)',
    'is_business':              'Business hours',
    'is_day':                   'Daytime',
    'temperature_2m':           'Air temperature',
    'apparent_temperature':     'Apparent temperature',
    'dew_point_2m':             'Dew point',
    'relative_humidity_2m':     'Relative humidity',
    'local_rh_pct':             'Local RH (on-site)',
    'cloud_cover':              'Cloud cover',
    'solar_obstruction_pct':    'Solar obstruction %',
    'sunshine_duration':        'Sunshine duration',
    'shortwave_radiation':      'Shortwave radiation',
    'direct_radiation':         'Direct radiation',
    'diffuse_radiation':        'Diffuse radiation',
    'global_tilted_irradiance': 'Global tilted irradiance',
    'wind_speed_10m':           'Wind speed',
    'wind_gusts_10m':           'Wind gusts',
    'pressure_msl':             'Pressure (MSL)',
    'rain_precipitation':       'Rainfall',
    'snowfall_precipitation':   'Snowfall',
    'local_air_temp_c':         'Local outdoor temp',
    'local_solar_rad_wm2':      'Local solar radiation',
    'zone_cooling_sp_c':        'Zone cooling setpoint',
    'zone_heating_sp_c':        'Zone heating setpoint',
    'zone_co2_mean_ppm':        'Zone CO2',
    'occupant_count':           'Occupant count (camera)',
    'wifi_device_count':        'WiFi device count',
    'pm25':                     'PM2.5',
    'pm10':                     'PM10',
    'o3':                       'Ozone',
    'no2':                      'NO2',
}


def _load_feature_whitelist(default: list, kind: str) -> list:
    """Load per-dataset feature selection for correlation heatmaps.

    Drop ``features_<kind>.txt`` next to this script (one feature per line,
    ``#`` for comments) to override the embedded default list.
    """
    path = os.path.join(_DIR, f'features_{kind}.txt')
    if not os.path.exists(path):
        return default
    selected = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith('#'):
                continue
            selected.append(s)
    return selected if selected else default


def _target_impact_heatmap(
    df: pd.DataFrame,
    targets: list,
    features: list,
    title: str,
    path: str,
    top_n: int = 18,
    cross_target_features: list | None = None,
) -> None:
    """Article-friendly compact correlation heatmap (matches NEST_UMAR /
    HouseZero). ``cross_target_features`` rows are prepended (in caller-supplied
    order, unsorted, no top_n filter) above the impact-sorted regular features;
    a thicker separator and italic violet y-labels mark the cross-target block.
    """
    targets  = [c for c in targets  if c in df.columns]
    features = [c for c in features if c in df.columns]
    cross    = [c for c in (cross_target_features or [])
                if c in df.columns and c not in targets and c not in features]
    if not targets or (not features and not cross):
        return

    all_cols  = list(dict.fromkeys(targets + features + cross))
    sub       = df[all_cols].replace([np.inf, -np.inf], np.nan).dropna()
    corr_full = sub.corr()

    mat_feat = corr_full.loc[features, targets]
    if features:
        order    = mat_feat[targets[0]].abs().sort_values(ascending=False).index.tolist()
        mat_feat = mat_feat.loc[order].iloc[:top_n]

    if cross:
        mat_cross = corr_full.loc[cross, targets]
        mat       = pd.concat([mat_cross, mat_feat], axis=0)
    else:
        mat = mat_feat
    n_cross_rows = len(cross)

    n_feat = len(mat)
    n_tgt  = len(targets)
    if n_feat == 0:
        return

    two_col = n_feat > 12
    if two_col:
        mid    = (n_feat + 1) // 2
        halves = [mat.iloc[:mid], mat.iloc[mid:]]
    else:
        halves = [mat]
    cross_index_set = set(cross)

    panel_w = 1.9 + n_tgt * 0.95
    panel_h = max(len(h) for h in halves) * 0.30 + 1.6
    fig_w   = panel_w * len(halves) + 0.4
    fig_h   = panel_h + 1.2

    fig, axes = plt.subplots(1, len(halves), figsize=(fig_w, fig_h),
                             gridspec_kw={'wspace': 0.2})
    if len(halves) == 1:
        axes = [axes]
    fig.patch.set_facecolor('#F7F8FA')

    im = None
    for k, (ax, half) in enumerate(zip(axes, halves)):
        ax.set_facecolor('#F7F8FA')
        nh = len(half)
        im = ax.imshow(half.values, cmap='RdBu_r', vmin=-1, vmax=1,
                       aspect='auto', interpolation='nearest')
        for x in np.arange(-0.5, n_tgt, 1):
            ax.axvline(x, color='white', linewidth=1.2)
        for y in np.arange(-0.5, nh, 1):
            ax.axhline(y, color='white', linewidth=0.6)
        if k == 0 and n_cross_rows and n_cross_rows < nh:
            ax.axhline(n_cross_rows - 0.5, color='#1a1a2e',
                       linewidth=1.6, zorder=3)
        ax.set_xticks(range(n_tgt))
        ax.set_xticklabels(
            [_TARGET_LABELS.get(c, c.replace('_', ' ')) for c in targets],
            fontsize=9, fontweight='bold', color='#1a1a2e',
        )
        ax.xaxis.set_ticks_position('top')
        ax.xaxis.set_label_position('top')
        ax.tick_params(axis='x', length=0, pad=4)
        ax.set_yticks(range(nh))
        y_labels = [_FEAT_LABELS.get(c, _TARGET_LABELS.get(c, c.replace('_', ' ')).replace('\n', ' '))
                    for c in half.index]
        ax.set_yticklabels(y_labels, fontsize=8, color='#2d2d2d')
        for tick_label, row_name in zip(ax.get_yticklabels(), half.index):
            if row_name in cross_index_set:
                tick_label.set_fontstyle('italic')
                tick_label.set_color('#5a2e8a')
        if k > 0:
            ax.yaxis.tick_right()
        ax.tick_params(axis='y', length=0, pad=3)
        for i in range(nh):
            if i % 2 == 0:
                ax.add_patch(plt.Rectangle(
                    (-0.5, i - 0.5), n_tgt, 1,
                    color='#00000008', zorder=0,
                ))
            for j in range(n_tgt):
                val = half.values[i, j]
                if np.isnan(val):
                    continue
                abs_val    = abs(val)
                text_color = 'white' if abs_val > 0.52 else '#1a1a2e'
                weight     = 'bold'  if abs_val >= 0.30 else 'normal'
                size       = 8       if abs_val >= 0.30 else 7
                ax.text(j, i, f'{val:+.2f}', ha='center', va='center',
                        fontsize=size, color=text_color, fontweight=weight)
        for spine in ax.spines.values():
            spine.set_visible(False)

    cbar = fig.colorbar(im, ax=axes, orientation='horizontal',
                        fraction=0.04, pad=0.08, aspect=40, shrink=0.6)
    cbar.set_label('Pearson r', fontsize=8, labelpad=3)
    cbar.ax.tick_params(labelsize=7)
    cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
    cbar.outline.set_visible(False)

    fig.suptitle(title, fontsize=11, fontweight='bold', y=0.995,
                 color='#1a1a2e')
    _save(fig, path)


def plot_consumption_corr(df: pd.DataFrame, out_dir: str) -> None:
    features = _load_feature_whitelist(_CONSUMPTION_FEATURES, 'consumption')
    for target in _CONSUMPTION_TARGETS:
        label = _TARGET_LABELS.get(target, target).replace('\n', ' ')
        cross = [t for t in _ALL_CONSUMPTION_TARGETS if t != target]
        _target_impact_heatmap(
            df,
            targets=[target],
            features=features,
            cross_target_features=cross,
            title=f'Consumption  --  Feature Impact on {label}',
            path=os.path.join(out_dir, f'consumption_corr_{target}.png'),
        )


# 6. HVAC demand vs outdoor temperature --------------------------------------

def plot_hvac_vs_temp(df: pd.DataFrame, out_dir: str) -> None:
    temp_col = 'temperature_2m' if 'temperature_2m' in df.columns else 'local_air_temp_c'
    if TARGET_THERMAL not in df.columns or temp_col not in df.columns:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sample = df[[TARGET_THERMAL, temp_col]].dropna().sample(
        min(5000, len(df)), random_state=42
    )
    axes[0].scatter(sample[temp_col], sample[TARGET_THERMAL],
                    alpha=0.15, s=5, color=PALETTE['thermal'])
    axes[0].set_xlabel('Outdoor Temperature (degC)')
    axes[0].set_ylabel('HVAC Electric Demand (kW)')
    axes[0].set_title(f'Scatter (sampled, using {temp_col})')

    bins = np.arange(0, 40, 2)
    df2 = df[[TARGET_THERMAL, temp_col]].dropna().copy()
    df2['t_bin'] = pd.cut(df2[temp_col], bins=bins, labels=bins[:-1])
    g = df2.groupby('t_bin', observed=True)[TARGET_THERMAL]
    mn, st = g.mean(), g.std()
    t_centers = mn.index.astype(float)
    axes[1].plot(t_centers, mn, color=PALETTE['thermal'], linewidth=2)
    axes[1].fill_between(t_centers, mn - st, mn + st, alpha=0.25, color=PALETTE['thermal'])
    axes[1].set_xlabel('Outdoor Temperature (degC)')
    axes[1].set_ylabel('HVAC Electric Demand (kW)')
    axes[1].set_title('Binned Mean +/- Std (2 degC bins)')

    fig.suptitle('HVAC Electric Demand vs Outdoor Temperature', fontsize=13)
    plt.tight_layout()
    _save(fig, os.path.join(out_dir, 'hvac_vs_temp.png'))


# 7. Load-duration curves ----------------------------------------------------

def plot_load_duration(df: pd.DataFrame, out_dir: str) -> None:
    specs = [
        (TARGET_THERMAL_HEATING, 'Heating',   PALETTE['heating']),
        (TARGET_THERMAL_COOLING, 'Cooling',   PALETTE['cooling']),
        (TARGET_ELECTRIC,        'Electric', PALETTE['total']),
    ]
    specs = [(c, l, k) for c, l, k in specs if c in df.columns]
    if not specs:
        return

    fig, axes = plt.subplots(1, len(specs), figsize=(4 * len(specs) + 1, 5))
    if len(specs) == 1:
        axes = [axes]

    for ax, (col, title, color) in zip(axes, specs):
        vals = df[col].dropna().sort_values(ascending=False).values
        pct  = np.linspace(0, 100, len(vals))
        ax.plot(pct, vals, color=color, linewidth=1.2)
        ax.fill_between(pct, vals, alpha=0.15, color=color)
        ax.set_xlabel('% of Hours Exceeded')
        ax.set_ylabel('kW')
        ax.set_title(title)
        for p, lbl in [(10, 'P10'), (50, 'P50'), (90, 'P90')]:
            v = np.percentile(vals, 100 - p)
            ax.axhline(v, color='grey', linewidth=0.7, linestyle=':')
            ax.text(p + 1, v * 1.02, f'{lbl}={v:.2f}', fontsize=7, color='grey')

    fig.suptitle('Load-Duration Curves', fontsize=13)
    plt.tight_layout()
    _save(fig, os.path.join(out_dir, 'load_duration.png'))


# 8. Missing-data heatmap ----------------------------------------------------

def plot_missing_data(df: pd.DataFrame, out_dir: str) -> None:
    key_cols = [
        TARGET_THERMAL_HEATING, TARGET_THERMAL_COOLING, TARGET_ELECTRIC,
        'hvac_n_kw', 'hvac_s_kw', 'mels_n_kw', 'mels_s_kw', 'lig_s_kw',
        'temperature_2m', 'relative_humidity_2m', 'cloud_cover',
        'shortwave_radiation', 'global_tilted_irradiance', 'sunshine_duration',
        'local_air_temp_c', 'local_solar_rad_wm2',
        'zone_cooling_sp_c', 'zone_heating_sp_c', 'zone_co2_mean_ppm',
        'occupant_count', 'wifi_device_count',
        'pm25', 'pm10', 'o3', 'no2',
    ]
    present = [c for c in key_cols if c in df.columns]
    df2 = df[present].copy()
    if df2.empty:
        return

    missing_rate = df2.resample('ME').apply(lambda g: g.isna().mean() * 100)
    missing_rate.index = missing_rate.index.strftime('%Y-%m')
    if missing_rate.empty:
        return

    fig, ax = plt.subplots(figsize=(max(10, len(present) * 0.6), 6))
    im = ax.imshow(missing_rate.values.T, aspect='auto',
                   cmap='Reds', vmin=0, vmax=100)
    ax.set_xticks(range(len(missing_rate.index)))
    ax.set_xticklabels([str(p) for p in missing_rate.index], rotation=45, ha='right', fontsize=7)
    ax.set_yticks(range(len(present)))
    ax.set_yticklabels(present, fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.02, label='% Missing')
    ax.set_title('Missing Data Rate by Month (key columns)', fontsize=12)
    plt.tight_layout()
    _save(fig, os.path.join(out_dir, 'missing_data.png'))


# 9. Monthly energy balance --------------------------------------------------

def plot_energy_balance(df: pd.DataFrame, out_dir: str) -> None:
    cols = [c for c in [
        TARGET_THERMAL_HEATING, TARGET_THERMAL_COOLING, TARGET_ELECTRIC,
    ] if c in df.columns]
    if not cols:
        return

    monthly = df[cols].resample('ME').sum() / 1000  # kWh -> MWh
    months  = [d.strftime('%b %Y') for d in monthly.index]
    x       = np.arange(len(months))
    width   = 0.25

    fig, ax = plt.subplots(figsize=(16, 5))
    offsets = np.linspace(-width * (len(cols) - 1) / 2, width * (len(cols) - 1) / 2, len(cols))
    for col, offset in zip(cols, offsets):
        color = {
            TARGET_THERMAL_HEATING: PALETTE['heating'],
            TARGET_THERMAL_COOLING: PALETTE['cooling'],
            TARGET_ELECTRIC:        PALETTE['total'],
        }.get(col, 'grey')
        label = {
            TARGET_THERMAL_HEATING: 'Heating',
            TARGET_THERMAL_COOLING: 'Cooling',
            TARGET_ELECTRIC:        'Electric',
        }.get(col, col)
        ax.bar(x + offset, monthly[col], width=width * 0.9,
               color=color, alpha=0.85, label=label)

    ax.set_xticks(x)
    ax.set_xticklabels(months, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('Energy (MWh)')
    ax.legend(fontsize=9)
    ax.set_title('Monthly Energy Balance', fontsize=13)
    plt.tight_layout()
    _save(fig, os.path.join(out_dir, 'energy_balance.png'))


# 10. Occupancy pattern heatmap (occupant / WiFi counts) ---------------------

def plot_occupancy_pattern(df: pd.DataFrame, out_dir: str) -> None:
    specs = [
        ('occupant_count',    'Occupant Count (camera)', 'YlOrRd'),
        ('wifi_device_count', 'WiFi Device Count',       'viridis'),
    ]
    specs = [(c, l, cm) for c, l, cm in specs if c in df.columns]
    if not specs:
        return

    fig, axes = plt.subplots(1, len(specs), figsize=(7 * len(specs), 6))
    if len(specs) == 1:
        axes = [axes]

    for ax, (col, title, cmap) in zip(axes, specs):
        df2 = df[[col]].copy()
        df2['doy']  = df.index.dayofyear
        df2['hour'] = df.index.hour
        pivot = df2.pivot_table(values=col, index='doy', columns='hour', aggfunc='mean')
        vmax = np.nanpercentile(pivot.values, 99) if np.isfinite(pivot.values).any() else None

        im = ax.imshow(pivot.values, aspect='auto', cmap=cmap,
                       origin='lower', interpolation='nearest', vmin=0, vmax=vmax)
        plt.colorbar(im, ax=ax, fraction=0.03, label='Count')
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Day of Year')
        ax.set_xticks(range(0, 24, 2))
        ax.set_xticklabels(range(0, 24, 2))
        month_starts = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
        month_names  = ['Jan','Feb','Mar','Apr','May','Jun',
                        'Jul','Aug','Sep','Oct','Nov','Dec']
        ax.set_yticks(month_starts)
        ax.set_yticklabels(month_names, fontsize=8)
        ax.set_title(title, fontsize=11)

    fig.suptitle('Occupancy Pattern -- Day-of-Year x Hour Heatmap', fontsize=13)
    plt.tight_layout()
    _save(fig, os.path.join(out_dir, 'occupancy_pattern.png'))


# 11. COVID impact (Building 59-specific regime comparison) ------------------

def plot_covid_impact(df: pd.DataFrame, out_dir: str) -> None:
    cols = [c for c in [TARGET_ELECTRIC, TARGET_THERMAL] if c in df.columns]
    if not cols:
        return
    fig, axes = plt.subplots(1, len(cols), figsize=(6 * len(cols), 4.5))
    if len(cols) == 1:
        axes = [axes]
    label = {TARGET_ELECTRIC: 'Electric', TARGET_THERMAL: 'HVAC Electric'}
    for ax, col in zip(axes, cols):
        for yr, color in [(2019, '#5B8DB8'), (2020, '#E05C5C')]:
            sub = df[df.index.year == yr]
            if sub.empty:
                continue
            g = sub.groupby(sub.index.month)[col].mean()
            ax.plot(g.index, g.values, '-o', color=color, label=str(yr))
        ax.set_title(label.get(col, col)); ax.set_xlabel('Month'); ax.set_ylabel('kW')
        ax.set_xticks(range(1, 13)); ax.legend(title='Year', fontsize=8)
    fig.suptitle('Building 59 -- Pre-pandemic (2019) vs COVID (2020)', fontsize=13)
    plt.tight_layout()
    _save(fig, os.path.join(out_dir, 'covid_impact.png'))


# Main entry point ------------------------------------------------------------

def generate_all(csv_path: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    print(f'[viz] Loading {csv_path} ...')
    df = pd.read_csv(csv_path, parse_dates=['date'], index_col='date')
    print(f'  {len(df)} rows x {len(df.columns)} columns  '
          f'({df.index[0].date()} -> {df.index[-1].date()})')

    print('[viz] Generating charts ...')
    plot_targets_overview(df, out_dir)
    plot_thermal_overview(df, out_dir)
    plot_monthly_profiles(df, out_dir)
    plot_daily_profiles(df, out_dir)
    plot_seasonal_daily(df, out_dir)
    plot_consumption_corr(df, out_dir)
    plot_hvac_vs_temp(df, out_dir)
    plot_load_duration(df, out_dir)
    plot_missing_data(df, out_dir)
    plot_energy_balance(df, out_dir)
    plot_occupancy_pattern(df, out_dir)
    plot_covid_impact(df, out_dir)
    print(f'[viz] All charts saved to {out_dir}')


def _parse_args():
    p = argparse.ArgumentParser(description='Generate LBNL Building 59 dataset visualizations.')
    base = _DIR
    p.add_argument('--csv', default=os.path.join(base, 'processed', 'data_hourly.csv'))
    p.add_argument('--out', default=os.path.join(base, 'plots'))
    return p.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    generate_all(args.csv, args.out)
