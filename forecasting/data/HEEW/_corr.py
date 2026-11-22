"""
HEEW / ASU Tempe Campus — target-impact correlation heatmaps.

Mirrors the KSRP Campus convention so the consumption + generation
correlation plots have the same look across all sites: features as rows
(sorted by |Pearson r|, top-N retained), targets as columns, ``RdBu_r``
diverging colormap with vmin/vmax in [-1, +1], numeric Pearson values
printed in each cell.  Cross-target rows (other consumption heads treated
as features) are prepended above the impact-sorted block with italic purple
labels and a thicker dark separator line.

Split out of ``visualize_data.py`` to keep both files well under the
cowork workspace's single-write byte budget.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from forecasting.constants import TARGET_ELECTRIC, TARGET_PV


# ---------------------------------------------------------------------------
# Default whitelists (overridable via features_<kind>.txt next to this module)
# ---------------------------------------------------------------------------
CONSUMPTION_FEATURES = [
    'is_heating_season', 'month_cos', 'hour_cos', 'is_business', 'is_day',
    'temperature_2m', 'apparent_temperature', 'dew_point_2m', 'local_air_temp_c',
    'local_dew_point_c',
    'relative_humidity_2m', 'local_rh_pct', 'cloud_cover', 'solar_obstruction_pct',
    'sunshine_duration', 'shortwave_radiation', 'direct_radiation',
    'diffuse_radiation', 'global_tilted_irradiance',
    'wind_speed_10m', 'wind_gusts_10m', 'local_wind_speed_ms', 'pressure_msl',
    'surface_pressure', 'local_pressure_hpa', 'vapour_pressure_deficit',
    'et0_fao_evapotranspiration', 'boundary_layer_height',
    'rain_precipitation', 'snowfall_precipitation', 'local_precip_mm',
    'pm25', 'pm10', 'o3',
]

GENERATION_FEATURES = [
    'hour_cos', 'month_cos', 'is_day', 'is_heating_season',
    'shortwave_radiation', 'direct_radiation', 'diffuse_radiation',
    'global_tilted_irradiance', 'direct_normal_irradiance', 'terrestrial_radiation',
    'sunshine_duration', 'solar_obstruction_pct', 'cloud_cover',
    'temperature_2m', 'apparent_temperature', 'dew_point_2m', 'local_air_temp_c',
    'local_dew_point_c',
    'relative_humidity_2m', 'local_rh_pct', 'pressure_msl',
    'surface_pressure', 'local_pressure_hpa', 'vapour_pressure_deficit',
    'et0_fao_evapotranspiration', 'boundary_layer_height',
    'rain_precipitation', 'snowfall_precipitation', 'local_precip_mm',
    'wind_speed_10m', 'local_wind_speed_ms',
    'pm25', 'pm10', 'o3',
]

TARGET_LABELS = {
    'thermal_heating_kw': 'Thermal\nHeating',
    'thermal_cooling_kw': 'Thermal\nCooling',
    TARGET_ELECTRIC:      'Electric\nDemand',
    TARGET_PV:            'PV\nGeneration',
}

FEAT_LABELS = {
    'is_heating_season':         'Heating season',
    'is_business':               'Business hours',
    'is_day':                    'Daytime',
    'hour':                      'Hour of day',
    'month':                     'Month',
    'day_of_week':               'Day of week',
    'season':                    'Season',
    'hour_sin':                  'Hour of day (sin)',
    'hour_cos':                  'Hour of day (cyclic)',
    'month_sin':                 'Month (sin)',
    'month_cos':                 'Month (cyclic)',
    'dow_sin':                   'Day of week (sin)',
    'dow_cos':                   'Day of week (cos)',
    'temperature_2m':            'Air temperature (open-meteo)',
    'apparent_temperature':      'Apparent temperature',
    'dew_point_2m':              'Dew point',
    'local_air_temp_c':          'Air temperature (on-site)',
    'local_dew_point_c':         'Dew point (on-site)',
    'relative_humidity_2m':      'Humidity (open-meteo)',
    'local_rh_pct':              'Humidity (on-site)',
    'cloud_cover':               'Cloud cover',
    'solar_obstruction_pct':     'Solar obstruction %',
    'sunshine_duration':         'Sunshine duration',
    'shortwave_radiation':       'Shortwave radiation',
    'direct_radiation':          'Direct radiation',
    'diffuse_radiation':         'Diffuse radiation',
    'global_tilted_irradiance':  'Global tilted irradiance',
    'direct_normal_irradiance':  'Direct-normal irradiance',
    'terrestrial_radiation':     'Terrestrial radiation',
    'wind_speed_10m':            'Wind speed (10 m)',
    'wind_speed_100m':           'Wind speed (100 m)',
    'wind_direction_10m':        'Wind direction (10 m)',
    'wind_direction_100m':       'Wind direction (100 m)',
    'wind_gusts_10m':            'Wind gusts (10 m)',
    'local_wind_speed_ms':       'Wind speed (on-site)',
    'local_wind_gust_ms':        'Wind gusts (on-site)',
    'pressure_msl':              'Pressure (MSL)',
    'surface_pressure':          'Surface pressure',
    'local_pressure_hpa':        'Pressure (on-site)',
    'vapour_pressure_deficit':   'Vapour pressure deficit',
    'et0_fao_evapotranspiration':'Evapotranspiration (ET0)',
    'boundary_layer_height':     'Boundary layer height',
    'rain_precipitation':        'Rainfall',
    'snowfall_precipitation':    'Snowfall',
    'local_precip_mm':           'Precipitation (on-site)',
    'pm25':                      'PM2.5',
    'pm10':                      'PM10',
    'o3':                        'Ozone',
    'no2':                       'NO2',
    'so2':                       'SO2',
    'co':                        'CO',
    'emission_ton':              'GHG emission (ton CO₂-eq)',
}

ALL_CONSUMPTION_TARGETS = [
    'thermal_heating_kw', 'thermal_cooling_kw', TARGET_ELECTRIC,
]


def load_feature_whitelist(this_dir: str, kind: str, default: list) -> list:
    """Read features_<kind>.txt next to this module; fall back to ``default``."""
    path = os.path.join(this_dir, f'features_{kind}.txt')
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


def _save(fig, path):
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {path}')


def target_impact_heatmap(df, targets, features, title, path,
                          top_n=18, cross_target_features=None):
    """Article-friendly compact heatmap.

    Rows = features sorted by |Pearson r|, columns = targets.  Top-N
    retained.  When ``cross_target_features`` is supplied those rows are
    prepended above the impact-sorted block in caller-supplied order with
    italic purple labels and a thicker dark separator line.
    """
    targets  = [c for c in targets  if c in df.columns]
    features = [c for c in features if c in df.columns]
    cross    = [c for c in (cross_target_features or [])
                if c in df.columns and c not in targets and c not in features]
    if not targets or (not features and not cross):
        return

    all_cols  = list(dict.fromkeys(targets + features + cross))
    sub       = df[all_cols].replace([np.inf, -np.inf], np.nan).dropna()
    if sub.empty:
        return
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
            [TARGET_LABELS.get(c, c.replace('_', ' ')) for c in targets],
            fontsize=9, fontweight='bold', color='#1a1a2e',
        )
        ax.xaxis.set_ticks_position('top')
        ax.xaxis.set_label_position('top')
        ax.tick_params(axis='x', length=0, pad=4)
        ax.set_yticks(range(nh))
        y_labels = [
            FEAT_LABELS.get(c, TARGET_LABELS.get(c, c.replace('_', ' ')).replace('\n', ' '))
            for c in half.index
        ]
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


def plot_consumption_corr(df, out_dir, this_dir):
    """Per-consumption-target heatmap with cross-target rows."""
    features = load_feature_whitelist(this_dir, 'consumption', CONSUMPTION_FEATURES)
    for target in ('thermal_heating_kw', 'thermal_cooling_kw', TARGET_ELECTRIC):
        if target not in df.columns:
            continue
        label = TARGET_LABELS.get(target, target).replace('\n', ' ')
        cross = [t for t in ALL_CONSUMPTION_TARGETS if t != target]
        target_impact_heatmap(
            df,
            targets=[target],
            features=features,
            cross_target_features=cross,
            title=f'Consumption  —  Feature Impact on {label}',
            path=os.path.join(out_dir, f'consumption_corr_{target}.png'),
        )


def plot_generation_corr(df, out_dir, this_dir):
    if TARGET_PV not in df.columns:
        return
    features = load_feature_whitelist(this_dir, 'generation', GENERATION_FEATURES)
    target_impact_heatmap(
        df,
        targets=[TARGET_PV],
        features=features,
        title='Generation  —  Feature Impact on PV Electric Output',
        path=os.path.join(out_dir, 'generation_corr_pv.png'),
    )
