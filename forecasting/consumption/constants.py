"""
Column definitions and hyper-parameter defaults for consumption forecasting.

Targets
-------
  thermal_heating_kw — space heating demand (Oct-15 → Apr-15, kW)
  thermal_cooling_kw — space cooling demand (Apr-15 → Oct-15, kW)
  electric_demand_kw — domestic electricity demand (kW)

  thermal_demand_kw  — combined heating + cooling (kept for diagnostics)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from forecasting.constants import (
    SEQUENCE_LENGTH as _SEQUENCE_LENGTH, FORECAST_HORIZON,
    TARGET_THERMAL, TARGET_THERMAL_HEATING, TARGET_THERMAL_COOLING,
    TARGET_DHW, TARGET_ELECTRIC,
    DEFAULT_LSTM_THERMAL_WEIGHT, DEFAULT_GBDT_THERMAL_WEIGHT,
    DEFAULT_LSTM_ELECTRIC_WEIGHT, DEFAULT_GBDT_ELECTRIC_WEIGHT,
)

SEQUENCE_LENGTH = 336   # 14 days of hourly history

# ---------------------------------------------------------------------------
# Target columns (order matters for LSTM-CNN scaler)
# ---------------------------------------------------------------------------
TARGET_COLUMNS        = [TARGET_THERMAL_HEATING, TARGET_THERMAL_COOLING, TARGET_DHW, TARGET_ELECTRIC]
TARGET_COLUMNS_LEGACY = [TARGET_THERMAL, TARGET_ELECTRIC]   # combined, for diagnostics

# ---------------------------------------------------------------------------
# LSTM-CNN input columns  (targets first, then features)
# ---------------------------------------------------------------------------
LSTM_COLUMNS = [
    TARGET_THERMAL_HEATING,
    TARGET_THERMAL_COOLING,
    TARGET_DHW,
    TARGET_ELECTRIC,
    # temporal — raw integers kept for human readability / embedding layers
    'hour', 'month', 'season', 'day_of_week', 'is_business', 'is_heating_season',
    # cyclic encodings: prevent Dec->Jan / 23:00->00:00 discontinuities in GBDT
    'hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'dow_sin', 'dow_cos',
    # open-meteo weather
    'temperature_2m', 'relative_humidity_2m', 'dew_point_2m',
    'apparent_temperature', 'rain_precipitation', 'snowfall_precipitation',
    'pressure_msl', 'surface_pressure', 'cloud_cover',
    'et0_fao_evapotranspiration', 'vapour_pressure_deficit',
    'wind_speed_10m', 'wind_speed_100m',
    'wind_direction_10m', 'wind_direction_100m', 'wind_gusts_10m',
    'boundary_layer_height', 'sunshine_duration',
    'direct_radiation', 'diffuse_radiation',
    'global_tilted_irradiance', 'terrestrial_radiation', 'is_day',
    # AQI
    'pm25', 'pm10', 'o3', 'no2', 'so2', 'co',
]

# ---------------------------------------------------------------------------
# GBDT feature columns  (all except the targets being predicted)
# ---------------------------------------------------------------------------
GBDT_FEATURE_COLUMNS = [c for c in LSTM_COLUMNS if c not in TARGET_COLUMNS]

# ---------------------------------------------------------------------------
# Optional activity-signal features
# ---------------------------------------------------------------------------
# Per-target "is the equipment running right now" sensors that aren't present
# in every dataset (HouseZero doesn't ship dhw_volume; NEST_UMAR does). They
# are auto-appended to the resolved feature list at training time IF the CSV
# contains them, and silently skipped otherwise — so listing them here is
# safe across datasets.
#
# Added because zero-heavy targets (notably NEST_UMAR DHW) were over-predicted
# 5× because the model had no direct signal for "DHW system is currently off".
OPTIONAL_LSTM_COLUMNS = (
    'dhw_volume_m3h',   # NEST_UMAR DHW circulation flow (m^3/h)
    'valve_open_frac',  # NEST_UMAR aggregate valve-open fraction [0, 1]
)
OPTIONAL_GBDT_FEATURE_COLUMNS = OPTIONAL_LSTM_COLUMNS
# The value of an activity signal in the hour being forecast is not known when
# a day-ahead forecast is issued. The sequence models only see its past values
# (their input is the context window). GBDT reads one row per forecast hour, so
# it gets the signal at these lags instead of the same-hour value, and never a
# value from inside the forecast window (see forecast_gbdt_iterative).
# Checkpoints trained before 21 Nov 2026 used the same-hour value and still
# load: each model reads the columns named in its own feature scaler.
ACTIVITY_LAG_STEPS = (24, 168)   # 1 day, 1 week

# ---------------------------------------------------------------------------
# Lag offsets used to replace contemporaneous cross-target features in GBDT
# ---------------------------------------------------------------------------
LAG_STEPS = (1, 24, 168)   # 1 h, 1 day, 1 week
