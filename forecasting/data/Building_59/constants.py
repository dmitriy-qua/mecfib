"""
Site-specific constants for the LBNL Building 59 dataset.

Source publication:
  N. Luo, Z. Wang, D. Blum, C. Weyandt, N. Bourassa, M. A. Piette, T. Hong,
  "A three-year dataset supporting research on building energy management and
  occupancy analytics," Scientific Data 9:156 (2022).
  DOI: 10.1038/s41597-022-01257-x.

Site:    Building 59, Lawrence Berkeley National Laboratory campus,
         Berkeley, California, USA. A LEED-certified office building
         constructed in 2015. The dataset covers two instrumented office
         floors (3rd and 4th, each ~2,325 m^2) served by packaged rooftop
         units (RTUs) with VAV terminal boxes.
Period:  2018-01-01 -> 2020-12-31 (3 years).
         Two pre-pandemic years (2018-2019) plus 2020, which captures the
         COVID-19 shelter-in-place occupancy collapse (late March 2020 on).

Sampling (raw):
  * Electrical end-use meters (ele.csv) ............ 15 min, kW
  * On-site weather station (site_weather.csv) ..... 15 min, SI units
  * Zone CO2 (zone_co2.csv) ........................ ~1 min, ppm
  * Zone heating/cooling setpoints (zone_temp_sp_*) . 5 min, degF
  * Camera occupant counts (occ.csv) ............... 1 min (ends 2019-02-21)
  * WiFi device counts (wifi.csv) .................. 10 min
  * open-meteo reanalysis (weather.csv) ............ hourly, SI units
  * Air-quality index (aqi.csv) .................... daily (limited coverage)

Like the other per-dataset modules under ``forecasting/data/*/constants.py``,
this file is auto-discovered by ``apply_dataset_thresholds`` when the input
CSV lives under ``forecasting/data/Building_59/``. Adding a constant here is
enough to propagate it to runs -- no orchestrator wiring required.
"""

# ---------------------------------------------------------------------------
# Site configuration
# ---------------------------------------------------------------------------
# Berkeley observes US Pacific time WITH daylight saving. The building BMS,
# electrical meters and on-site weather station all log in building-local
# (Pacific) wall-clock time, so that is the canonical clock for this dataset.
# The open-meteo weather export carries no timezone metadata block, so
# prepare.py auto-aligns it to the building clock by cross-correlating the
# on-site pyranometer against open-meteo shortwave radiation (see
# _auto_align_weather).
SITE_TIMEZONE   = 'America/Los_Angeles'
SITE_LATITUDE   = 37.8732           # Berkeley Lab campus
SITE_LONGITUDE  = -122.2606
SITE_ELEVATION  = 150.0             # metres above sea level (approx)

# Dataset window (inclusive)
DATASET_START = '2018-01-01'
DATASET_END   = '2020-12-31'

# ---------------------------------------------------------------------------
# Chronological train/test boundary
# ---------------------------------------------------------------------------
# 1.5 years train (2018-01 -> 2019-06) / 1.5 years test (2019-07 -> 2020-12),
# a 50/50 chronological split. Training still stops well before the March-2020
# shelter-in-place, so the COVID-19 drift (occupancy collapse, HVAC schedule
# changes, MPC field tests) lands inside the test window and remains a strong
# real-world drift scenario for the adaptive-forecasting harness -- mirroring
# the HEEW design rationale.
TRAIN_END = '2019-06-30 23:00'

# ---------------------------------------------------------------------------
# Target mapping (Building 59 is an all-electric office; see prepare.py)
# ---------------------------------------------------------------------------
# Building 59 meters electricity by END USE, not thermal output. There is no
# heat-meter and no COP measurement, so we cannot recover thermal kW. The
# closest analogue to the framework's "thermal" target is the HVAC electric
# input (RTU compressors, fans, electric reheat), which we map as follows:
#
#   electric_demand_kw  = mels_N + mels_S + lig_S      (plug loads + lighting)
#   thermal_demand_kw   = hvac_N + hvac_S              (HVAC electric input)
#   thermal_heating_kw  = HVAC electric, heating season only (else 0)
#   thermal_cooling_kw  = HVAC electric, cooling season only (else 0)
#   dhw_kw              = 0                            (no DHW sub-metering)
#
# IMPORTANT: thermal_* columns are HVAC ELECTRIC demand (kW_electric), NOT
# thermal output (kW_thermal). The heating/cooling split is calendar-based
# (forecasting.utils.is_heating_season, Oct-15 -> Apr-15) because the single
# combined HVAC meter cannot be physically separated into the two modes.
HVAC_USES   = ('hvac_N', 'hvac_S')
PLUG_USES   = ('mels_N', 'mels_S')
LIGHT_USES  = ('lig_S',)            # only South-floor lighting meter is present

# ---------------------------------------------------------------------------
# Unit conversions
# ---------------------------------------------------------------------------
F_TO_C_OFFSET = 32.0
F_TO_C_SCALE  = 5.0 / 9.0

# ---------------------------------------------------------------------------
# Documented operational-change windows (paper Table 2)
# ---------------------------------------------------------------------------
# Listed for reference / optional regime annotation. Not auto-applied as
# outages; the 2020 windows fall inside the test split on purpose.
OPERATIONAL_CHANGES = [
    ('2018-11-08', '2018-11-21', 'Camp Fire wildfire smoke - OA dampers closed'),
    ('2020-03-23', '2020-12-31', 'COVID-19 shelter-in-place - occupancy collapse'),
    ('2020-08-19', '2020-09-09', 'Wildfire smoke - OA dampers closed'),
    ('2020-10-12', '2020-11-16', 'MPC field test on RTU HVAC operation'),
]

# ---------------------------------------------------------------------------
# Adaptive-forecasting retraining thresholds
# ---------------------------------------------------------------------------
# Magnitudes in the test split (hourly means, kW):
#     plug+light electric ~12 kW   HVAC electric ~25 kW (combined two floors)
# The 2020 COVID collapse drops plug/light loads sharply, so thresholds sit at
# roughly 1.5x a competent-baseline MAE. Tune empirically after first run.
ADAPTIVE_MAE_THRESHOLD_KW  = 6.0
ADAPTIVE_RMSE_THRESHOLD_KW = 9.0
ADAPTIVE_MASE_THRESHOLD    = 1.5

# ---------------------------------------------------------------------------
# Data-driven inactivity gate (mirrors HouseZero / KSRP / UMAR convention)
# ---------------------------------------------------------------------------
# Catches shoulder-season periods where the HVAC mode is off but the
# is_heating_season calendar flag hasn't flipped yet. Berkeley is a mild,
# cooling-light climate, so heating runs in winter and the (small) cooling
# load is summer-only; a 24 h lookback with a 1 kW floor avoids tripping on
# normal operation.
INACTIVITY_GATES = {
    'thermal_heating_kw': {'lookback_hours': 24, 'threshold_kw': 1.0},
    'thermal_cooling_kw': {'lookback_hours': 24, 'threshold_kw': 1.0},
}

# ---------------------------------------------------------------------------
# Per-target training hyperparameters
# ---------------------------------------------------------------------------
# Schema mirrors NEST_UMAR / KSRP_Campus / HEEW. The dhw_kw target is
# identically zero for Building 59 (no DHW meter); keep it out of any special
# handling. Light defaults -- the cleaned LBNL signal is high quality.
TRAINING_HYPERPARAMS = {
    'lstm_cnn': {},
    'gbdt': {},
}
