"""
Global constants for the HouseZero forecasting module.

Site: HouseZero®, Harvard University, Cambridge, MA, USA
Dataset: Jun 2022 → May 2024 (2 years, hourly)
Source: Han et al., Scientific Data 11:938 (2024)
"""

# ---------------------------------------------------------------------------
# Site configuration
# ---------------------------------------------------------------------------
SITE_TIMEZONE   = "America/New_York"
SITE_LATITUDE   = 42.355007
SITE_LONGITUDE  = -71.12906

# ---------------------------------------------------------------------------
# Date/time formats
# ---------------------------------------------------------------------------
INITIAL_FORMAT  = '%Y-%m-%d %H:%M %z'
WEATHER_FORMAT  = '%Y-%m-%dT%H:00'
AQI_FORMAT      = '%Y/%m/%d'

# Dataset window (inclusive)
DATASET_START = '2022-06-01'
DATASET_END   = '2024-05-31'

# ---------------------------------------------------------------------------
# Forecasting window
# ---------------------------------------------------------------------------
SEQUENCE_LENGTH  = 336    # 14 days of hourly history fed to LSTM-CNN
FORECAST_HORIZON = 72     # 3-day ahead forecast
RETRAIN_WINDOW   = 2000   # hours used for adaptive fine-tuning (~83 days)
MIN_DATA_LENGTH  = SEQUENCE_LENGTH + FORECAST_HORIZON

# ---------------------------------------------------------------------------
# Heating season  (Oct-15 → Apr-15 for Cambridge, MA)
# ---------------------------------------------------------------------------
HEATING_START_MONTH = 10
HEATING_START_DAY   = 15
HEATING_END_MONTH   = 4
HEATING_END_DAY     = 15

# ---------------------------------------------------------------------------
# Unit conversion
# ---------------------------------------------------------------------------
KBTU_H_TO_KW  = 0.29307107   # kBTU/h  →  kW (thermal)
BTU_TO_KWH    = 2.93071e-4    # BTU     →  kWh
F_TO_C_OFFSET = 32.0
F_TO_C_SCALE  = 5.0 / 9.0

# ---------------------------------------------------------------------------
# Solar Hot Water analytical model defaults
# (used in generation/solar_thermal.py)
# ---------------------------------------------------------------------------
SHW_COLLECTOR_AREA_M2  = 4.0    # m²  (typical 2-panel residential, recalibrate if known)
SHW_EFFICIENCY_ETA0    = 0.70   # peak optical efficiency
SHW_HEAT_LOSS_COEFF    = 3.5    # W/(m²·K)  — first-order heat loss coefficient
SHW_TARGET_TEMP_C      = 60.0   # desired output temperature (°C)

# ---------------------------------------------------------------------------
# Target column names
# ---------------------------------------------------------------------------
TARGET_THERMAL          = 'thermal_demand_kw'    # space conditioning (TABS slab, kW)
TARGET_THERMAL_HEATING  = 'thermal_heating_kw'   # space heating only    (Oct-15 -> Apr-15)
TARGET_THERMAL_COOLING  = 'thermal_cooling_kw'   # space cooling only    (Apr-15 -> Oct-15)
TARGET_DHW              = 'dhw_kw'               # domestic hot water electric demand (EHW + pump)
TARGET_ELECTRIC         = 'electric_demand_kw'   # domestic electricity demand  (kW)
TARGET_PV               = 'pv_electric_kw'       # PV generation (kW)
TARGET_SOLAR_THERMAL    = 'solar_thermal_kw'      # SHW collector thermal output (analytical)

# ---------------------------------------------------------------------------
# Default ensemble weights (consumption)
# ---------------------------------------------------------------------------
DEFAULT_LSTM_THERMAL_WEIGHT  = 0.3
DEFAULT_GBDT_THERMAL_WEIGHT  = 0.7
DEFAULT_LSTM_ELECTRIC_WEIGHT = 0.3
DEFAULT_GBDT_ELECTRIC_WEIGHT = 0.7

# ---------------------------------------------------------------------------
# Default ensemble weights (generation — PV only)
# ---------------------------------------------------------------------------
DEFAULT_LSTM_PV_WEIGHT = 0.35
DEFAULT_GBDT_PV_WEIGHT = 0.65

# ---------------------------------------------------------------------------
# Adaptive forecasting thresholds
# ---------------------------------------------------------------------------
DEFAULT_MAE_THRESHOLD        = 0.08
DEFAULT_RMSE_THRESHOLD       = 0.12
DEFAULT_MASE_THRESHOLD       = 1.2    # retrain when model is >20 % worse than weekly naive
# A test split of ~3500 windows × eval_freq=10 = 350 candidate retrain
# checks per run, capped at max_retrains=100 — far more than necessary
# and noisy. Bumping to eval_freq=200, max_retrains=20 gives ~18 checks
# per run with plenty of headroom on the cap; the analysis of
# runs/20260520-081105 (where eval_freq=1000 left only 3 checks total
# and electric MAE still dropped 34%) shows the loop benefits from
# *more* checks than the original defaults provided, but the legacy
# cadence over-fires under realistic test windows.
DEFAULT_EVALUATION_FREQUENCY = 200
DEFAULT_MAX_RETRAINS         = 20
