"""
Site-specific constants for the HEEW (Hierarchical Energy, Emissions, and
Weather) dataset.

Source publication:
  H. Dong, J. Zhu, C.-Y. Chung, Z. Liang, H. Yang, X. Wen,
  "A hierarchical dataset on multiple energy consumption and PV generation
  with emissions and weather information," Scientific Data 12:1727 (2025).
  DOI: 10.1038/s41597-025-06010-8.

Period:  2014-01-01 -> 2022-12-31 (9 years, hourly).
Site:    Arizona State University - Tempe Campus (community CN003)
         33.427 N, -111.916 E, ~361 m elevation, Phoenix MST (UTC-7, no DST).

Equipment / coverage at CN003 (per Table 1 of the paper):
  * Electricity (E) and Heat (H) channels are available for ALL buildings in
    the community aggregation that supplies CN003_energy.csv.
  * Cooling (C) is provided by ASU central plants via chilled water (Tons).
  * PV (PV) generation is available at the community aggregation level.
  * GHG emissions (GE) are derived (paper Sec. 'Refined data estimation') from
    energy consumption and PV offset, NOT a direct meter reading.

The CN003 file is one of four aggregated communities (CN001 Downtown, CN002
Polytechnic, CN003 Tempe, CN004 West Valley). We process the Tempe Campus
because (a) it is the largest community by load magnitude, (b) it has the
most stable cooling-dominated regime suited for forecasting research, and
(c) it has complete coverage (no missing intervals per Table 4 of the paper:
range 2014-22, MR = 0, AR = 0 for CN003).

This dataset module produces a single community-aggregated hourly time
series. The raw data is already cleaned through the paper's two-stage scheme
(TCCA + CRAC), so no additional outlier handling is performed here.

Like the other per-dataset modules under ``forecasting/data/*/constants.py``,
this file is auto-discovered by ``apply_dataset_thresholds`` when the input
CSV lives under ``forecasting/data/HEEW/``.
"""

# ---------------------------------------------------------------------------
# Site configuration
# ---------------------------------------------------------------------------
# Arizona does not observe daylight-saving time; America/Phoenix is a fixed
# UTC-7 (= MST) clock year-round. The open-meteo CSV ships with
# 'America/Los_Angeles' as the timezone label but a utc_offset_seconds of
# -25200 (= -7 h fixed), so the values are physically aligned with Phoenix
# local time and we treat them as such (no DST re-anchoring needed).
SITE_TIMEZONE   = 'America/Phoenix'
SITE_LATITUDE   = 33.427063
SITE_LONGITUDE  = -111.91594
SITE_ELEVATION  = 361.0          # metres above sea level

# ---------------------------------------------------------------------------
# Dataset window (inclusive)
# ---------------------------------------------------------------------------
DATASET_START = '2014-01-01'
DATASET_END   = '2022-12-31'

# Chronological train/test boundary.
# 5 years train (2014-2018) / 4 years test (2019-2022) ~= 56/44 split.
# The longer test window deliberately straddles the full ASU regime shift:
# 2019 is a normal pre-COVID year, 2020 captures the COVID occupancy collapse,
# and 2021-2022 the recovery toward baseline. Training stops before any of it,
# so the test period is a strong multi-year drift scenario for the adaptive
# forecasting harness.
TRAIN_END     = '2018-12-31 23:00'

# HEEW data has been cleaned (TCCA + CRAC) and has no documented outage
# windows per the paper (Table 4 reports 0% missing for CN003). Leave the
# list empty for parity with other dataset modules.
OUTAGE_WINDOWS = []

# ---------------------------------------------------------------------------
# Equipment / generation operating windows
# ---------------------------------------------------------------------------
# CN003 PV is operational across the full record. The paper does not give a
# precise commissioning date for the rooftop arrays; we treat the full
# window as in-service and rely on zero-clipping for night-hour rows.
PV_OP_START = DATASET_START
PV_OP_END   = DATASET_END + ' 23:00'

# ---------------------------------------------------------------------------
# Unit conversions (raw -> kW, kW -> mmBTU/hr) used by prepare.py.
# Reference values from Methods, "Refined data estimation" (Dong et al., 2025).
# ---------------------------------------------------------------------------
TON_TO_KW       = 3.51685         # 1 refrigeration ton -> 3.51685 kW (cooling)
KW_TO_MMBTU_HR  = 0.003412        # 1 kW = 3412 BTU/hr  = 0.003412 mmBTU/hr
MMBTU_HR_TO_KW  = 1.0 / 0.003412  # ~= 293.071 kW per mmBTU/hr (heat conversion)
F_TO_C_OFFSET   = 32.0
F_TO_C_SCALE    = 5.0 / 9.0
MPH_TO_MS       = 0.44704         # miles/hr -> m/s
INHG_TO_HPA     = 33.8639         # inches of mercury -> hPa
INCH_TO_MM      = 25.4            # inches of precipitation -> mm

# ---------------------------------------------------------------------------
# Adaptive-forecasting retraining thresholds
# ---------------------------------------------------------------------------
# CN003 (Tempe Campus) is a multi-MW-scale community aggregation:
#   * Electric demand mean ~15 MW, p99 ~28 MW
#   * Cooling load mean ~6 MW, summer peak ~25 MW
#   * Heat load mean ~2 MW
#   * PV peak ~1 MW (small relative to consumption)
# Magnitudes are ~100x KSRP and ~5000x HouseZero, so the global default
# thresholds (0.08/0.12 kW) and even the KSRP thresholds (100/150 kW) are
# far too tight. The values below sit at roughly 1.5x a competent-baseline
# MAE for the campus scale; tune empirically after the first training run.
ADAPTIVE_MAE_THRESHOLD_KW  = 1500.0
ADAPTIVE_RMSE_THRESHOLD_KW = 2200.0
ADAPTIVE_MASE_THRESHOLD    = 1.5

# Lookback window for adaptive fine-tuning. ASU has a strong weekly
# (academic-calendar) and yearly (cooling-season) periodicity; 1500h (~62
# days) preserves a recent seasonality cycle without going stale.
ADAPTIVE_RETRAIN_WINDOW = 1500

# Per-target MASE retrain trigger. Heat at CN003 is the smallest channel
# (~2 MW mean vs. ~15 MW electric) and could sit at chronically poor MASE
# without firing the overall gate; the 2.0 per-target trigger fixes this.
ADAPTIVE_PER_TARGET_MASE_THRESHOLD = {
    'thermal_heating_kw': 2.0,
}

# Per-target weekly-naive (lag-168 h) anchor weights. ASU's calendar-driven
# load curve makes the weekly-naive baseline competitive; modest blending
# (0.3) recovers near-term diurnal shape after a regime shift (e.g. the
# COVID rebound visible in the 2021-2022 test window).
ADAPTIVE_NAIVE_ANCHOR_WEIGHTS = {
    'thermal_heating_kw': 0.3,
}

# Per-horizon GBDT-weight floor for the ensemble. Tempe's cooling-dominated
# signal is highly predictable from temperature + diurnal cycles; GBDT base
# learners typically beat LSTM-CNN/TFT at longer horizons. The floor pulls
# the optimised weights back toward GBDT at h+12 and h+24.
ADAPTIVE_GBDT_FLOOR_HORIZONS = (1, 12, 24)
ADAPTIVE_GBDT_FLOOR_WEIGHTS  = (0.80, 0.55, 0.45)

# ---------------------------------------------------------------------------
# Legacy calendar seasonal gate (HouseZero-style hard zeroing).
# ---------------------------------------------------------------------------
# The legacy gate in ``apply_physical_constraints`` zeros heating outside the
# is_heating_season window and cooling inside it. That calendar is anchored to
# Cambridge, MA (Oct-15 -> Apr-15) and is physically wrong for Tempe: CN003 is
# a desert, cooling-dominated campus where BOTH thermal channels run all year
# (cooling averages ~12.8 MW even inside the "heating" window and never drops
# to zero; heating averages ~1.4 MW outside it). Applying the gate would zero
# out multi-MW of real demand for ~half the year. Disable it and rely on the
# data-driven INACTIVITY_GATES below for equipment-off detection.
APPLY_SEASONAL_GATE = False

# ---------------------------------------------------------------------------
# Data-driven inactivity gate (mirrors HouseZero / KSRP convention).
# Catches shoulder-season periods where heating/cooling are off but the
# is_heating_season calendar flag hasn't flipped yet. Tempe's heating
# season is essentially nominal (Dec-Feb), so the gate matters more for
# heating than cooling.
# ---------------------------------------------------------------------------
INACTIVITY_GATES = {
    'thermal_heating_kw': {'lookback_hours': 24, 'threshold_kw': 50.0},
    'thermal_cooling_kw': {'lookback_hours': 24, 'threshold_kw': 50.0},
}

# ---------------------------------------------------------------------------
# Per-target training / retraining hyperparameters
# ---------------------------------------------------------------------------
# Schema mirrors NEST_UMAR / KSRP_Campus TRAINING_HYPERPARAMS:
#   {'lstm_cnn': {target: {...}}, 'gbdt': {target: {...}}, 'tft': {target: {...}}}
#
# Defaults are conservative and intentionally light — the HEEW signal is
# clean (paper-cleaned via TCCA + CRAC) and large-scale, so the global
# trainer defaults already produce solid baselines. Override per-target
# only when run-traces show a systematic miss.
TRAINING_HYPERPARAMS = {
    'lstm_cnn': {
        # Cooling load occasionally clips at training-max during summer
        # peaks; modest over-weight protects against under-prediction.
        'thermal_cooling_kw': {'over_weight': 1.5},
    },
    'gbdt': {
        # Allow more boost rounds at retrain time for the larger thermal
        # signals — Tempe drift (climate / COVID rebound) needs real
        # adapter capacity, not a 20-tree nudge.
        'thermal_heating_kw': {'_retrain': {'num_boost_round': 80, 'learning_rate': 0.03}},
        'thermal_cooling_kw': {'_retrain': {'num_boost_round': 80, 'learning_rate': 0.03}},
        'electric_demand_kw': {'_retrain': {'num_boost_round': 50, 'learning_rate': 0.03}},
    },
    'tft': {
        'thermal_cooling_kw': {'over_weight': 1.5},
    },
}
