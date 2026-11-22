"""
Site-specific constants for the KSRP Campus (Kitakyushu Science and Research
Park, Japan) twenty-year district-energy dataset.

Source publication: W. Liao *et al.*, "A twenty-year dataset of hourly energy
generation and consumption from district campus building energy systems,"
*Scientific Data* 11:1400 (2024). DOI: 10.1038/s41597-024-04244-6.

Period:  2002-01-01 -> 2021-12-31 (20 years, hourly).
Site:    Kitakyushu Science and Research Park, Kyushu, Japan
         (~33.85 N, 130.83 E, sea-level, JST = UTC+9 fixed, no DST).

Equipment:
  * Gas engine     : 160 kW electric, 28.7 % electric efficiency.    Op. 2002–2016.
  * Fuel cell      : 200 kW electric, 40 % electric efficiency.      Op. 2002–2011.
  * Gas boiler     : 80  kW thermal (hot-water generating).
  * Absorption chillers : 2110 kW cooling / 1780 kW heating capacity.
  * PV system      : 153 kW peak, 13 % module efficiency, rooftop.    Op. 2006–2021.

The KSRP campus supplies electricity, hot water, space heating, and cooling to
six buildings (two offices, one laboratory, one gymnasium, one library, one
classroom/office complex). This dataset module produces a single
district-aggregated hourly time series; per-building water-loop and
heating/cooling channels are read but not split out by default.

The Tohoku earthquake of March 2011 produced a campus-wide outage; data in
March 2011 are missing in every source folder. The pipeline drops those rows
explicitly rather than interpolating them.

Like the other per-dataset modules under ``forecasting/data/*/constants.py``,
this file is auto-discovered by ``apply_dataset_thresholds`` when the input
CSV lives under ``forecasting/data/KSRP_Campus/``.
"""

# ---------------------------------------------------------------------------
# Site configuration
# ---------------------------------------------------------------------------
# Japan Standard Time has no DST: UTC+9 year-round. Use the IANA Asia/Tokyo
# alias for clarity; downstream code that needs a fixed-offset clock can fall
# back to Etc/GMT-9 (IANA sign is inverted) and obtain the same offsets.
SITE_TIMEZONE   = 'Asia/Tokyo'
SITE_LATITUDE   = 33.854
SITE_LONGITUDE  = 130.832
SITE_ELEVATION  = 12.0           # metres above sea level, approx.

# ---------------------------------------------------------------------------
# Dataset window (inclusive)
# ---------------------------------------------------------------------------
DATASET_START = '2002-01-01'
DATASET_END   = '2021-12-31'

# Chronological train/test boundary (decade split, matches the paper's
# "first ten years / last ten years" framing in Section 3 of the descriptor).
TRAIN_END     = '2011-12-31 23:00'

# Earthquake / outage window (Tohoku, 11 March 2011) — all source folders are
# silent in this window. We drop these rows rather than interpolate them.
OUTAGE_WINDOWS = [
    ('2011-03-01', '2011-03-31 23:00'),
]

# ---------------------------------------------------------------------------
# Equipment operating windows
# ---------------------------------------------------------------------------
# Used by the QC step to flag (not drop) rows where an off-line generator
# reports a small non-zero residual due to standby telemetry noise.
GAS_ENGINE_OP_START = '2002-01-01'
GAS_ENGINE_OP_END   = '2016-07-31 23:00'   # paper: "until July 2016"
FUEL_CELL_OP_START  = '2002-01-01'
FUEL_CELL_OP_END    = '2011-11-30 23:00'   # paper: "until November 2011"
PV_OP_START         = '2006-01-01'         # 15-year service life ending in 2021
PV_OP_END           = '2021-12-31 23:00'

# ---------------------------------------------------------------------------
# Adaptive-forecasting retraining thresholds
# ---------------------------------------------------------------------------
# KSRP operates at 30-500x the HouseZero residential scale (hourly mean
# magnitudes in the test split: heating ~110 kW, cooling ~150 kW, dhw
# ~15 kW, electric ~250 kW; overall mean ~135 kW), so the global defaults
# of 0.08 / 0.12 kW would trip on every window and force constant
# retraining. The thresholds below sit at roughly 1.5x best-base MAE from
# run-20260520-* (where GBDT MAE was ~80-100 kW on thermal, ~50 kW on
# electric) — high enough that retraining only fires on real drift, low
# enough to catch a regime shift before it dominates the metrics. The
# MASE bar of 1.5 is also raised relative to the HouseZero 1.2 because
# KSRP's weekly-naive baseline is harder to beat (the campus has very
# regular weekly schedules) and a tighter bar would over-fire.
ADAPTIVE_MAE_THRESHOLD_KW  = 100.0
ADAPTIVE_RMSE_THRESHOLD_KW = 150.0
ADAPTIVE_MASE_THRESHOLD    = 1.5

# Lookback window for adaptive fine-tuning. The global default is 2000 h
# (~83 days); KSRP's drift is slower (decade-scale equipment retirements
# rather than per-week occupancy changes), so a shorter 1500 h (~62 days)
# window gives fresher signal without losing seasonality.
ADAPTIVE_RETRAIN_WINDOW = 1500

# Per-target MASE retrain trigger. Augments the overall MAE/RMSE/MASE gates
# above (which average across all four targets and so never react to a
# chronically-bad small-scale target). DHW MAE ~12 kW is dwarfed by heating
# /cooling/electric (50-100 kW) in the overall mean, so without this gate
# DHW drift can sit at MASE ~3.8 indefinitely. The 2.0 threshold (vs the
# overall 1.5) trips when DHW is solidly worse than the weekly-naive
# baseline, which is the right scale for a per-target alarm.
ADAPTIVE_PER_TARGET_MASE_THRESHOLD = {
    'dhw_kw': 2.0,
}

# Per-target weekly-naive (lag-168 h) anchor weights for the post-combine
# blend. KSRP DHW underwent a control/schedule regime change around 2016
# (morning-peak diurnal shape inverted to night-flat), invisible to the
# 2002-2011 training window. The trained ensemble lands at MAE ~12 kW and
# R² ~-2; the weekly-naive baseline alone lands at MAE ~5 kW. Blending at
# 0.6 (model 0.4, naive 0.6) preserves the model's short-term weather
# response while pulling the diurnal shape back to the recent observed
# week. Targets without an entry keep the pure-ensemble output.
ADAPTIVE_NAIVE_ANCHOR_WEIGHTS = {
    'dhw_kw': 0.6,
}

# Per-horizon GBDT-weight floor for the ensemble. Run-20260520-131855
# showed GBDT as the per-target-best learner on heating, cooling, and
# electric (R^2 0.74 / 0.86 / 0.90) while the ensemble was 26–50 % worse
# than GBDT alone on heating/cooling. Tightening the GBDT floor pulls
# the optimised weights back toward GBDT at long horizons where the
# deep learners' confidence interval is widest. Anchors interpolated:
#   h+1  → w_gbdt ≥ 0.85
#   h+12 → w_gbdt ≥ 0.55
#   h+24 → w_gbdt ≥ 0.45
# The global default (0.80 / 0.45 / 0.30) is a sensible mid-point for
# datasets where the three base learners are roughly comparable; KSRP's
# additional 15-pp boost reflects its specific empirical dominance.
ADAPTIVE_GBDT_FLOOR_HORIZONS = (1, 12, 24)
ADAPTIVE_GBDT_FLOOR_WEIGHTS  = (0.85, 0.55, 0.45)

# ---------------------------------------------------------------------------
# Data-driven inactivity gate (mirrors HouseZero/NEST_UMAR convention).
# Catches shoulder-season periods where heating/cooling are off but the
# is_heating_season calendar flag hasn't flipped yet.
# ---------------------------------------------------------------------------
INACTIVITY_GATES = {
    'thermal_heating_kw': {'lookback_hours': 24, 'threshold_kw': 1.0},
    'thermal_cooling_kw': {'lookback_hours': 24, 'threshold_kw': 1.0},
}

# ---------------------------------------------------------------------------
# Per-target training / retraining hyperparameters
# ---------------------------------------------------------------------------
# Schema mirrors NEST_UMAR's TRAINING_HYPERPARAMS:
#   {'lstm_cnn': {target: {...}}, 'gbdt': {target: {...}}, 'tft': {target: {...}}}
#
# Rationale for KSRP-specific values:
#   * DHW now uses `reg:squarederror` on the GBDT side (overrides the
#     module-level `reg:tweedie` default). Tweedie targets a compound
#     Poisson-Gamma distribution suitable for the HouseZero DHW signal
#     (~90% zero, rare bursts) — KSRP DHW is 99.8% non-zero with a smooth
#     Gaussian-like distribution (mean 55 kW, std 11 kW), so Tweedie's
#     mass-at-zero prior is actively wrong. An earlier KSRP A/B
#     (run 20260522-162305) showed squarederror going from R²=-1.07
#     (Tweedie) to -1.30, but that comparison was confounded: the model
#     missed the campus DHW schedule inversion around 2016 (midday-peak
#     pattern of 2002-2014 inverts to overnight-peak by 2017-2019 and
#     flattens by 2020-2021) regardless of objective, and the metric was
#     dominated by the regime change. The fixes are layered: switch to
#     squarederror (right loss for the distribution), and blend with a
#     weekly-naive anchor at predict time (see ADAPTIVE_NAIVE_ANCHOR_WEIGHTS)
#     to recover the recent diurnal shape that the trained model can't see.
#   * Electric has the most stable signal (best R^2 already) — a modest
#     over_weight=1.5 protects against the LSTM-CNN slipping into
#     systematic under-prediction during peaks once fine-tuning kicks in.
#     This entry was dead code prior to the 2026.05 refactor of
#     _apply_dataset_training_overrides (which only forwarded DHW knobs);
#     it now takes effect at both initial training and adaptive fine-tune.
#   * Cooling currently uses symmetric loss (no `under_weight` override).
#     Run 20260522-162305 showed ~31% systematic under-prediction on
#     cooling totals — test cooling peaks exceed training peaks by ~17%
#     at p99 and ~30% at the absolute max (climate-shift over the test
#     decade), and the LSTM scaler caps output at the training max. The
#     `--under_weight_cooling` knob (added 2026.05, see the LSTM-CNN
#     trainer) plus a `thermal_cooling_kw: {'under_weight': 2.0}` entry
#     here is the canonical fix; it is intentionally not enabled by
#     default while the right value is being calibrated.
#   * GBDT thermal heads benefit from more boost rounds at retrain time
#     (50 vs the global 20) and a lower learning rate (0.03) because the
#     baseline boosters carry 200-500 trees: 20 extra trees barely moves
#     the prediction surface; 50 lower-LR trees actually adapt.
TRAINING_HYPERPARAMS = {
    'lstm_cnn': {
        'electric_demand_kw': {'over_weight': 1.5},
    },
    'gbdt': {
        # Per-family retrain knobs read by AdaptiveForecaster._finetune_gbdt.
        # Empty inner dicts (e.g. {'_retrain': {...}}) leave the base
        # objective unchanged but override the boost schedule. Tree counts
        # roughly doubled from the earlier 50/50/30 (matching the
        # global 20→50 default bump) — KSRP's 1500h retrain window can
        # absorb the extra capacity, and the regime shifts visible in
        # the test split (gas engine phaseout, DHW schedule inversion)
        # need real adapter capacity to track, not a 20-tree nudge.
        'thermal_heating_kw': {'_retrain': {'num_boost_round': 100, 'learning_rate': 0.03}},
        'thermal_cooling_kw': {'_retrain': {'num_boost_round': 100, 'learning_rate': 0.03}},
        'electric_demand_kw': {'_retrain': {'num_boost_round':  60, 'learning_rate': 0.03}},
        # Override the module-level Tweedie default — see the rationale block
        # above (DHW at KSRP is not zero-inflated; Tweedie is the wrong loss).
        # The {'objective': ...} key is consumed by both
        # _apply_dataset_objective_overrides at initial training and
        # _finetune_gbdt at adaptive retrain. DHW inherits the new 50-tree
        # default (no _retrain block needed).
        'dhw_kw':             {'objective': 'reg:squarederror'},
    },
    'tft': {
        # TFT is more sensitive than the LSTM-CNN to fine-tune nudges
        # (attention layers re-route easily). A heavier over_weight stops
        # KSRP's clipped peaks from biasing the model upward — the
        # training-time target_clip_quantile already lowered scaler.data_max_,
        # so a free model output is much more likely to overshoot than
        # under-shoot. Combined with the runtime clamp_scaled_to_unit
        # guardrail in forecaster._predict_tft, this should bring TFT into
        # line with the LSTM/GBDT base learners.
        'thermal_heating_kw': {'over_weight': 2.0},
        'thermal_cooling_kw': {'over_weight': 2.0},
        'electric_demand_kw': {'over_weight': 1.5},
    },
}

# ---------------------------------------------------------------------------
# 2014 calibration table.
#
# The 2014 loads file ships its three thermal channels in non-kW units that
# the publication does not document. The hot-water column is in MJ/h (the
# 1 MJ/h = 0.2778 kW conversion matches the 2013/2015 envelope to better
# than 6% at every hour). The cooling and heating columns appear to be in
# different physical units (likely water-flow proxies); for those we apply
# empirical multiplicative factors derived by same-hour matching against
# the (2013 + 2015) / 2 reference envelope. The shape of the 2014 seasonal
# and diurnal patterns is preserved verbatim from the source file; only the
# overall magnitude is calibrated.
#
# The calibration is intentionally a simple per-column scalar so the
# behaviour stays auditable. Users wanting a physically-rigorous unit
# conversion for 2014 should consult the data descriptor and override
# these factors after contacting the data authors.
# ---------------------------------------------------------------------------
YEAR_CALIBRATION = {
    2014: {
        'thermal_cooling_kw': 155.0,    # empirical (median ratio 158, linear-fit 127)
        'thermal_heating_kw':  52.0,    # empirical (median ratio 50,  linear-fit 43)
        'dhw_kw':              0.2778,  # exact MJ/h -> kW conversion
    },
}
