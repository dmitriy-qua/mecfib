"""
Site-specific constants for the NEST UMAR (Empa, Switzerland) dataset.

Site:    NEST UMAR — Urban Mining and Recycling unit, Empa NEST building,
         Dübendorf, Switzerland.
Period:  2019-07-01 -> 2023-06-30 (4 years, hourly).

Like the other per-dataset modules under forecasting/data/*/constants.py, this
file is auto-discovered by ``apply_dataset_thresholds`` when the input CSV
lives under ``forecasting/data/NEST_UMAR/``. Adding a new constant here is
enough to propagate it to runs — no orchestrator wiring required.
"""

# ---------------------------------------------------------------------------
# Site configuration
# ---------------------------------------------------------------------------
SITE_TIMEZONE  = 'Europe/Zurich'
SITE_LATITUDE  = 47.404
SITE_LONGITUDE = 8.515
SITE_ELEVATION = 440.0           # metres above sea level (approx)

# Dataset window (inclusive)
DATASET_START = '2019-07-01'
DATASET_END   = '2023-06-30'

# Default train/test boundary used by split_train_test.py
TRAIN_END = '2021-06-30 23:00'

# ---------------------------------------------------------------------------
# Adaptive-forecasting retraining thresholds
# ---------------------------------------------------------------------------
# Magnitudes in the test split (hourly means):
#     heating  ~0.53 kW   cooling  ~0.42 kW
#     dhw      ~0.12 kW   electric ~0.21 kW
# Overall mean across all four ~0.32 kW, so a 0.3 kW MAE threshold sits at
# roughly 1× typical demand — fine-tuning only kicks in after a real drift.
ADAPTIVE_MAE_THRESHOLD_KW  = 0.30
ADAPTIVE_RMSE_THRESHOLD_KW = 0.50
ADAPTIVE_MASE_THRESHOLD    = 1.50

# ---------------------------------------------------------------------------
# Data-driven inactivity gate
# ---------------------------------------------------------------------------
# When a target's actuals have been ~zero across the recent context window,
# the forecaster forces its prediction to zero for the entire horizon. This
# handles sensor outages and equipment-off periods the model can't otherwise
# detect (e.g. NEST_UMAR's DHW circuit dropped to zero for several multi-week
# stretches from late 2021 onward — the model had no signal to learn from).
#
# Schema: {target_column: {'lookback_hours': int, 'threshold_kw': float}}
# The gate fires when ``mean(data[target][start - lookback : start]) < threshold``.
#
# DHW: 48 h lookback + 5 W threshold. In the training data 95% of natural
# zero-runs end within 35 h (median 27 h between hot-water draws), so 48 h
# essentially never trips on normal operation. It catches multi-day and
# multi-week outages — which is the failure mode the model can't handle
# on its own, because the LSTM-CNN was trained on full-year patterns where
# October DHW was non-zero, but the post-2021 test data has months of zeros.
INACTIVITY_GATES = {
    'dhw_kw':             {'lookback_hours': 48, 'threshold_kw': 0.005},
    'thermal_heating_kw': {'lookback_hours': 24, 'threshold_kw': 0.05},
    'thermal_cooling_kw': {'lookback_hours': 24, 'threshold_kw': 0.05},
}

# ---------------------------------------------------------------------------
# Per-target training hyperparameters
# ---------------------------------------------------------------------------
# Auto-loaded by ``load_dataset_training_hyperparams`` in the train scripts.
# Each sub-key picks the family ('lstm_cnn' or 'gbdt'); inside, each target
# column maps to a dict of overrides the script knows how to apply.
#
# NEST_UMAR DHW is the failure mode driving these defaults:
#   - Tweedie loss (variance_power=1.5) for the zero-inflated XGB target.
#   - Heavier asymmetric-MSE penalty (over_weight=2.0) on the LSTM-CNN so
#     it stops emitting small positive predictions during all-zero hours.
#   - Relax the zero-sequence filter (min_nonzero_fraction=0.1) so the model
#     actually trains on the rare-active DHW windows instead of discarding
#     them as "predominantly zero".
TRAINING_HYPERPARAMS = {
    'lstm_cnn': {
        'dhw_kw': {
            'over_weight':          2.0,
            'min_nonzero_fraction': 0.1,
        },
    },
    'gbdt': {
        'dhw_kw': {
            'objective':              'reg:tweedie',
            'tweedie_variance_power': 1.5,
        },
    },
}
