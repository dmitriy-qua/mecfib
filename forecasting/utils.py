"""
Shared utilities for the HouseZero forecasting module.
"""

from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from forecasting.constants import (
    HEATING_START_MONTH, HEATING_START_DAY,
    HEATING_END_MONTH, HEATING_END_DAY,
    F_TO_C_OFFSET, F_TO_C_SCALE,
)


# ---------------------------------------------------------------------------
# Heating season
# ---------------------------------------------------------------------------

def is_heating_season(date: datetime) -> bool:
    """Return True if date falls within the Cambridge heating season."""
    m, d = date.month, date.day
    after_start = (m > HEATING_START_MONTH) or (m == HEATING_START_MONTH and d >= HEATING_START_DAY)
    before_end  = (m < HEATING_END_MONTH)   or (m == HEATING_END_MONTH   and d <= HEATING_END_DAY)
    return after_start or before_end


def get_season(month: int) -> int:
    """Return season index: 1=winter, 2=spring, 3=summer, 4=autumn."""
    return month % 12 // 3 + 1


# ---------------------------------------------------------------------------
# Unit conversions
# ---------------------------------------------------------------------------

def fahrenheit_to_celsius(f_series):
    """Convert a pandas Series from °F to °C."""
    return (f_series - F_TO_C_OFFSET) * F_TO_C_SCALE


# ---------------------------------------------------------------------------
# Sequence builder for LSTM-CNN
# ---------------------------------------------------------------------------

def create_sequences(data: np.ndarray, input_length: int, output_length: int):
    """
    Build (X, y) pairs for time-series model training.

    Parameters
    ----------
    data         : 2-D array of shape (T, F)
    input_length : look-back window (e.g. 336)
    output_length: forecast horizon  (e.g.  24)

    Returns
    -------
    X : (N, input_length,  F)
    y : (N, output_length, F)
    """
    X, y = [], []
    for i in range(len(data) - input_length - output_length):
        X.append(data[i : i + input_length])
        y.append(data[i + input_length : i + input_length + output_length])
    return np.array(X), np.array(y)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

# Period used for the seasonal naive baseline (168 h = 1 week, hourly data).
# MASE < 1 means the model beats the naive weekly forecast.
SEASONAL_NAIVE_PERIOD = 168


def calculate_mase(y_true: np.ndarray, y_pred: np.ndarray,
                   y_naive: np.ndarray) -> float:
    """
    Mean Absolute Scaled Error.

    MASE = MAE(model) / MAE(seasonal-naive)

    Returns np.nan when the naive baseline is constant (MAE_naive ≈ 0),
    which makes the denominator ill-defined.
    """
    mae_naive = float(np.mean(np.abs(y_true - y_naive)))
    if mae_naive < 1e-8:
        return float('nan')
    return float(np.mean(np.abs(y_true - y_pred))) / mae_naive


def calculate_cv_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Coefficient of Variation of RMSE (CV(RMSE)), expressed as a percentage.

    CV(RMSE) = RMSE / mean(y_true) × 100

    Returns np.nan when the mean of y_true is near zero (e.g. solar at night),
    which makes the denominator ill-defined.
    """
    mean_y = float(np.mean(y_true))
    if abs(mean_y) < 1e-8:
        return float('nan')
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return float(rmse / mean_y * 100)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                      y_naive: np.ndarray | None = None) -> dict:
    """
    Compute MAE, RMSE, CV(RMSE), and (optionally) MASE.

    Parameters
    ----------
    y_true  : ground-truth values
    y_pred  : model predictions
    y_naive : seasonal-naive forecast (same length as y_true).
              When provided, MASE is included in the output.
              Typically y_naive = y_true shifted by SEASONAL_NAIVE_PERIOD.

    Returns
    -------
    dict with keys: mae, rmse, cv_rmse, and optionally mase.
    cv_rmse is RMSE / mean(y_true) × 100 %; nan when mean(y_true) ≈ 0.
    """
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    out  = {'mae': mae, 'rmse': rmse, 'cv_rmse': calculate_cv_rmse(y_true, y_pred)}
    if y_naive is not None:
        out['mase'] = calculate_mase(y_true, y_pred, y_naive)
    return out


# ---------------------------------------------------------------------------
# Weekly-naive baseline (the MASE denominator)
# ---------------------------------------------------------------------------
# Both helpers return the seasonal-naive forecast ŷ(t+h) = y(t+h - period),
# i.e. "same hour last week" for period=168. They exist so every model family
# divides MASE by the *same* correctly-defined baseline. The standalone
# per-dataset floor is produced by ``forecasting.baselines.run``.

def weekly_naive_from_window(X_window: np.ndarray, target_col_idx: int,
                             seq_len: int, horizon: int,
                             period: int = SEASONAL_NAIVE_PERIOD) -> np.ndarray:
    """Weekly-naive forecast for contiguous sequence windows.

    ``create_sequences`` lays each sample's context ``data[i:i+seq_len]``
    immediately before its targets ``data[i+seq_len:i+seq_len+horizon]``. The
    weekly-naive forecast for forecast step h (1-indexed) is therefore the
    context value at position ``seq_len - period + (h - 1)`` — i.e. the slice
    ``X_window[:, seq_len-period : seq_len-period+horizon, target_col_idx]``.

    Parameters
    ----------
    X_window       : (n_samples, seq_len, n_features) input windows (scaled or raw).
    target_col_idx : column index of the target channel within the feature axis.
    seq_len        : context length used to build the windows.
    horizon        : forecast horizon.
    period         : seasonal period (default 168 h = 1 week).

    Returns
    -------
    (n_samples, n_steps) array in the same space as ``X_window``, aligned to
    forecast steps 1..n_steps. ``n_steps = min(horizon, period)`` when the
    context spans at least one full period (``seq_len >= period``). When the
    context is shorter than the period the weekly lag falls outside the window
    entirely, so an empty (n_samples, 0) array is returned and the caller
    should report MASE as nan (use ``forecasting.baselines.run`` for the floor).
    """
    start = seq_len - period
    if start < 0:
        # Context shorter than one seasonal period — no in-window weekly lag.
        return X_window[:, :0, target_col_idx]
    n_steps = min(horizon, period)
    return X_window[:, start:start + n_steps, target_col_idx]


def weekly_naive_calendar(series: pd.Series,
                          period_hours: int = SEASONAL_NAIVE_PERIOD) -> pd.Series:
    """Weekly-naive forecast for a flat, datetime-indexed target series.

    Returns, for each timestamp τ, the observed value at τ − ``period_hours``
    by *calendar time* — robust to gaps and to row-dropping filters (e.g. the
    active-season filter), which a plain ``.shift(period_hours)`` on a filtered
    frame would corrupt. Timestamps with no observation one period earlier are
    NaN; the caller masks them out before computing MASE.

    The input index must be datetime-like. The series is reindexed onto a
    regular hourly grid, shifted by ``period_hours`` rows, then realigned back
    to the original index.
    """
    s = series.sort_index()
    full = pd.date_range(s.index.min(), s.index.max(), freq='h')
    shifted = s.reindex(full).shift(period_hours)
    return shifted.reindex(series.index)


# ---------------------------------------------------------------------------
# Lag feature names helper
# ---------------------------------------------------------------------------

def get_lag_names(prefix: str, count: int) -> list:
    return [f"{prefix}_lag_{i}" for i in range(1, count + 1)]


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

SEED_ENV_VAR = 'FORECAST_SEED'
DEFAULT_SEED = 42


def resolve_seed(seed=None) -> int:
    """Pick the run seed: explicit value -> $FORECAST_SEED -> DEFAULT_SEED.

    A missing or negative ``seed`` means "not given", so a parent process
    (e.g. ``train_pipeline``) can hand its seed to subprocesses through the
    environment without every sub-script needing the flag on its command line.
    """
    import os
    if seed is not None:
        try:
            if int(seed) >= 0:
                return int(seed)
        except (TypeError, ValueError):
            pass
    env = os.environ.get(SEED_ENV_VAR, '').strip()
    if env:
        try:
            if int(env) >= 0:
                return int(env)
        except ValueError:
            pass
    return DEFAULT_SEED


def seed_everything(seed=None, deterministic: bool = True, verbose: bool = True) -> int:
    """Seed Python, NumPy and (when installed) PyTorch; return the seed used.

    Also exports the seed as ``$FORECAST_SEED`` so child processes inherit it.
    XGBoost has no global seed: pass the returned value as its ``seed`` param.

    ``deterministic=True`` switches cuDNN to deterministic kernels. Results on
    a GPU are then repeatable on the same hardware and library versions, but
    bit-identical output across different GPUs or CUDA versions is not
    guaranteed.
    """
    import os
    import random
    seed = resolve_seed(seed)
    os.environ[SEED_ENV_VAR] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except Exception:
        torch = None
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            try:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            except Exception:
                pass
    if verbose:
        print(f'[seed] random seed = {seed}')
    return seed


def code_fingerprint() -> str:
    """Short SHA-256 over the consumption-forecasting sources.

    Covers forecasting/consumption/**/*.py, forecasting/utils.py,
    forecasting/constants.py and every forecasting/data/*/constants.py, with
    line endings normalised, so two runs can be checked to have used the same
    code and dataset settings even when the working tree is not committed.
    """
    import glob
    import hashlib
    import os
    root = os.path.dirname(os.path.abspath(__file__))
    files = sorted(
        glob.glob(os.path.join(root, 'consumption', '**', '*.py'), recursive=True)
        + glob.glob(os.path.join(root, 'data', '*', 'constants.py'))
        + [os.path.join(root, 'utils.py'), os.path.join(root, 'constants.py')]
    )
    h = hashlib.sha256()
    for f in files:
        try:
            with open(f, 'rb') as fh:
                body = fh.read().replace(b'\r\n', b'\n')
        except OSError:
            continue
        h.update(os.path.relpath(f, root).replace(os.sep, '/').encode())
        h.update(b'\0')
        h.update(body)
    return h.hexdigest()[:16]
