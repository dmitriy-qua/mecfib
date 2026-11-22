"""
Consumption Forecasting Interface for Optimization Integration
==============================================================

Returns 24-hour demand vectors (kW, hourly) for all four consumption targets:
  thermal_heating, thermal_cooling, dhw, electric.

Usage
-----
    from forecasting.consumption.interface import get_forecast

    forecast = get_forecast('forecasting/data/HouseZero/processed/data_hourly_test.csv')
    thermal_heating = forecast['thermal_heating']   # shape (24,), kW
    thermal_cooling = forecast['thermal_cooling']   # shape (24,), kW
    dhw             = forecast['dhw']               # shape (24,), kW
    electric        = forecast['electric']          # shape (24,), kW
"""

import os
import numpy as np
import pandas as pd
import torch

from forecasting.consumption.adaptive import AdaptiveForecaster, create_default_args, validate_args
from forecasting.constants import (
    TARGET_THERMAL_HEATING, TARGET_THERMAL_COOLING, TARGET_DHW, TARGET_ELECTRIC,
)


def _result_to_dict(combined: dict) -> dict:
    return {
        'thermal_heating': np.array(combined[TARGET_THERMAL_HEATING], dtype=float),
        'thermal_cooling': np.array(combined[TARGET_THERMAL_COOLING], dtype=float),
        'dhw':             np.array(combined[TARGET_DHW],             dtype=float),
        'electric':        np.array(combined[TARGET_ELECTRIC],        dtype=float),
    }


def get_forecast(
    data_path: str,
    *,
    non_adaptive: bool = True,
    weights: dict = None,
    forecast_start_idx: int = None,
) -> dict:
    """
    Run a single 24-hour ensemble forecast and return demand arrays.

    Parameters
    ----------
    data_path : str
        Path to the hourly CSV with all feature columns and a 'date' index.
        Requires at least sequence_length (168) + forecast_horizon (24) rows.
    non_adaptive : bool
        If True (default), frozen initial models -- no fine-tuning.
    weights : dict, optional
        Per-target ensemble weights: {target_name: (lstm_w, gbdt_w)}.
        Defaults to (0.3, 0.7) for thermal targets and (0.3, 0.7) for electric.
    forecast_start_idx : int, optional
        Row index of first forecast hour. Defaults to last available window.

    Returns
    -------
    dict
        'thermal_heating' : np.ndarray (24,) -- space heating demand, kW
        'thermal_cooling' : np.ndarray (24,) -- space cooling demand, kW
        'dhw'             : np.ndarray (24,) -- domestic hot water demand, kW
        'electric'        : np.ndarray (24,) -- domestic electricity demand, kW
    """
    data_path_abs = os.path.abspath(data_path)
    if not os.path.exists(data_path_abs):
        raise FileNotFoundError(f"Data file not found: {data_path_abs}")

    data = pd.read_csv(data_path_abs, parse_dates=['date'], index_col='date')

    args = create_default_args()
    args.non_adaptive  = non_adaptive
    args.plot_interval = 0
    args.save_plots    = False
    if weights is not None:
        args.weights = weights
    validate_args(args)

    min_rows = args.sequence_length + args.forecast_horizon
    if len(data) < min_rows:
        raise RuntimeError(
            f"Data has only {len(data)} rows; need at least {min_rows} "
            f"(sequence_length={args.sequence_length} + "
            f"forecast_horizon={args.forecast_horizon})."
        )

    if forecast_start_idx is None:
        forecast_start_idx = len(data) - args.forecast_horizon

    lstm_start_idx = forecast_start_idx - args.sequence_length
    if lstm_start_idx < 0:
        raise RuntimeError(
            f"Not enough context rows before forecast_start_idx={forecast_start_idx}: "
            f"need at least sequence_length={args.sequence_length} rows."
        )
    if forecast_start_idx + args.forecast_horizon > len(data):
        raise RuntimeError(
            f"forecast_start_idx={forecast_start_idx} + "
            f"horizon={args.forecast_horizon} exceeds dataset length {len(data)}."
        )

    device     = 'cuda' if torch.cuda.is_available() else 'cpu'
    forecaster = AdaptiveForecaster(args, device)
    combined   = forecaster.predict(data, forecast_start_idx)
    return _result_to_dict(combined)


class ConsumptionForecaster:
    """
    Persistent forecaster for MPC/rolling-horizon simulations.

    Loads models once at construction; reuse predict() for each MPC step.

    Usage
    -----
    fc   = ConsumptionForecaster()
    data = pd.read_csv('...', parse_dates=['date'], index_col='date')
    r    = fc.predict(data, forecast_start_idx=336)
    thermal_heating = r['thermal_heating']   # (24,)
    thermal_cooling = r['thermal_cooling']   # (24,)
    dhw             = r['dhw']               # (24,)
    electric        = r['electric']          # (24,)
    """

    def __init__(
        self,
        non_adaptive: bool = True,
        weights: dict = None,
        model_dir: str = None,
    ):
        args = create_default_args()
        args.non_adaptive  = non_adaptive
        args.plot_interval = 0
        args.save_plots    = False
        args.model_base_dir = model_dir   # None -> legacy source-tree paths
        if weights is not None:
            args.weights = weights
        validate_args(args)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._forecaster = AdaptiveForecaster(args, device)
        self._args       = args

    def predict(self, data: pd.DataFrame, forecast_start_idx: int) -> dict:
        """
        Run ensemble on data[forecast_start_idx : forecast_start_idx+24].

        Parameters
        ----------
        data                : full DataFrame with datetime index and all feature columns
        forecast_start_idx  : row index of first forecast hour

        Returns
        -------
        dict with 'thermal_heating', 'thermal_cooling', 'dhw', 'electric'
        arrays of shape (forecast_horizon,).
        """
        args = self._args
        if forecast_start_idx - args.sequence_length < 0:
            raise RuntimeError(
                f"Not enough context: forecast_start_idx={forecast_start_idx} "
                f"requires at least {args.sequence_length} preceding rows."
            )
        if forecast_start_idx + args.forecast_horizon > len(data):
            raise RuntimeError(
                f"forecast_start_idx={forecast_start_idx} + "
                f"horizon={args.forecast_horizon} exceeds dataset length {len(data)}."
            )

        combined = self._forecaster.predict(data, forecast_start_idx)
        return _result_to_dict(combined)
