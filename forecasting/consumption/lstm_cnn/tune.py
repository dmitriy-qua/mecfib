"""
LSTM-CNN hyperparameter tuning for consumption forecasting.
============================================================

Uses Optuna to search jointly over architecture and training hyperparameters.
One shared architecture is evaluated across all four consumption targets;
the objective is the average validation loss (MSE in scaled space).

To keep each trial fast the tuner uses a reduced epoch budget with tighter
early stopping.  The best architecture is then passed to the full trainer
(printed as a ready-to-run command at the end).

Usage
-----
    pip install optuna
    python -m forecasting.consumption.lstm_cnn.tune
    python -m forecasting.consumption.lstm_cnn.tune --n_trials 40 --tune_epochs 40

After tuning, retrain with the full epoch budget:
    python -m forecasting.consumption.lstm_cnn.train \\
        --hidden_dim 128 --num_layers 2 --dropout 0.35 --learning_rate 0.0005 ...
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_DIR)))))

from forecasting.constants import FORECAST_HORIZON
from forecasting.utils import create_sequences
from forecasting.consumption.constants import LSTM_COLUMNS, TARGET_COLUMNS, SEQUENCE_LENGTH
from forecasting.consumption.lstm_cnn.model import LSTM_CNN_Model
from forecasting.consumption.lstm_cnn.train import _TAG, _IS_HEATING_IDX, _SEASON_FILTER


def parse_args():
    p = argparse.ArgumentParser(description='Tune LSTM-CNN hyperparameters with Optuna.')
    base = os.path.join(os.path.dirname(_DIR), '..', 'data', 'HouseZero', 'processed')
    p.add_argument('--csv',          default=os.path.join(base, 'data_hourly_train.csv'))
    p.add_argument('--n_trials',     type=int, default=40,
                   help='Number of Optuna trials')
    p.add_argument('--tune_epochs',  type=int, default=40,
                   help='Max epochs per trial (lower = faster search, less reliable)')
    p.add_argument('--tune_patience', type=int, default=0,
                   help='Early-stopping patience within each trial')
    p.add_argument('--tune_warmup',  type=int, default=2,
                   help='Warmup epochs (no early stopping) per trial')
    p.add_argument('--batch_size',   type=int, default=32)
    p.add_argument('--timeout',      type=int,   default=None,
                   help='Stop after this many seconds (optional)')
    p.add_argument('--lambda_smooth',       type=float, default=0.1,
                   help='Smoothness regularisation weight (fixed across trials).')
    p.add_argument('--min_target_fraction', type=float, default=0.02,
                   help='Drop seasonal sequences whose peak scaled target is below this.')
    return p.parse_args()


def _load_sequences(csv_path: str):
    """Load data, fit scaler, build all sequences.  Returns (X_all, y_all, scaler)."""
    data = pd.read_csv(csv_path, parse_dates=['date'], index_col='date',
                       usecols=['date'] + LSTM_COLUMNS)
    data = data[LSTM_COLUMNS].dropna(subset=TARGET_COLUMNS).ffill().fillna(0)
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    X_all, y_all = create_sequences(data_scaled, SEQUENCE_LENGTH, FORECAST_HORIZON)
    return X_all, y_all, scaler


def _split_for_target(X_all, y_all, target, min_target_fraction: float = 0.02):
    """Apply seasonal mask, zero-demand filter, and 70/15 temporal split for one target."""
    t_idx  = LSTM_COLUMNS.index(target)
    season = _SEASON_FILTER[target]

    if season is True:
        mask = y_all[:, 0, _IS_HEATING_IDX] >= 0.5
    elif season is False:
        mask = y_all[:, 0, _IS_HEATING_IDX] < 0.5
    else:
        mask = np.ones(len(y_all), dtype=bool)

    X_s = X_all[mask]
    y_s = y_all[mask, :, t_idx:t_idx + 1]

    # Drop near-zero-demand sequences for seasonal targets
    if season is not None:
        peak     = y_s[:, :, 0].max(axis=1)
        nonzero  = peak > min_target_fraction
        X_s = X_s[nonzero]
        y_s = y_s[nonzero]

    n         = len(X_s)
    train_end = int(n * 0.70)
    val_end   = int(n * 0.85)
    return (X_s[:train_end], y_s[:train_end],
            X_s[train_end:val_end], y_s[train_end:val_end])


def _train_one_target(X_train, y_train, X_val, y_val, model, args_tune,
                      device, criterion, lambda_smooth: float = 0.1):
    """Train model for one target; return best val loss."""
    def _t(a):
        return torch.tensor(a, dtype=torch.float32).to(device)

    loader_tr = DataLoader(TensorDataset(_t(X_train), _t(y_train)),
                           batch_size=args_tune.batch_size, shuffle=True)
    loader_v  = DataLoader(TensorDataset(_t(X_val),   _t(y_val)),
                           batch_size=args_tune.batch_size, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=model._tune_lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6)

    best_val  = float('inf')
    no_impr   = 0

    for epoch in range(args_tune.tune_epochs):
        model.train()
        for xb, yb in loader_tr:
            optimizer.zero_grad()
            pred   = model(xb)
            loss   = criterion(pred, yb) + lambda_smooth * torch.mean((pred[:, 1:] - pred[:, :-1]) ** 2)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in loader_v:
                pred      = model(xb)
                val_loss += (criterion(pred, yb) + lambda_smooth * torch.mean((pred[:, 1:] - pred[:, :-1]) ** 2)).item()
        val_loss /= len(loader_v)
        scheduler.step(val_loss)

        if epoch < args_tune.tune_warmup:
            continue

        if val_loss < best_val:
            best_val = val_loss
            no_impr  = 0
        else:
            no_impr += 1
            if no_impr >= args_tune.tune_patience:
                break

    return best_val


def main():
    args_tune = parse_args()

    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print('Optuna not found. Install with:  pip install optuna')
        sys.exit(1)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Loading sequences from {args_tune.csv}  (device={device})')
    X_all, y_all, _ = _load_sequences(args_tune.csv)
    splits = {t: _split_for_target(X_all, y_all, t, args_tune.min_target_fraction)
              for t in TARGET_COLUMNS}
    n_feat = X_all.shape[2]
    print(f'  Sequences: {len(X_all)}  |  features: {n_feat}')
    print(f'  Trials: {args_tune.n_trials}  |  max epochs/trial: {args_tune.tune_epochs}\n')

    criterion = nn.MSELoss()

    def objective(trial):
        hidden_dim   = trial.suggest_categorical('hidden_dim',   [64, 128, 256])
        num_layers   = trial.suggest_int('num_layers', 1, 3)
        cnn_channels = trial.suggest_categorical('cnn_channels', [64, 128, 256])
        kernel_size  = trial.suggest_categorical('kernel_size',  [3, 5, 7])
        dropout      = trial.suggest_float('dropout', 0.1, 0.5)
        lr           = trial.suggest_float('learning_rate', 5e-5, 5e-3, log=True)

        val_losses = []
        for t in TARGET_COLUMNS:
            X_tr, y_tr, X_v, y_v = splits[t]
            model = LSTM_CNN_Model(
                input_size=n_feat,
                output_size=FORECAST_HORIZON,
                num_features=1,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                cnn_channels=cnn_channels,
                kernel_size=kernel_size,
                dropout=dropout,
            ).to(device)
            model._tune_lr = lr   # passed through as attr for simplicity

            val = _train_one_target(X_tr, y_tr, X_v, y_v, model,
                                    args_tune, device, criterion,
                                    lambda_smooth=args_tune.lambda_smooth)
            val_losses.append(val)
            trial.set_user_attr(f'{_TAG[t]}_val_loss', round(val, 6))

        return float(np.mean(val_losses))

    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5),
    )
    study.optimize(objective, n_trials=args_tune.n_trials,
                   timeout=args_tune.timeout, show_progress_bar=True)

    best = study.best_trial
    print(f'\n{"="*60}')
    print(f'Best trial #{best.number}  mean_val_loss = {best.value:.6f}')
    print(f'{"="*60}')
    print('\nPer-target val loss:')
    for t in TARGET_COLUMNS:
        tag = _TAG[t]
        print(f'  {tag:<10s}  {best.user_attrs[f"{tag}_val_loss"]:.6f}')

    print('\nBest hyperparameters:')
    for k, v in best.params.items():
        print(f'  {k:<22s} = {v}')

    cmd_parts = ['python -m forecasting.consumption.lstm_cnn.train']
    for k, v in best.params.items():
        cmd_parts.append(f'--{k} {v:.6g}' if isinstance(v, float) else f'--{k} {v}')
    print(f'\nRun with best params:\n  {" ".join(cmd_parts)}')


if __name__ == '__main__':
    main()
