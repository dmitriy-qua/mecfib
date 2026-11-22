"""Train one TFT model per consumption target.

Mirrors ``forecasting.consumption.lstm_cnn.train``: same data path, same
season-pure sequence builder, same asymmetric MSE, same per-target
TRAINING_HYPERPARAMS overrides — only the model class differs.

Each model predicts a single target for FORECAST_HORIZON steps ahead from
the full LSTM_COLUMNS input stack.
"""

import argparse
import json
import os
import pickle
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_DIR)))))

from forecasting.constants import FORECAST_HORIZON
from forecasting.utils import create_sequences, seed_everything
from forecasting.consumption.constants import (
    LSTM_COLUMNS, TARGET_COLUMNS, SEQUENCE_LENGTH, OPTIONAL_LSTM_COLUMNS,
)
from forecasting.consumption.lstm_cnn.train import (
    _asymmetric_mse, _season_sequences, _TAG, _SEASON_FILTER,
)
from forecasting.consumption.transformer.model import (
    TFTModel, HIDDEN_DIM, N_HEADS, LSTM_LAYERS, DROPOUT,
)


def parse_args():
    p = argparse.ArgumentParser()
    base = os.path.join(os.path.dirname(_DIR), '..', 'data', 'HouseZero', 'processed')
    p.add_argument('csv_path', nargs='?', default=os.path.join(base, 'data_hourly_train.csv'))
    p.add_argument('--features',         type=str, default='',
                   help='Comma-separated feature columns to use INSTEAD of LSTM_COLUMNS. '
                        'Targets + is_heating_season are auto-prepended when missing.')
    p.add_argument('--targets',          type=str, default='',
                   help='Comma-separated target subset of ' + ','.join(TARGET_COLUMNS) +
                        '. Empty = train all four.')
    p.add_argument('--sequence_length',  type=int,   default=SEQUENCE_LENGTH)
    p.add_argument('--forecast_horizon', type=int,   default=FORECAST_HORIZON)
    p.add_argument('--learning_rate',    type=float, default=5e-4)
    p.add_argument('--epochs',           type=int,   default=100)
    p.add_argument('--patience',         type=int,   default=5)
    p.add_argument('--warmup_epochs',    type=int,   default=5)
    p.add_argument('--batch_size',       type=int,   default=32)
    p.add_argument('--hidden_dim',       type=int,   default=HIDDEN_DIM)
    p.add_argument('--n_heads',          type=int,   default=N_HEADS)
    p.add_argument('--lstm_layers',      type=int,   default=LSTM_LAYERS)
    p.add_argument('--dropout',          type=float, default=DROPOUT)
    p.add_argument('--lambda_smooth',    type=float, default=0.01)
    p.add_argument('--target_clip_quantile', type=float, default=0.999)
    p.add_argument('--over_weight_dhw',   type=float, default=2.0)
    p.add_argument('--over_weight_other', type=float, default=1.0)
    p.add_argument('--min_target_fraction',   type=float, default=0.02)
    p.add_argument('--min_nonzero_fraction',  type=float, default=0.5)
    p.add_argument('--min_nonzero_fraction_dhw',      type=float, default=0.1)
    p.add_argument('--min_nonzero_fraction_electric', type=float, default=0.0)
    p.add_argument('--model_prefix',     type=str, default='tft_consumption')
    _runs_dir = os.path.join(_DIR, 'runs')
    _default_run = os.path.join(_runs_dir, datetime.now().strftime('%Y%m%d_%H%M%S'))
    p.add_argument('--output_dir',       type=str, default=_default_run)
    p.add_argument('--seed', type=int, default=None,
                   help='Random seed. Omit to use $FORECAST_SEED when set '
                        '(train_pipeline exports it), else 42.')
    return p.parse_args()


def _apply_dataset_training_overrides(args) -> None:
    """Pull TFT-specific overrides (or fall back to lstm_cnn ones) from the
    dataset's TRAINING_HYPERPARAMS dict.

    Recognised schemas, in priority order:
        TRAINING_HYPERPARAMS['transformer'][target] = {...}
        TRAINING_HYPERPARAMS['lstm_cnn'][target]    = {...}   # shared
    """
    try:
        from forecasting.consumption.adaptive import load_dataset_training_hyperparams
    except ImportError:
        return
    hp = load_dataset_training_hyperparams(getattr(args, 'csv_path', None))
    tft_hp  = (hp.get('transformer') or {}) if isinstance(hp, dict) else {}
    lstm_hp = (hp.get('lstm_cnn')    or {}) if isinstance(hp, dict) else {}
    merged  = {**lstm_hp, **tft_hp}  # transformer overrides take precedence
    if not merged:
        return

    applied: list[str] = []
    dhw_cfg = merged.get('dhw_kw') or {}
    if 'over_weight' in dhw_cfg:
        args.over_weight_dhw = float(dhw_cfg['over_weight'])
        applied.append(f"over_weight_dhw={args.over_weight_dhw}")
    if 'min_nonzero_fraction' in dhw_cfg:
        args.min_nonzero_fraction_dhw = float(dhw_cfg['min_nonzero_fraction'])
        applied.append(f"min_nonzero_fraction_dhw={args.min_nonzero_fraction_dhw}")
    el_cfg = merged.get('electric_demand_kw') or {}
    if 'min_nonzero_fraction' in el_cfg:
        args.min_nonzero_fraction_electric = float(el_cfg['min_nonzero_fraction'])
        applied.append(f"min_nonzero_fraction_electric={args.min_nonzero_fraction_electric}")
    if applied:
        print(f'[train-tft] dataset overrides: {", ".join(applied)}')


def main():
    args = parse_args()
    args.seed = seed_everything(args.seed)
    _apply_dataset_training_overrides(args)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    _csv_header = pd.read_csv(args.csv_path, nrows=0).columns.tolist()
    available_targets = [t for t in TARGET_COLUMNS if t in _csv_header]
    if set(available_targets) != set(TARGET_COLUMNS):
        _missing_tgts = [t for t in TARGET_COLUMNS if t not in available_targets]
        print(f'[train-tft] dataset is missing target column(s) {_missing_tgts}; those heads will be skipped.')

    if args.targets.strip():
        targets_to_train = [t.strip() for t in args.targets.split(',') if t.strip()]
        unknown = [t for t in targets_to_train if t not in TARGET_COLUMNS]
        if unknown:
            raise SystemExit(f'--targets contains unknown column(s): {unknown}')
        print(f'[train-tft] training {len(targets_to_train)} target head(s): {targets_to_train}')
    else:
        targets_to_train = list(available_targets)

    if args.features.strip():
        user_features = [c.strip() for c in args.features.split(',') if c.strip()]
        required      = list(available_targets) + ['is_heating_season']
        feature_cols  = list(user_features)
        for col in required:
            if col not in feature_cols:
                feature_cols.insert(0, col)
    else:
        feature_cols = [c for c in LSTM_COLUMNS if c in _csv_header]

    _appended = [c for c in OPTIONAL_LSTM_COLUMNS
                 if c in _csv_header and c not in feature_cols]
    if _appended:
        feature_cols.extend(_appended)
        print(f'[train-tft] auto-appended optional features: {_appended}')

    data = pd.read_csv(args.csv_path, parse_dates=['date'], index_col='date',
                       usecols=['date'] + feature_cols)
    data = data[feature_cols].dropna(subset=available_targets).ffill().fillna(0)

    hs_idx = feature_cols.index('is_heating_season')

    q = float(args.target_clip_quantile)
    if 0.0 < q < 1.0:
        for t in TARGET_COLUMNS:
            if t in data.columns:
                thr = float(data[t].quantile(q))
                if (data[t] > thr).any():
                    data[t] = data[t].clip(upper=thr)

    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, f'scaler_{args.model_prefix}_initial.sav'), 'wb') as f:
        pickle.dump(scaler, f)

    # Per-dataset seasonal gate (see lstm_cnn/train.py). When the dataset sets
    # APPLY_SEASONAL_GATE=False (e.g. HEEW), heating/cooling train all-year so
    # training matches the inference gate being skipped.
    try:
        from forecasting.consumption.adaptive import load_dataset_seasonal_gate
        _seasonal_gate = load_dataset_seasonal_gate(getattr(args, 'csv_path', None))
    except ImportError:
        _seasonal_gate = True
    season_filter = dict(_SEASON_FILTER)
    if not _seasonal_gate:
        season_filter['thermal_heating_kw'] = None
        season_filter['thermal_cooling_kw'] = None
        print('[train-tft] APPLY_SEASONAL_GATE=False: training heating/cooling '
              'all-year (no heating-season split).')

    if any(s is not None for s in season_filter.values()):
        _hs_col = data_scaled[:, hs_idx]
        assert _hs_col.min() < 0.5 < _hs_col.max(), (
            'Training data must include both heating and non-heating rows.'
        )
    _hs_raw = data['is_heating_season'].values.astype(bool)

    print('Building sequence sets...')
    X_heating, y_heating = _season_sequences(
        data_scaled, _hs_raw, args.sequence_length, args.forecast_horizon)
    X_cooling, y_cooling = _season_sequences(
        data_scaled, ~_hs_raw, args.sequence_length, args.forecast_horizon)
    X_all,     y_all     = create_sequences(
        data_scaled, args.sequence_length, args.forecast_horizon)
    print(f'  heating:  {len(X_heating)} sequences')
    print(f'  cooling:  {len(X_cooling)} sequences')
    print(f'  all-year: {len(X_all)} sequences')

    for target in targets_to_train:
        tag    = _TAG[target]
        t_idx  = feature_cols.index(target)
        season = season_filter[target]

        if season is True:
            X_s, y_s = X_heating, y_heating[:, :, t_idx : t_idx + 1]
        elif season is False:
            X_s, y_s = X_cooling, y_cooling[:, :, t_idx : t_idx + 1]
        else:
            X_s, y_s = X_all, y_all[:, :, t_idx : t_idx + 1]

        _per_target_min_nonzero = {
            'thermal_heating_kw': args.min_nonzero_fraction,
            'thermal_cooling_kw': args.min_nonzero_fraction,
            'dhw_kw':             args.min_nonzero_fraction_dhw,
            'electric_demand_kw': args.min_nonzero_fraction_electric,
        }
        min_nz = float(_per_target_min_nonzero.get(target, args.min_nonzero_fraction))
        if min_nz > 0.0 and len(y_s) > 0:
            nonzero_steps = (y_s[:, :, 0] > args.min_target_fraction)
            keep = nonzero_steps.mean(axis=1) >= min_nz
            X_s, y_s = X_s[keep], y_s[keep]
            print(f'  [{tag}] {int(keep.sum())}/{len(keep)} sequences kept')

        n         = len(X_s)
        train_end = int(n * 0.70)
        val_end   = int(n * 0.85)
        X_train, y_train = X_s[:train_end],        y_s[:train_end]
        X_val,   y_val   = X_s[train_end:val_end], y_s[train_end:val_end]

        def to_tensor(arr):
            return torch.tensor(arr, dtype=torch.float32).to(device)

        train_loader = DataLoader(TensorDataset(to_tensor(X_train), to_tensor(y_train)),
                                  batch_size=args.batch_size, shuffle=True)
        val_loader   = DataLoader(TensorDataset(to_tensor(X_val),   to_tensor(y_val)),
                                  batch_size=args.batch_size, shuffle=False)

        model = TFTModel(
            input_size=X_train.shape[2],
            output_size=args.forecast_horizon,
            num_features=1,
            hidden_dim=args.hidden_dim,
            n_heads=args.n_heads,
            lstm_layers=args.lstm_layers,
            dropout=args.dropout,
        ).to(device)

        over_weight = float({'dhw_kw': args.over_weight_dhw}.get(target, args.over_weight_other))
        if over_weight != 1.0:
            print(f'  [{tag}] asymmetric MSE over_weight = {over_weight}')

        def criterion(pred, actual, _ow=over_weight):
            return _asymmetric_mse(pred, actual, over_weight=_ow,
                                   eps=args.min_target_fraction)

        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6,
        )

        print(f'\n=== {tag} | {n} sequences  train={train_end}  val={val_end - train_end} '
              f'| warmup={args.warmup_epochs} epochs ===')
        best_val_loss  = float('inf')
        epochs_no_impr = 0
        for epoch in range(args.epochs):
            model.train()
            train_loss = 0.0
            for xb, yb in train_loader:
                optimizer.zero_grad()
                pred   = model(xb)
                mse    = criterion(pred, yb)
                smooth = torch.mean((pred[:, 1:] - pred[:, :-1]) ** 2)
                loss   = mse + args.lambda_smooth * smooth
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()
            train_loss /= max(len(train_loader), 1)

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    pred   = model(xb)
                    mse    = criterion(pred, yb)
                    smooth = torch.mean((pred[:, 1:] - pred[:, :-1]) ** 2)
                    val_loss += (mse + args.lambda_smooth * smooth).item()
            val_loss /= max(len(val_loader), 1)

            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            in_warmup = epoch < args.warmup_epochs
            print(f'Epoch {epoch+1}/{args.epochs}  train={train_loss:.4f}  val={val_loss:.4f}  '
                  f'lr={current_lr:.2e}{"  [warmup]" if in_warmup else ""}')

            if in_warmup:
                continue
            if val_loss < best_val_loss:
                best_val_loss  = val_loss
                epochs_no_impr = 0
                torch.save(model.state_dict(),
                           os.path.join(args.output_dir, f'{args.model_prefix}_{tag}_initial.pth'))
            else:
                epochs_no_impr += 1
                if epochs_no_impr >= args.patience:
                    print(f'Early stopping at epoch {epoch+1}')
                    break

        print(f'Saved {args.model_prefix}_{tag}_initial.pth  (best val={best_val_loss:.4f})')

    arch_config = {
        'hidden_dim':       args.hidden_dim,
        'n_heads':          args.n_heads,
        'lstm_layers':      args.lstm_layers,
        'dropout':          args.dropout,
        'sequence_length':  args.sequence_length,
        'forecast_horizon': args.forecast_horizon,
        'input_size':       int(data_scaled.shape[1]),
        'features':         list(feature_cols),
        'targets':          list(targets_to_train),
    }
    arch_path = os.path.join(args.output_dir, f'{args.model_prefix}_arch.json')
    with open(arch_path, 'w') as f:
        json.dump(arch_config, f, indent=2)
    print(f'Saved architecture config -> {arch_path}')
    print(f'\nAll {len(targets_to_train)} TFT models saved to {args.output_dir}')


if __name__ == '__main__':
    main()
