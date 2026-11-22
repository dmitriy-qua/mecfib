"""
Consumption Forecasting — Training Pipeline
============================================

Trains the base models for consumption forecasting end-to-end and evaluates
them on the held-out test sets:

    Step 1  LSTM-CNN     forecasting.consumption.lstm_cnn.train
    Step 2  GBDT         forecasting.consumption.gbdt.train
    Step 3  Transformer  forecasting.consumption.transformer.train  (optional)
    Step 4  Evaluate     lstm_cnn.evaluate + gbdt.evaluate (+ transformer.evaluate)
    Step 5  Tune         forecasting.consumption.adaptive.tune_weights

After evaluation the per-family metrics.json files are aggregated into
<output_dir>/metrics.json so the orchestrator can persist them alongside
the trained models.

Usage
-----
    python -m forecasting.consumption.train_pipeline \\
        --train_csv path/to/data_hourly_train.csv \\
        --test_csv  path/to/data_hourly_test.csv \\
        --output_dir runs/<id>
"""

import argparse
import json
import os
import subprocess
import sys
import time

_DIR      = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_DIR))

if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
import plot_i18n  # noqa: E402  (repo-root module; needs sys.path above)


def _banner(title: str):
    bar = '=' * 60
    print(f'\n{bar}\n  {title}\n{bar}')


def _run(step_name: str, module: str, extra_args: list[str]) -> float:
    cmd = [sys.executable, '-m', module] + extra_args
    print(f'\n$ {" ".join(cmd)}\n')
    t0 = time.time()
    result = subprocess.run(cmd, cwd=_REPO_ROOT)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f'\n[FAILED]  {step_name}  (exit code {result.returncode})')
        sys.exit(result.returncode)
    print(f'\n[OK]  {step_name}  ({elapsed:.1f}s)')
    return elapsed


def parse_args():
    p = argparse.ArgumentParser(
        description='Train LSTM-CNN, GBDT, and TFT base models for consumption forecasting.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    default_data = os.path.join(_REPO_ROOT, 'forecasting', 'data', 'HouseZero', 'processed')
    # Inputs / outputs. A single training CSV (and an optional test CSV) feeds
    # every training step — GBDT, LSTM-CNN, and TFT all use ``usecols=`` to
    # pull just the columns each one needs, so a "wide" CSV with all features
    # works without preprocessing.
    p.add_argument('--train_csv', default=os.path.join(default_data, 'data_hourly_train.csv'),
                   help='Training CSV consumed by every training step.')
    p.add_argument('--test_csv',  default=None,
                   help='Optional held-out test CSV used by every evaluation step.')
    p.add_argument('--output_dir', default=None)

    # Pipeline flow control
    p.add_argument('--skip_lstm',        action='store_true')
    p.add_argument('--skip_gbdt',        action='store_true')
    p.add_argument('--skip_transformer', action='store_true',
                   help='Skip TFT training. When skipped, the adaptive forecaster '
                        'falls back to the 2-way LSTM+GBDT blend.')
    p.add_argument('--skip_evaluate',    action='store_true')
    p.add_argument('--skip_tune',        action='store_true',
                   help='Skip per-horizon ensemble weight tuning.')
    p.add_argument('--tune_val_frac',    type=float, default=0.15,
                   help='Fraction of training data used as OOF validation window for weight tuning.')
    p.add_argument('--lstm_prefix',      type=str, default='lstm_cnn_consumption')
    p.add_argument('--gbdt_prefix',      type=str, default='xgb_consumption')
    p.add_argument('--tft_prefix',       type=str, default='tft_consumption')

    # Feature-set override (LSTM-CNN only — GBDT keeps GBDT_FEATURE_COLUMNS +
    # lag features). Leave empty to use the canonical consumption LSTM_COLUMNS.
    # Targets and is_heating_season are auto-prepended by the LSTM train
    # script when missing, so the caller is free to pass a pure feature list.
    p.add_argument('--features',      type=str, default='',
                   help='Comma-separated feature columns to override the '
                        'canonical consumption LSTM feature stack. Leave empty '
                        'to use the default LSTM_COLUMNS. Applies to LSTM-CNN '
                        'only; GBDT training is unaffected.')
    p.add_argument('--targets',       type=str, default='',
                   help='Comma-separated target column(s) to train. Pick any '
                        'subset of the 4 consumption heads (thermal_heating_kw, '
                        'thermal_cooling_kw, dhw_kw, electric_demand_kw). '
                        'Leave empty to train all four. Applies to LSTM-CNN; '
                        'GBDT training currently trains all 4 heads.')

    # Forecast windowing (forwarded to LSTM training; GBDT uses lag features and
    # is unaffected at training time but inherits the horizon at inference).
    from forecasting.constants import FORECAST_HORIZON as _HOR_DEF
    from forecasting.consumption.constants import SEQUENCE_LENGTH as _SEQ_DEF
    p.add_argument('--sequence_length',  type=int, default=_SEQ_DEF,
                   help='Hours of history fed to LSTM-CNN per sequence.')
    p.add_argument('--forecast_horizon', type=int, default=_HOR_DEF,
                   help='Hours predicted per forecast window.')

    # LSTM-CNN hyperparameters (forwarded to forecasting.consumption.lstm_cnn.train)
    p.add_argument('--lstm_epochs',        type=int,   default=100)
    p.add_argument('--lstm_patience',      type=int,   default=10)
    p.add_argument('--lstm_warmup_epochs', type=int,   default=5,
                   help='Initial epochs with early-stopping and best-val '
                        'checkpointing disabled. The best checkpoint is only '
                        'written after warmup, so keep this < lstm_epochs.')
    p.add_argument('--lstm_batch_size',    type=int,   default=32)
    p.add_argument('--lstm_learning_rate', type=float, default=0.001)
    p.add_argument('--lstm_hidden_dim',    type=int,   default=128)
    p.add_argument('--lstm_dropout',       type=float, default=0.4)

    # GBDT hyperparameters (forwarded to forecasting.consumption.gbdt.train)
    p.add_argument('--gbdt_n_estimators',          type=int,   default=15000)
    p.add_argument('--gbdt_max_depth',             type=int,   default=4)
    p.add_argument('--gbdt_learning_rate',         type=float, default=0.008)
    p.add_argument('--gbdt_subsample',             type=float, default=0.96)
    p.add_argument('--gbdt_colsample_bytree',      type=float, default=0.99)
    p.add_argument('--gbdt_reg_alpha',             type=float, default=0.26)
    p.add_argument('--gbdt_reg_lambda',            type=float, default=0.23)
    p.add_argument('--gbdt_min_child_weight',      type=int,   default=6)
    p.add_argument('--gbdt_early_stopping_rounds', type=int,   default=300)

    # TFT hyperparameters (forwarded to forecasting.consumption.transformer.train)
    p.add_argument('--tft_epochs',        type=int,   default=80)
    p.add_argument('--tft_patience',      type=int,   default=8)
    p.add_argument('--tft_warmup_epochs', type=int,   default=5,
                   help='Initial epochs with early-stopping and best-val '
                        'checkpointing disabled for the TFT step. Keep this '
                        '< tft_epochs.')
    p.add_argument('--tft_batch_size',    type=int,   default=32)
    p.add_argument('--tft_learning_rate', type=float, default=5e-4)
    p.add_argument('--tft_hidden_dim',    type=int,   default=64)
    p.add_argument('--tft_n_heads',       type=int,   default=4)
    p.add_argument('--tft_lstm_layers',   type=int,   default=1)
    p.add_argument('--tft_dropout',       type=float, default=0.1)
    p.add_argument('--models_from', type=str, default='',
                   help='Reuse trained models instead of training: a folder name under runs/ '
                        '(or an absolute path) whose lstm_cnn/, gbdt/ and transformer/ are '
                        'copied into this run. Only the weight-tuning steps are executed.')
    p.add_argument('--protocol', type=str, default='legacy', choices=['legacy', 'clean'],
                   help="'clean' ignores the hand-set per-dataset overrides (loss weights, "
                        "GBDT floor) and selects the GBDT floor on the validation windows. "
                        "Exported as $FORECAST_PROTOCOL to every step.")
    p.add_argument('--seed', type=int, default=42,
                   help='Random seed shared by every training step (LSTM-CNN, '
                        'GBDT, TFT) and exported to the weight-tuning step. '
                        'Stored in train_meta.json.')
    plot_i18n.add_arg(p)
    return p.parse_args()


def main():
    args = parse_args()
    # Plot language: exported via PLOT_LANG so the evaluation subprocesses
    # spawned below inherit it and translate their plots too.
    plot_i18n.setup_from_args(args)

    # Reproducibility: one seed for the whole pipeline. seed_everything()
    # exports it as $FORECAST_SEED, so the evaluation and weight-tuning
    # subprocesses inherit it; the three trainers also get it explicitly.
    from forecasting.utils import seed_everything, code_fingerprint
    args.seed = seed_everything(args.seed)
    os.environ['FORECAST_PROTOCOL'] = args.protocol

    if not os.path.exists(args.train_csv):
        print(f'[ERROR] File not found: {args.train_csv}')
        sys.exit(1)

    out_lstm = os.path.join(args.output_dir, 'lstm_cnn')    if args.output_dir else None
    out_gbdt = os.path.join(args.output_dir, 'gbdt')        if args.output_dir else None
    out_tft  = os.path.join(args.output_dir, 'transformer') if args.output_dir else None
    for d in [out_lstm, out_gbdt, out_tft, args.output_dir]:
        if d:
            os.makedirs(d, exist_ok=True)

    timings: dict[str, float] = {}
    t_start = time.time()

    # Reuse mode: copy the trained base models of an earlier run and redo only
    # the weight tuning (e.g. after a change in the forecaster or the protocol).
    reuse_tft_train_skip = False
    if args.models_from:
        import shutil
        src = (args.models_from if os.path.isabs(args.models_from)
               else os.path.join(_REPO_ROOT, 'runs', args.models_from))
        if not (args.output_dir and os.path.isdir(os.path.join(src, 'lstm_cnn'))
                and os.path.isdir(os.path.join(src, 'gbdt'))):
            print(f'[ERROR] --models_from: no lstm_cnn/ and gbdt/ under {src}, or no --output_dir')
            sys.exit(1)
        for sub in ('lstm_cnn', 'gbdt', 'transformer'):
            s = os.path.join(src, sub)
            if os.path.isdir(s):
                shutil.copytree(s, os.path.join(args.output_dir, sub), dirs_exist_ok=True)
        print(f'[reuse] base models copied from {src}')
        args.skip_lstm = args.skip_gbdt = args.skip_evaluate = True
        has_tft = os.path.isdir(os.path.join(src, 'transformer'))
        if has_tft:
            reuse_tft_train_skip = True       # keep skip_transformer False: two-way tune still runs
        else:
            args.skip_transformer = True

    _banner('Step 1 / 5  -  LSTM-CNN training')
    if args.skip_lstm:
        print('  Skipped (--skip_lstm)')
    else:
        extra = [
            args.train_csv,
            '--features',         args.features,
            '--targets',          args.targets,
            '--epochs',           str(args.lstm_epochs),
            '--patience',         str(args.lstm_patience),
            '--warmup_epochs',    str(args.lstm_warmup_epochs),
            '--batch_size',       str(args.lstm_batch_size),
            '--learning_rate',    str(args.lstm_learning_rate),
            '--hidden_dim',       str(args.lstm_hidden_dim),
            '--dropout',          str(args.lstm_dropout),
            '--sequence_length',  str(args.sequence_length),
            '--forecast_horizon', str(args.forecast_horizon),
            '--model_prefix',     args.lstm_prefix,
            '--seed',             str(args.seed),
        ]
        if out_lstm:
            extra += ['--output_dir', out_lstm]
        timings['lstm_train'] = _run(
            'LSTM-CNN training', 'forecasting.consumption.lstm_cnn.train', extra,
        )

    _banner('Step 2 / 5  -  GBDT training')
    if args.skip_gbdt:
        print('  Skipped (--skip_gbdt)')
    else:
        extra = [
            args.train_csv,
            '--model_prefix',         args.gbdt_prefix,
            '--seed',                 str(args.seed),
            '--n_estimators',         str(args.gbdt_n_estimators),
            '--max_depth',            str(args.gbdt_max_depth),
            '--learning_rate',        str(args.gbdt_learning_rate),
            '--subsample',            str(args.gbdt_subsample),
            '--colsample_bytree',     str(args.gbdt_colsample_bytree),
            '--reg_alpha',            str(args.gbdt_reg_alpha),
            '--reg_lambda',           str(args.gbdt_reg_lambda),
            '--min_child_weight',     str(args.gbdt_min_child_weight),
            '--early_stopping_rounds', str(args.gbdt_early_stopping_rounds),
        ]
        if out_gbdt:
            extra += ['--output_dir', out_gbdt]
        timings['gbdt_train'] = _run(
            'GBDT training', 'forecasting.consumption.gbdt.train', extra,
        )

    _banner('Step 3 / 5  -  TFT training')
    if reuse_tft_train_skip:
        print('  Skipped (--models_from: TFT checkpoints reused)')
    elif args.skip_transformer:
        print('  Skipped (--skip_transformer)')
    else:
        extra = [
            args.train_csv,
            '--features',         args.features,
            '--targets',          args.targets,
            '--epochs',           str(args.tft_epochs),
            '--patience',         str(args.tft_patience),
            '--warmup_epochs',    str(args.tft_warmup_epochs),
            '--batch_size',       str(args.tft_batch_size),
            '--learning_rate',    str(args.tft_learning_rate),
            '--hidden_dim',       str(args.tft_hidden_dim),
            '--n_heads',          str(args.tft_n_heads),
            '--lstm_layers',      str(args.tft_lstm_layers),
            '--dropout',          str(args.tft_dropout),
            '--sequence_length',  str(args.sequence_length),
            '--forecast_horizon', str(args.forecast_horizon),
            '--model_prefix',     args.tft_prefix,
            '--seed',             str(args.seed),
        ]
        if out_tft:
            extra += ['--output_dir', out_tft]
        timings['tft_train'] = _run(
            'TFT training', 'forecasting.consumption.transformer.train', extra,
        )

    _banner('Step 4 / 5  -  Evaluation')
    if args.skip_evaluate:
        print('  Skipped (--skip_evaluate)')
    elif not (args.test_csv and os.path.exists(args.test_csv)):
        print('  [SKIP] test_csv not provided or missing')
    else:
        eval_args = ['--csv', args.test_csv, '--prefix', args.lstm_prefix]
        if out_lstm:
            eval_args += ['--model_dir', out_lstm]
        timings['lstm_eval'] = _run(
            'LSTM-CNN evaluation', 'forecasting.consumption.lstm_cnn.evaluate',
            eval_args,
        )

        eval_args = ['--csv', args.test_csv, '--prefix', args.gbdt_prefix]
        if out_gbdt:
            eval_args += ['--model_dir', out_gbdt]
        timings['gbdt_eval'] = _run(
            'GBDT evaluation', 'forecasting.consumption.gbdt.evaluate',
            eval_args,
        )

        if not args.skip_transformer:
            eval_args = ['--csv', args.test_csv, '--prefix', args.tft_prefix]
            if out_tft:
                eval_args += ['--model_dir', out_tft]
            timings['tft_eval'] = _run(
                'TFT evaluation', 'forecasting.consumption.transformer.evaluate',
                eval_args,
            )

    _banner('Step 5 / 5  -  Per-horizon ensemble weight tuning')
    if args.skip_tune:
        print('  Skipped (--skip_tune)')
    else:
        tune_out = os.path.join(args.output_dir, 'weights') if args.output_dir else None
        if tune_out:
            os.makedirs(tune_out, exist_ok=True)
        tune_args = [
            '--csv',      args.train_csv,
            '--val_frac', str(args.tune_val_frac),
            '--no_adaptive',
        ]
        if tune_out:
            tune_args += ['--output_dir', tune_out]
        if args.output_dir:
            tune_args += ['--model_dir', args.output_dir]
        timings['tune_weights'] = _run(
            'Ensemble weight tuning', 'forecasting.consumption.adaptive.tune_weights',
            tune_args,
        )
        # Two-way weights for the same LSTM-CNN and GBDT checkpoints, so the
        # two-head variant differs from the three-head one by the TFT alone.
        if not args.skip_transformer and args.output_dir:
            tune2_out = os.path.join(args.output_dir, 'weights_2way')
            os.makedirs(tune2_out, exist_ok=True)
            timings['tune_weights_2way'] = _run(
                'Ensemble weight tuning (two-head variant)',
                'forecasting.consumption.adaptive.tune_weights',
                ['--csv', args.train_csv, '--val_frac', str(args.tune_val_frac),
                 '--no_adaptive', '--no_tft',
                 '--output_dir', tune2_out, '--model_dir', args.output_dir],
            )

    total = time.time() - t_start

    # ── Aggregate per-family metrics into <output_dir>/metrics.json ──────────
    if args.output_dir:
        meta = _build_run_meta(args, timings, total)
        _write_aggregated_metrics(
            args.output_dir,
            {
                'lstm': os.path.join(out_lstm, 'metrics.json') if out_lstm else None,
                'gbdt': os.path.join(out_gbdt, 'metrics.json') if out_gbdt else None,
                'tft':  os.path.join(out_tft,  'metrics.json') if out_tft  else None,
            },
            meta=meta,
        )

        # Sidecar metadata used by the adaptive forecaster / tune_weights to
        # restore the exact training-time window when loading this run.
        with open(os.path.join(args.output_dir, 'train_meta.json'), 'w', encoding='utf-8') as fh:
            json.dump({
                'sequence_length':  int(args.sequence_length),
                'forecast_horizon': int(args.forecast_horizon),
                'seed':             int(args.seed),
                'protocol':         str(args.protocol),
                'models_from':      str(args.models_from or ''),
                'code_sha256':      code_fingerprint(),
            }, fh, indent=2)

    _banner('Pipeline complete')
    for step, t in timings.items():
        print(f'  {step:<20s}  {t:6.1f}s')
    print(f'  {"TOTAL":<20s}  {total:6.1f}s')


def _count_csv_rows(path: str | None) -> int | None:
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return max(sum(1 for _ in f) - 1, 0)
    except Exception:
        return None


def _build_run_meta(args, timings: dict, total_seconds: float) -> dict:
    """Collect run-level context for metrics.json (`_meta` section)."""
    meta: dict = {
        'train_rows':          _count_csv_rows(args.train_csv),
        'test_rows':           _count_csv_rows(args.test_csv),
        'train_seconds_lstm':  float(timings.get('lstm_train', 0.0)),
        'train_seconds_gbdt':  float(timings.get('gbdt_train', 0.0)),
        'train_seconds_tft':   float(timings.get('tft_train',  0.0)),
        'eval_seconds_lstm':   float(timings.get('lstm_eval',  0.0)),
        'eval_seconds_gbdt':   float(timings.get('gbdt_eval',  0.0)),
        'eval_seconds_tft':    float(timings.get('tft_eval',   0.0)),
        'tune_seconds':        float(timings.get('tune_weights', 0.0)),
        'total_seconds':       float(total_seconds),
        'forecast_horizon':    int(args.forecast_horizon),
        'sequence_length':     int(args.sequence_length),
        'hyperparameters': {
            'sequence_length':    args.sequence_length,
            'forecast_horizon':   args.forecast_horizon,
            'lstm_epochs':        args.lstm_epochs,
            'lstm_patience':      args.lstm_patience,
            'lstm_warmup_epochs': args.lstm_warmup_epochs,
            'lstm_batch_size':    args.lstm_batch_size,
            'lstm_learning_rate': args.lstm_learning_rate,
            'lstm_hidden_dim':    args.lstm_hidden_dim,
            'lstm_dropout':       args.lstm_dropout,
            'gbdt_n_estimators':          args.gbdt_n_estimators,
            'gbdt_max_depth':             args.gbdt_max_depth,
            'gbdt_learning_rate':         args.gbdt_learning_rate,
            'gbdt_subsample':             args.gbdt_subsample,
            'gbdt_colsample_bytree':      args.gbdt_colsample_bytree,
            'gbdt_reg_alpha':             args.gbdt_reg_alpha,
            'gbdt_reg_lambda':            args.gbdt_reg_lambda,
            'gbdt_min_child_weight':      args.gbdt_min_child_weight,
            'gbdt_early_stopping_rounds': args.gbdt_early_stopping_rounds,
            'tft_epochs':        args.tft_epochs,
            'tft_patience':      args.tft_patience,
            'tft_warmup_epochs': args.tft_warmup_epochs,
            'tft_batch_size':    args.tft_batch_size,
            'tft_learning_rate': args.tft_learning_rate,
            'tft_hidden_dim':    args.tft_hidden_dim,
            'tft_n_heads':       args.tft_n_heads,
            'tft_lstm_layers':   args.tft_lstm_layers,
            'tft_dropout':       args.tft_dropout,
            'tft_skipped':       bool(args.skip_transformer),
        },
    }

    # Merge per-horizon ensemble weights if tuning was run
    weights_path = (os.path.join(args.output_dir, 'weights', 'weights_per_horizon.json')
                    if args.output_dir else None)
    if weights_path and os.path.exists(weights_path):
        try:
            with open(weights_path, 'r', encoding='utf-8') as fh:
                meta['ensemble_weights_per_horizon'] = json.load(fh)
        except Exception:
            pass

    return meta


def _write_aggregated_metrics(output_dir: str,
                              family_paths: dict[str, str | None],
                              meta: dict | None = None) -> None:
    """
    Merge per-family metrics.json files into a single {target: {metric: value}}
    mapping with `<family>_` prefixes on each metric key. Persisted to
    <output_dir>/metrics.json so the orchestrator can store them in the run.

    When ``meta`` is provided, its contents are written under the reserved
    ``_meta`` key alongside the per-target metrics so the orchestrator can
    surface run-level context (timings, row counts, hyperparameters) without
    co-mingling with numeric per-target metrics.
    """
    aggregated: dict[str, dict] = {}
    for family, path in family_paths.items():
        if not path or not os.path.exists(path):
            continue
        try:
            with open(path, 'r', encoding='utf-8') as f:
                per_family = json.load(f)
        except Exception as exc:
            print(f'  [WARN] Could not read {path}: {exc}')
            continue
        for target, metrics in per_family.items():
            bucket = aggregated.setdefault(target, {})
            for metric_key, value in metrics.items():
                try:
                    bucket[f'{family}_{metric_key}'] = float(value)
                except (TypeError, ValueError):
                    # Non-numeric entries land in _meta, not per-target metrics.
                    continue

    if not aggregated and not meta:
        print('  [WARN] No per-family metrics.json files found — skipping aggregation')
        return

    payload: dict = dict(aggregated)
    if meta:
        payload['_meta'] = meta

    out_path = os.path.join(output_dir, 'metrics.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)
    print(f'\nSaved aggregated metrics -> {out_path}')


if __name__ == '__main__':
    main()
