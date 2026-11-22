"""Re-run a training or forecasting configuration stored in ``configs/``.

Every folder in ``configs/`` holds the ``params.json`` of one run of the paper
(see ``configs/index.csv``). This script turns it into the command line of the
corresponding module and runs it. Outputs go to ``<runs_dir>/<run id>/``.

A forecasting configuration needs the models of its training configuration
(column ``training_run`` of the index), so run that one first.

Usage:
    python reproduce.py configs/<run id> [--runs_dir runs] [--dry_run]
"""
import argparse
import json
import os
import subprocess
import sys


def command(config_dir, runs_dir):
    run_id = os.path.basename(os.path.normpath(config_dir))
    cfg = json.load(open(os.path.join(config_dir, 'params.json'), encoding='utf-8'))
    module, params = cfg['module'], dict(cfg['params'])
    forecasting = module.endswith('adaptive.run')
    cmd = [sys.executable, '-m', module, '--output_dir', os.path.join(runs_dir, run_id)]
    for item in cfg.get('inputs', []):
        flag = {'train': '--train_csv', 'test': '--csv' if forecasting else '--test_csv'}[item['role']]
        cmd += [flag, item['path']]
    model_run = params.pop('__modelRunId', None)
    if model_run:
        cmd += ['--model_dir', os.path.join(runs_dir, model_run)]
    for key, value in params.items():
        if value is None or value is False or value == '' or value == []:
            continue
        if value is True:
            cmd.append(f'--{key}')
        else:
            cmd += [f'--{key}', ','.join(map(str, value)) if isinstance(value, list) else str(value)]
    return cmd, model_run


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('config_dir')
    ap.add_argument('--runs_dir', default='runs')
    ap.add_argument('--dry_run', action='store_true', help='print the command and exit')
    args = ap.parse_args()
    cmd, model_run = command(args.config_dir, args.runs_dir)
    print(' '.join(cmd))
    if args.dry_run:
        return
    if model_run and not os.path.isdir(os.path.join(args.runs_dir, model_run)):
        sys.exit(f'Models not found: run "python reproduce.py configs/{model_run}" first.')
    sys.exit(subprocess.call(cmd))


if __name__ == '__main__':
    main()
