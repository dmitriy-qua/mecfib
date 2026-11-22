"""Final ensemble forecast and its accuracy from the member forecasts of a run.

A forecasting run (``python -m forecasting.consumption.adaptive.run``) stores,
for every test day, forecast hour and energy channel, the measured load, the
weekly naive forecast and the forecast of each base learner
(``predictions_non_adaptive.csv.gz`` for frozen models,
``predictions_adaptive.csv.gz`` for periodically retrained models).

This script passes these member forecasts through the combiner of the paper:

1. rolling weights: for each channel and forecast hour, convex weights of the
   members (simplex grid, step 0.05) that minimise the absolute error over the
   last 42 days with known measurements; the first 14 days use the weights
   stored by the run;
2. causal anchor: blend with the weekly naive forecast, with a weight in
   {0, 0.1, ..., 0.9} chosen from the last 28 days (zero until 14 days exist);
3. gates: windows that the run marked as inactive are set to zero.

Member sets reported:
  gbdt            GBDT alone
  gbdt+zero       GBDT and a zero forecast
  gbdt+lstm       GBDT and LSTM-CNN
  ensemble        GBDT, LSTM-CNN and a zero forecast (the method of the paper)

Nothing from the day being forecast, or later, is used for that day.

Usage:
    python ensemble_replay.py RUN_DIR [RUN_DIR ...] [--out metrics.csv]
"""
import argparse
import itertools
import os

import numpy as np
import pandas as pd

WINDOW, MIN_DAYS, STEP = 42, 14, 0.05
ANCHOR_WINDOW, ANCHOR_GRID = 28, np.round(np.arange(0.0, 0.91, 0.1), 2)
SCENARIOS = {'non_adaptive': 'frozen', 'adaptive': 'retrained'}


def simplex(k, step=STEP):
    """All weight vectors of length k on the simplex grid."""
    m = int(round(1 / step))
    return np.array([c for c in itertools.product(range(m + 1), repeat=k) if sum(c) == m], float) / m


def rolling_argmin(err, window, min_days):
    """err: (candidates, days). For each day the candidate with the smallest
    summed error over the `window` days before it (-1 during warm-up)."""
    n = err.shape[1]
    cs = np.concatenate([np.zeros((err.shape[0], 1)), np.cumsum(err, axis=1)], axis=1)
    out = np.full(n, -1, dtype=int)
    for d in range(n):
        lo = max(0, d - window)
        if d - lo >= min_days:
            out[d] = int(np.argmin(cs[:, d] - cs[:, lo]))
    return out


def rolling_blend(members, actual, warmup, window=WINDOW, min_days=MIN_DAYS):
    """members: (k, days, hours). Blend with per-hour rolling weights."""
    grid = simplex(members.shape[0])
    cand = np.tensordot(grid, members, axes=(1, 0))          # (candidates, days, hours)
    err = np.abs(cand - actual[None])
    out = warmup.copy()
    for h in range(actual.shape[1]):
        pick = rolling_argmin(err[:, :, h], window, min(min_days, window))
        ok = pick >= 0
        out[ok, h] = cand[pick[ok], np.where(ok)[0], h]
    return out


def causal_anchor(blend, naive, actual, window=ANCHOR_WINDOW, min_days=MIN_DAYS):
    """Blend with the weekly naive forecast; the weight for a day is chosen from the days before it."""
    n = blend.shape[0]
    ok = ~np.isnan(naive).any(axis=1)
    alpha = np.zeros(n)
    idx = np.where(ok)[0]
    if len(idx):
        err = np.stack([np.abs(actual[idx] - ((1 - a) * blend[idx] + a * naive[idx])).sum(axis=1) for a in ANCHOR_GRID])
        cs = np.concatenate([np.zeros((len(ANCHOR_GRID), 1)), np.cumsum(err, axis=1)], axis=1)
        cur, j = 0.0, 0
        for d in range(n):
            alpha[d] = cur
            if ok[d]:
                j += 1
                lo = max(0, j - window)
                cur = 0.0 if j < min_days else float(ANCHOR_GRID[int(np.argmin(cs[:, j] - cs[:, lo]))])
    out = blend.copy()
    use = ok & (alpha > 0)
    out[use] = (1 - alpha[use, None]) * blend[use] + alpha[use, None] * naive[use]
    return out


def replay(df):
    """Final forecasts of every member set for one prediction file. Returns rows of metrics."""
    rows = []
    for channel, d in df.groupby('target', sort=True):
        d = d.sort_values(['window_idx', 'h'])
        shape = (d.window_idx.nunique(), -1)
        actual = d.actual.to_numpy(float).reshape(shape)
        naive = d.naive_168h.to_numpy(float).reshape(shape)
        gated = d.gate_zeroed.to_numpy().reshape(shape) == 1
        lstm = d.lstm.to_numpy(float).reshape(shape)
        gbdt = d.gbdt.to_numpy(float).reshape(shape)
        stored = d.blend.to_numpy(float).reshape(shape)
        zero = np.zeros_like(gbdt)

        def final(blend):
            return np.where(gated, 0.0, causal_anchor(blend, naive, actual))

        forecasts = {
            'weekly naive': np.where(np.isnan(naive), actual, naive),
            'gbdt': final(gbdt),
            'gbdt+zero': final(rolling_blend(np.stack([zero, gbdt]), actual, gbdt)),
            'gbdt+lstm': final(rolling_blend(np.stack([lstm, gbdt]), actual, stored)),
            'ensemble': final(rolling_blend(np.stack([lstm, gbdt, zero]), actual, stored)),
        }
        nv = float(np.abs(forecasts['weekly naive'] - actual).mean())
        for name, f in forecasts.items():
            e = f - actual
            rows.append({'channel': channel, 'members': name, 'days': shape[0], 'mae_kw': float(np.abs(e).mean()),
                         'rmse_kw': float(np.sqrt((e ** 2).mean())), 'mase': float(np.abs(e).mean() / nv) if nv > 0 else np.nan})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('run_dirs', nargs='+')
    ap.add_argument('--out', default=None, help='write the metrics to this CSV file')
    args = ap.parse_args()
    rows = []
    for run in args.run_dirs:
        for key, label in SCENARIOS.items():
            path = os.path.join(run, f'predictions_{key}.csv.gz')
            if not os.path.exists(path):
                continue
            df = pd.read_csv(path, usecols=lambda c: c != 'timestamp')
            for r in replay(df):
                rows.append({'run': os.path.basename(os.path.normpath(run)), 'models': label, **r})
    out = pd.DataFrame(rows)
    if args.out:
        out.to_csv(args.out, index=False)
    pd.set_option('display.width', 200)
    print(out.round(4).to_string(index=False))


if __name__ == '__main__':
    main()
