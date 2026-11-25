"""Final ensemble forecast and its accuracy from the member forecasts of a run.

A forecasting run (``python -m forecasting.consumption.adaptive.run``) stores,
for every test day, forecast hour and energy channel, the measured load, the
weekly naive forecast, the forecast of each base learner and the static blend
(``predictions_non_adaptive.csv.gz`` for frozen models,
``predictions_adaptive.csv.gz`` for periodically retrained models).

This script passes these member forecasts through the combiner of the paper.
Every quantity for a day is computed from earlier days only.

Members (per channel and forecast hour):
  LSTM-CNN and GBDT  the stored forecasts of the two base learners;
  zero               a constant forecast of 0;
  affine             a + b * GBDT, a and b fitted by least squares on the
                     last 42 days with measurements, clipped at 0 (Eq. 2);
  Kalman             a + b * GBDT, a and b following a random walk with
                     variance q = 0.01 relative to the observation noise,
                     updated every day by a Kalman filter, clipped at 0.

Combiner: ML-Poly online aggregation (Eqs. 3-4 of the paper); the static
blend stored by the run is used for the first 14 days.
Then: causal anchor (blend with the weekly naive forecast, weight in
{0, 0.1, ..., 0.9} chosen from the last 28 days; Eq. 5) and the gates
(channels that the run marked as inactive are set to 0).

Rules reported (column ``rule``):
  weekly naive        the weekly naive forecast
  gbdt                GBDT alone (with anchor and gates)
  affine              affine recalibration of GBDT alone
  kalman              Kalman recalibration of GBDT alone
  gbdt+corrections    ML-Poly of GBDT, zero, affine and Kalman members
  ensemble            ML-Poly of LSTM-CNN, GBDT, zero, affine and Kalman members
                      (the configuration of the paper)
With --all also:
  gbdt+zero           ML-Poly of GBDT and the zero member
  gbdt+lstm           GBDT and LSTM-CNN with rolling weights
  rolling4            LSTM-CNN, GBDT, zero and affine members with rolling
                      weights (simplex grid, step 0.05, chosen per hour on the
                      last 42 days; one of the candidates of Table 4b)

Usage:
    python ensemble_replay.py RUN_DIR [RUN_DIR ...] [--all] [--out metrics.csv]
"""
import argparse
import itertools
import os

import numpy as np
import pandas as pd

K_AFFINE, MIN_DAYS, Q_KALMAN = 42, 14, 0.01
K_ROLL, STEP = 42, 0.05
ANCHOR_WINDOW, ANCHOR_GRID = 28, np.round(np.arange(0.0, 0.91, 0.1), 2)
SCENARIOS = {'non_adaptive': 'frozen', 'adaptive': 'retrained'}


# ---------------------------------------------------------------- members
def rolling_affine(X, A, K=K_AFFINE, min_n=MIN_DAYS):
    """Per forecast hour, least-squares fit A = a + b X over the K days before each day; forecast max(a + b X, 0).
    Days with fewer than min_n earlier days keep X."""
    n, H = X.shape
    cs = lambda Z: np.vstack([np.zeros((1, H)), np.cumsum(Z, axis=0)])      # noqa: E731
    Sx, Sy, Sxx, Sxy = cs(X), cs(A), cs(X * X), cs(X * A)
    out = X.copy()
    for w in range(min_n, n):
        lo = max(0, w - K); m = w - lo
        sx, sy, sxx, sxy = Sx[w] - Sx[lo], Sy[w] - Sy[lo], Sxx[w] - Sxx[lo], Sxy[w] - Sxy[lo]
        var = sxx - sx * sx / m
        b = np.where(var > 1e-9, (sxy - sx * sy / m) / np.maximum(var, 1e-12), 1.0)
        a = (sy - b * sx) / m
        out[w] = np.maximum(a + b * X[w], 0.0)
    return out


def kalman_affine(X, A, q=Q_KALMAN, warm=None, min_n=MIN_DAYS):
    """Per forecast hour, intercept and slope of A = a + b X as a random walk (variance q relative to unit
    observation noise) on loads scaled by the mean |X| of the first min_n days; updated by a Kalman filter after
    every day. Days before min_n keep `warm`."""
    n, H = X.shape
    s = np.maximum(np.abs(X[:min_n]).mean(0), 1e-6)
    th = np.zeros((H, 2)); th[:, 1] = 1.0
    P = np.tile(np.eye(2), (H, 1, 1)); Q = q * np.eye(2)
    out = (X if warm is None else warm).copy()
    for w in range(n):
        x = np.stack([np.ones(H), X[w] / s], axis=1)
        P = P + Q
        yh = np.einsum('hi,hi->h', th, x)
        if w >= min_n:
            out[w] = np.maximum(s * yh, 0.0)
        e = A[w] / s - yh
        Px = np.einsum('hij,hj->hi', P, x)
        Kg = Px / (np.einsum('hi,hi->h', x, Px) + 1.0)[:, None]
        th = th + Kg * e[:, None]
        P = P - np.einsum('hi,hj->hij', Kg, Px)
    return out


# ---------------------------------------------------------------- combiners
def mlpol(M, A, warm, min_n=MIN_DAYS):
    """ML-Poly aggregation of the members M (k, days, hours), one weight vector per forecast hour.
    Weights proportional to (R)+ / S, R the cumulative regret of a member against the blend and S the sum of
    its squares; uniform while no member has positive regret. Days before min_n keep `warm`."""
    k, n, H = M.shape
    Rg, S = np.zeros((k, H)), np.zeros((k, H))
    out = warm.copy()
    for w in range(n):
        p = np.maximum(Rg, 0) / np.maximum(S, 1e-12)
        tot = p.sum(0)
        wt = np.where(tot > 0, p / np.where(tot > 0, tot, 1.0), 1.0 / k)
        yhat = (wt * M[:, w]).sum(0)
        if w >= min_n:
            out[w] = yhat
        g = np.sign(yhat - A[w])
        r = g * (yhat - M[:, w])
        Rg += r; S += r * r
    return out


def simplex(k, step=STEP):
    m = int(round(1 / step))
    return np.array([c for c in itertools.product(range(m + 1), repeat=k) if sum(c) == m], float) / m


def rolling_blend(M, A, warm, K=K_ROLL, min_n=MIN_DAYS):
    """Per forecast hour, the simplex weight vector with the smallest absolute error over the K days before
    each day. Days with fewer than min_n earlier days keep `warm`."""
    W = simplex(M.shape[0])
    n, H = A.shape
    out = warm.copy()
    for h in range(H):
        cand = W @ M[:, :, h]                                  # (candidates, days)
        cs = np.concatenate([np.zeros((len(W), 1)), np.cumsum(np.abs(cand - A[None, :, h]), axis=1)], axis=1)
        for d in range(n):
            lo = max(0, d - K)
            if d - lo >= min(min_n, K):
                out[d, h] = cand[int(np.argmin(cs[:, d] - cs[:, lo])), d]
    return out


def causal_anchor(B, N, A, K=ANCHOR_WINDOW, min_n=MIN_DAYS):
    """Blend with the weekly naive forecast N; the weight for a day is chosen from the days before it."""
    n = B.shape[0]
    ok = ~np.isnan(N).any(axis=1)
    alpha = np.zeros(n)
    idx = np.where(ok)[0]
    if len(idx):
        err = np.stack([np.abs(A[idx] - ((1 - a) * B[idx] + a * N[idx])).sum(axis=1) for a in ANCHOR_GRID])
        cs = np.concatenate([np.zeros((len(ANCHOR_GRID), 1)), np.cumsum(err, axis=1)], axis=1)
        cur, j = 0.0, 0
        for w in range(n):
            alpha[w] = cur
            if ok[w]:
                j += 1
                lo = max(0, j - K)
                cur = 0.0 if j < min_n else float(ANCHOR_GRID[int(np.argmin(cs[:, j] - cs[:, lo]))])
    out = B.copy()
    use = ok & (alpha > 0)
    out[use] = (1 - alpha[use, None]) * B[use] + alpha[use, None] * N[use]
    return out


# ---------------------------------------------------------------- replay
def replay(df, scenario, all_rules=False):
    """Final forecasts of every rule for one prediction file; returns rows of metrics."""
    rows = []
    for channel, d in df.groupby('target', sort=True):
        d = d.sort_values(['window_idx', 'h'])
        shape = (d.window_idx.nunique(), -1)
        A = d.actual.to_numpy(float).reshape(shape)
        N = d.naive_168h.to_numpy(float).reshape(shape)
        G = d.gate_zeroed.to_numpy().reshape(shape) == 1
        L = d.lstm.to_numpy(float).reshape(shape)
        X = d.gbdt.to_numpy(float).reshape(shape)
        Z = np.zeros_like(X)
        warm = d.blend.to_numpy(float).reshape(shape)                       # static blend stored by the run
        if scenario == 'adaptive':                                          # retraining runs start from rolling weights
            warm = rolling_blend(np.stack([L, X]), A, warm)

        def final(B):
            return np.where(G, 0.0, causal_anchor(B, N, A))

        Af, Kq = rolling_affine(X, A), kalman_affine(X, A, Q_KALMAN, X)
        F = {'weekly naive': np.where(np.isnan(N), A, N),
             'gbdt': final(X), 'affine': final(Af), 'kalman': final(Kq),
             'gbdt+corrections': final(mlpol(np.stack([X, Z, Af, Kq]), A, X)),
             'ensemble': final(mlpol(np.stack([L, X, Z, Af, Kq]), A, warm))}
        if all_rules:
            F['gbdt+zero'] = final(mlpol(np.stack([X, Z]), A, X))
            F['gbdt+lstm'] = final(rolling_blend(np.stack([L, X]), A, warm))
            F['rolling4'] = final(rolling_blend(np.stack([L, X, Z, Af]), A, warm))
        nv = float(np.abs(F['weekly naive'] - A).mean())
        scale = max(float(np.abs(A).mean()), 1e-9)
        for name, f in F.items():
            e = f - A
            rows.append({'channel': channel, 'rule': name, 'days': shape[0], 'mae_kw': float(np.abs(e).mean()),
                         'rmse_kw': float(np.sqrt((e ** 2).mean())), 'mase': float(np.abs(e).mean() / nv) if nv > 0 else np.nan,
                         'scaled_mae': float(np.abs(e).mean() / scale), 'bias_pct': float(100 * e.mean() / scale)})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('run_dirs', nargs='+')
    ap.add_argument('--all', action='store_true', help='also the rolling-weight rules and GBDT with the zero member (slower)')
    ap.add_argument('--out', default=None, help='write the metrics to this CSV file')
    args = ap.parse_args()
    rows = []
    for run in args.run_dirs:
        for key, label in SCENARIOS.items():
            path = os.path.join(run, f'predictions_{key}.csv.gz')
            if not os.path.exists(path):
                continue
            df = pd.read_csv(path, usecols=['window_idx', 'h', 'target', 'actual', 'naive_168h', 'gate_zeroed', 'lstm', 'gbdt', 'blend'])
            for r in replay(df, key, args.all):
                rows.append({'run': os.path.basename(os.path.normpath(run)), 'models': label, **r})
    out = pd.DataFrame(rows)
    if args.out:
        out.to_csv(args.out, index=False)
    pd.set_option('display.width', 200)
    print(out.round(4).to_string(index=False))


if __name__ == '__main__':
    main()
