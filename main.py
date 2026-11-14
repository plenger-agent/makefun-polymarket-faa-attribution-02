#!/usr/bin/env python3
"""Entry for make.fun's polymarket-faa-attribution-02 (program/v1).

Forecasts the Yes price of every market in a case at every minute of the
240 minutes after the case's checkpoint, as five quantiles (0.05, 0.25,
0.5, 0.75, 0.95) per minute. Scored by mean pinball loss.

Model, per market, from that market's own minute prices up to the
checkpoint:

* Historical simulation. For a grid of horizons h, take every h-minute
  price change in a recent window of the history and read off the five
  quantiles of those changes directly. Step moves, quiet stretches,
  jumps and the [0, 1] bounds are all in that distribution already, so
  the forecast inherits them instead of assuming a normal walk.
* Volatility regime scaling. Changes are rescaled by the ratio of the
  last 6 hours' root-mean-square 1-minute move to the window's, clamped
  to [0.6, 1.7]: a market that has woken up gets a wider band, one that
  has gone quiet a narrower one.
* Recency weighting. A change counts with weight exp(-age / 480 min),
  so the market's recent behaviour shapes the band more than its
  distant past, while old jumps still fatten the tails a little.
* Horizons between grid points are interpolated in sqrt(h); the whole
  forecast is then blended 25% toward the house persistence baseline
  (flat price, spread 0.0025 * sqrt(h)) as a hedge, and the two outer
  quantiles are pushed 10% further from the median as insurance against
  a news jump, which this live story can produce at any minute.

Deterministic, standard library only, no network: two runs of a case
write the same bytes.

Reads every <id>.in in $MAKEFUN_INPUT_DIR (else /input) in name order
and writes <id>.out to $MAKEFUN_OUTPUT_DIR (else /output).
"""

import json
import math
import os
import sys

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.644854, -0.674490, 0.0, 0.674490, 1.644854]

WINDOW = 2160        # minutes of history the changes are drawn from
DECAY = 480.0        # recency decay of a change's weight, minutes
VOL_SHORT = 360      # minutes for the recent-volatility estimate
VOL_LO, VOL_HI = 0.6, 1.7
BLEND = 0.25         # weight on the persistence baseline
TAIL = 1.1           # outer-quantile push, away from the median
BASE_SIGMA = 0.0025  # baseline spread per sqrt minute
MIN_HIST = 240       # history needed before the empirical model is used
GRID = (1, 2, 3, 5, 8, 12, 20, 30, 45, 70, 100, 140, 180, 240)


def clamp6(v):
    if v < 0.0:
        v = 0.0
    elif v > 1.0:
        v = 1.0
    return round(v, 6) + 0.0


def baseline_rows(x, horizon):
    rows = []
    for h in range(1, horizon + 1):
        s = BASE_SIGMA * math.sqrt(h)
        rows.append([clamp6(x + z * s) for z in Z])
    return rows


def _wquantile(pairs, tau):
    """Weighted quantile of (value, weight) pairs (inverse empirical CDF
    with linear interpolation on cumulative weight)."""
    pairs.sort(key=lambda vw: vw[0])
    total = sum(w for _, w in pairs)
    if total <= 0.0:
        return pairs[0][0]
    target = tau * total
    cum = 0.0
    prev_v, prev_c = pairs[0][0], 0.0
    for v, w in pairs:
        cum += w
        if cum >= target:
            if cum == prev_c:
                return v
            frac = (target - prev_c) / (cum - prev_c)
            return prev_v + frac * (v - prev_v)
        prev_v, prev_c = v, cum
    return pairs[-1][0]


def forecast_market(prices, horizon):
    n = len(prices)
    x = prices[-1]
    if n - 1 < MIN_HIST:
        return baseline_rows(x, horizon)

    # Volatility regime: recent RMS move vs the window's.
    rets = [prices[i] - prices[i - 1] for i in range(1, n)]

    def rms(seg):
        return math.sqrt(sum(r * r for r in seg) / len(seg)) if seg else 0.0

    s_short = rms(rets[-VOL_SHORT:])
    s_win = rms(rets[-WINDOW:])
    scale = 1.0
    if s_win > 1e-9 and s_short > 1e-9:
        scale = min(VOL_HI, max(VOL_LO, s_short / s_win))
    elif s_win <= 1e-9 < s_short:
        scale = VOL_HI

    grid = [g for g in GRID if g <= horizon]
    if grid[-1] != horizon:
        grid.append(horizon)

    # Empirical change quantiles on the horizon grid.
    qgrid = []
    end_hi = n - 1
    for g in grid:
        end_lo = max(g, n - 1 - WINDOW)
        pairs = []
        for end in range(end_lo, end_hi + 1):
            d = (prices[end] - prices[end - g]) * scale
            w = math.exp(-(end_hi - end) / DECAY)
            pairs.append((d, w))
        qgrid.append([_wquantile(list(pairs), tau) for tau in LEVELS])

    sq = [math.sqrt(g) for g in grid]
    base = baseline_rows(x, horizon)
    rows = []
    for h in range(1, horizon + 1):
        sh = math.sqrt(h)
        if sh <= sq[0]:
            qs = qgrid[0]
        elif sh >= sq[-1]:
            qs = qgrid[-1]
        else:
            k = 0
            while sq[k + 1] < sh:
                k += 1
            frac = (sh - sq[k]) / (sq[k + 1] - sq[k])
            qs = [qgrid[k][j] + frac * (qgrid[k + 1][j] - qgrid[k][j])
                  for j in range(len(LEVELS))]
        row = [x + qs[j] for j in range(len(LEVELS))]
        brow = base[h - 1]
        row = [(1.0 - BLEND) * row[j] + BLEND * brow[j]
               for j in range(len(LEVELS))]
        med = row[2]
        row = [med + (row[j] - med) * (TAIL if j in (0, 4) else 1.0)
               for j in range(len(LEVELS))]
        row = [clamp6(v) for v in row]
        for j in range(1, len(row)):
            if row[j] < row[j - 1]:
                row[j] = row[j - 1]
        rows.append(row)
    return rows


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(f for f in os.listdir(in_dir) if f.endswith(".in"))
    for fname in files:
        case_id = fname[:-3]
        with open(os.path.join(in_dir, fname), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        levels = case.get("quantiles", LEVELS)
        if [float(t) for t in levels] != LEVELS:
            raise ValueError(f"{fname} asks for quantiles {levels}, "
                             f"not {LEVELS}")
        horizon = int(case["horizon_minutes"])
        forecasts = []
        for m in case["markets"]:
            prices = m.get("prices") or []
            if prices:
                q = forecast_market([float(p) for p in prices], horizon)
            else:
                q = [[0.5] * len(LEVELS) for _ in range(horizon)]
            forecasts.append({"id": m["id"], "q": q})
        out_path = os.path.join(out_dir, case_id + ".out")
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"forecasts": forecasts}) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
