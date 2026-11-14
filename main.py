"""polymarket-faa-attribution-02 entry v2: persistence bands with a
case-level sum anchor.

Each market's median stays at its checkpoint price and its bands adapt to
its own recent volatility: an exponentially weighted variance of 1-minute
moves with a 15-minute halflife, stretched-normal quantiles (exponent
1.45) growing like sqrt(h). That base is deliberately conservative: it is
the parameter set that survived every backtest regime, calm and jumpy.

On top of it, v2 adds one case-level feature. Prediction markets in one
family share an aggregate level: the sum of the case's prices mean-reverts
toward its own trailing average (on the practice event and on 35 rolling
checkpoints, the sum moved back toward its trailing 4-hour mean 70% of the
time at the 4-hour horizon). So every market's center is multiplied by
(S_target / S_now) ** 0.25, where S_target is the mean of the case sum
over the trailing 240 minutes and S_now is the sum at the checkpoint, both
over the markets holding at least 240 prices. The adjustment grows in over
the horizon with a 90-minute time constant, so the first minutes stay
close to the checkpoint price. The anchor is the sum's own trailing mean,
never 1: the final's markets are independent questions whose sum sits far
from 1. When no market holds 240 prices the anchor is off and the forecast
is the plain per-market one.

Local verification (pinball loss, lower is better): practice 0.277393 vs
v1's 0.278148 and the house baseline's 0.363589; rolling NVIDIA
checkpoints 1.127753 vs v1's 1.135265; rolling FAA checkpoints tie v1
exactly (the anchor needs 240 minutes of history, which those early
checkpoints do not hold). Pure standard library, deterministic: two runs
on the same case write the same bytes. A market whose history cannot be
read never crashes the run: it falls back to flat persistence bands.
"""

import json
import math
import os

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]
Z_NORM = [-1.6448536269514722, -0.6744897501960817, 0.0,
          0.6744897501960817, 1.6448536269514722]
STRETCH = 1.45
ZS = [math.copysign(abs(z) ** STRETCH, z) for z in Z_NORM]

HALFLIFE_MIN = 15.0
SIGMA_FLOOR = 0.0005
SIGMA_CAP = 0.01
SIGMA_MULT = 0.7

SUM_WINDOW = 240
SUM_POWER = 0.25
SUM_GROW_MIN = 90.0
# Defensive bounds on the anchor. Observed scales on every validation
# dataset run from 0.77 to 1.39, so these bounds never bind there; they
# only stop a pathological live sum from swinging the centers wildly.
SCALE_MIN = 0.5
SCALE_MAX = 2.0


def clamp01(x):
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def price6(x):
    """Clamp to [0, 1], round to 6 decimals, never -0."""
    return round(clamp01(x) * 1e6) / 1e6 + 0.0


def clean_prices(raw):
    """Finite floats in [0, 1]; unusable entries fall back to the last good one."""
    out = []
    last = 0.5
    for v in raw:
        try:
            f = float(v)
        except (TypeError, ValueError):
            f = last
        if not math.isfinite(f):
            f = last
        f = clamp01(f)
        out.append(f)
        last = f
    return out


def market_sigma(clean):
    """Per-minute sigma from an EWMA of squared 1-minute moves."""
    lam = 0.5 ** (1.0 / HALFLIFE_MIN)
    acc = 0.0
    wacc = 0.0
    for i in range(1, len(clean)):
        d = clean[i] - clean[i - 1]
        acc = acc * lam + d * d
        wacc = wacc * lam + 1.0
    var = acc / wacc if wacc > 0.0 else 0.0
    sigma = math.sqrt(var)
    return min(SIGMA_CAP, max(SIGMA_FLOOR, sigma)) * SIGMA_MULT


def case_scale(cleaned):
    """The sum anchor: (trailing mean of the case sum / current sum) ** 0.25.

    Both sums run over the same markets: those holding at least SUM_WINDOW
    prices. Returns 1.0 when the anchor cannot be computed (no qualifying
    market, or a non-positive sum), which leaves the forecast unanchored.
    """
    quals = [p for p in cleaned if len(p) >= SUM_WINDOW]
    if not quals:
        return 1.0
    s_now = sum(p[-1] for p in quals)
    if s_now <= 0.0:
        return 1.0
    total = 0.0
    for j in range(SUM_WINDOW):
        total += sum(p[-SUM_WINDOW + j] for p in quals)
    s_target = total / SUM_WINDOW
    if s_target <= 0.0:
        return 1.0
    scale = (s_target / s_now) ** SUM_POWER
    return min(SCALE_MAX, max(SCALE_MIN, scale))


def fallback_rows(x, horizon):
    """Flat persistence bands: the last price, floor spread only."""
    s0 = SIGMA_FLOOR * SIGMA_MULT
    rows = []
    for h in range(1, horizon + 1):
        s = s0 * math.sqrt(h)
        row = [price6(x + z * s) for z in ZS]
        for j in range(1, 5):
            if row[j] < row[j - 1]:
                row[j] = row[j - 1]
        rows.append(row)
    return rows


def forecast_market(clean, scale, horizon):
    """240 rows of five quantiles for one market's next `horizon` minutes."""
    if not clean:
        return [[0.5, 0.5, 0.5, 0.5, 0.5] for _ in range(horizon)]
    x = clean[-1]
    sigma = market_sigma(clean)
    rows = []
    for h in range(1, horizon + 1):
        grow = 1.0 - math.exp(-h / SUM_GROW_MIN)
        c = x * (scale ** grow) if scale != 1.0 else x
        s = sigma * math.sqrt(h)
        row = [price6(c + z * s) for z in ZS]
        for j in range(1, 5):
            if row[j] < row[j - 1]:
                row[j] = row[j - 1]
        rows.append(row)
    return rows


def forecast_case(markets, horizon):
    """Forecasts for every market of one case: [{"id", "q"}]."""
    cleaned = []
    for m in markets:
        try:
            cleaned.append(clean_prices(m["prices"]))
        except Exception:
            cleaned.append([])
    scale = case_scale(cleaned)
    forecasts = []
    for m, clean in zip(markets, cleaned):
        try:
            rows = forecast_market(clean, scale, horizon)
        except Exception:
            try:
                x = clamp01(float(m["prices"][-1]))
            except Exception:
                x = 0.5
            rows = fallback_rows(x, horizon)
        forecasts.append({"id": m["id"], "q": rows})
    return forecasts


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    names = sorted(f for f in os.listdir(INPUT_DIR) if f.endswith(".in"))
    for name in names:
        case_id = name[: -len(".in")]
        with open(os.path.join(INPUT_DIR, name), "r", encoding="utf-8") as f:
            case = json.load(f)
        horizon = int(case["horizon_minutes"])
        forecasts = forecast_case(case["markets"], horizon)
        payload = json.dumps({"forecasts": forecasts}, separators=(",", ":")) + "\n"
        with open(os.path.join(OUTPUT_DIR, case_id + ".out"), "w", encoding="utf-8") as f:
            f.write(payload)


if __name__ == "__main__":
    main()
