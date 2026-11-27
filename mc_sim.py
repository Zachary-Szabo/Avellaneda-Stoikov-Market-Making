import numpy as np
import pandas as pd
from pathlib import Path
from fetch_kalshi_trades import list_markets

from calibrate_fair_value import price_grid, window_values, SIGMA_WINDOW
from fair_value import fair_value_up
from analyze_brti import DATA_DIR, DAY_FILE


K = 150.0
MAX_Q = 10

def sim_btc_path (n_sec, s0=100_000.0, sigma_sec=0.0002, jump_prob=0.0, jump_std=0.002, rng=None):
    if rng is None:
        np.random.default_rng()
    r = rng.normal(0.0, sigma_sec, size=n_sec - 1)
    if jump_prob > 0:
       r += rng.normal(0.0, jump_std, size=n_sec - 1) * (rng.random(n_sec - 1) < jump_prob)
    return s0 * np.exp(np.concatenate([[0.0], np.cumsum(r)]))

def sim_window(s0, sigma_sec, k=0.7, gamma=0.1, lag=2, noise_std=0.01,
               jump_prob=0.0, jump_std=0.002, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    n_sec = SIGMA_WINDOW + 960
    start = SIGMA_WINDOW + 60
    path = sim_btc_path(n_sec, s0=s0, sigma_sec=sigma_sec,
                        jump_prob=jump_prob, jump_std=jump_std, rng=rng)

    target_mean = float(path[start - 60 : start].mean())
    final_mean = float(path[start + 840 : start + 900].mean())
    outcome = 1.0 if final_mean >= target_mean else 0.0

    history = path[start - SIGMA_WINDOW : start + 1]
    sigma = float(np.std(np.diff(history), ddof=1))

    # 1. Compute true fair value each second
    fair = np.empty(900)
    for i in range(900):
        seconds_left = 900 - i
        spot = float(path[start + i])
        if seconds_left > 60:
            fair[i] = fair_value_up(spot, target_mean, sigma, seconds_left)
        else:
            observed_count = 60 - seconds_left
            observed_sum = float(path[start + 840 : start + i].sum()) if observed_count > 0 else 0.0
            fair[i] = fair_value_up(spot, target_mean, sigma, seconds_left,
                                    observed_sum=observed_sum, observed_count=observed_count)

    # 2. Build lagged, noisy synthetic Kalshi mid
    mkt = np.clip(
        np.array([fair[max(0, i - lag)] for i in range(900)]) + rng.normal(0.0, noise_std, size=900),
        0.01, 0.99,
    )

    # 3. Quote and fill loop
    q, cash = 0, 0.0
    n_bid = n_ask = 0
    for i in range(1, 900):
        seconds_left = 900 - i
        s = float(fair[i])
        recent = fair[max(0, i - 59) : i + 1]
        sigma_p = float(np.std(np.diff(recent))) if len(recent) > 2 else 0.0

        m = float(mkt[i - 1])
        center = m + k * (s - m)
        var_left = sigma_p**2 * seconds_left
        r = center - q * gamma * var_left
        spread = gamma * var_left + (2 / gamma) * np.log(1 + gamma / K)
        bid = max(0.01, min(0.99, r - spread / 2))
        ask = max(0.01, min(0.99, r + spread / 2))
        if bid >= ask:
            continue

        trade_px = float(np.clip(fair[i] + rng.normal(0.0, noise_std), 0.01, 0.99))
        if q < MAX_Q and trade_px < bid:
            q += 1
            cash -= bid
            n_bid += 1
        if q > -MAX_Q and trade_px > ask:
            q -= 1
            cash += ask
            n_ask += 1

    return {"pnl": cash + q * outcome, "q": q, "fills": n_bid + n_ask}


def summarize(name, results):
    pnls = np.array([r["pnl"] for r in results])
    qs = np.array([abs(r["q"]) for r in results])
    fills = np.array([r["fills"] for r in results])
    sharpe = pnls.mean() / pnls.std() if pnls.std() > 0 else float("nan")
    p05 = float(np.percentile(pnls, 5))
    print(
        f"{name:<26} n={len(results)}  mean={pnls.mean():.4f}  std={pnls.std():.4f}  "
        f"sharpe={sharpe:.3f}  p05={p05:.4f}  worst={pnls.min():.4f}  "
        f"|q|={qs.mean():.2f}  fills={fills.mean():.1f}"
    )


def main() -> None:
    paths = sorted(p for p in DATA_DIR.glob("*.csv") if DAY_FILE.match(p.name))
    grid, _ = price_grid(paths)
    valid = grid[~np.isnan(grid)]
    s0 = float(np.median(valid))
    sigma_sec = float(np.std(np.diff(np.log(valid))))
    print(f"Calibrated from real BTC: s0={s0:.2f}, sigma_sec={sigma_sec:.6f}\n")

    n_windows = 500
    scenarios = [
        ("GBM lag=2",   dict(lag=2, jump_prob=0.0,   jump_std=0.0)),
        ("GBM lag=0",   dict(lag=0, jump_prob=0.0,   jump_std=0.0)),
        ("Jumps lag=2", dict(lag=2, jump_prob=0.002, jump_std=0.003)),
    ]
    for label, params in scenarios:
        for k in (0.0, 0.7):
            rng = np.random.default_rng(42)
            res = [sim_window(s0, sigma_sec, k=k, gamma=0.1, rng=rng, **params)
                   for _ in range(n_windows)]
            summarize(f"{label} k={k}", res)


if __name__ == "__main__":
    main()