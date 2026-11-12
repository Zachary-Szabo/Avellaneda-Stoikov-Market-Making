import numpy as np

from calibrate_fair_value import price_grid, window_values, SIGMA_WINDOW
from fair_value import fair_value_up
from analyze_brti import DATA_DIR, DAY_FILE

GAMMA = 1.0
K = 150.0
A = 0.05
MAX_Q = 10


def run_window(grid, origin, close, use_fair=True, verbose=False, gamma=GAMMA):
    target = window_values(grid, origin, close - 960, close - 900)
    final = window_values(grid, origin, close - 60, close)
    if target is None or final is None:
        return None
    target_mean = float(target.mean())
    final_mean = float(final.mean())
    outcome = 1.0 if final_mean >= target_mean else 0.0

    start = close - 900
    history = window_values(grid, origin, start - SIGMA_WINDOW, start + 1)
    if history is None:
        return None
    sigma = float(np.std(np.diff(history), ddof=1))

    rng = np.random.default_rng(close)
    q, cash = 0, 0.0
    fair_history = []

    for t in range(start, close):
        seconds_left = close - t
        spot = window_values(grid, origin, t, t + 1)
        if spot is None:
            return None

        if seconds_left > 60:
            s = fair_value_up(float(spot[0]), target_mean, sigma, seconds_left)
        else:
            observed_count = 60 - seconds_left
            observed_sum = 0.0
            if observed_count > 0:
                observed = window_values(grid, origin, close - 60, t)
                observed_sum = float(observed.sum())
            s = fair_value_up(float(spot[0]), target_mean, sigma, seconds_left,
                              observed_sum=observed_sum, observed_count=observed_count)

        fair_history.append(s)
        if len(fair_history) < 2:
            continue
        recent = np.array(fair_history[-60:])
        sigma_p = float(np.std(np.diff(recent))) if len(recent) > 2 else 0.0

        mid = s if use_fair else 0.5
        tau = seconds_left
        var_left = sigma_p**2 * tau
        r = mid - q * gamma * var_left
        spread = gamma * var_left + (2 / gamma) * np.log(1 + gamma / K)
        bid = max(0.01, min(0.99, r - spread / 2))
        ask = max(0.01, min(0.99, r + spread / 2))
        if bid >= ask:
            continue

        # fills depend on the TRUE fair value s, not on where we quote
        p_bid = min(1.0, A * np.exp(-K * (s - bid)))
        p_ask = min(1.0, A * np.exp(-K * (ask - s)))
        if q < MAX_Q and rng.random() < p_bid:
            q += 1
            cash -= bid
        if q > -MAX_Q and rng.random() < p_ask:
            q -= 1
            cash += ask

        if verbose and seconds_left in (600, 300, 60):
            print(close, seconds_left, round(s, 3), round(bid, 3), round(ask, 3), q)

    pnl = cash + q * outcome
    return {"pnl": pnl, "q": q, "outcome": outcome,"gamma":gamma}


def episodes(grid, origin, use_fair=True, gamma=GAMMA):
    results = []
    end = origin + len(grid) - 1
    close = origin + (900 - origin % 900) % 900
    if close == origin:
        close += 900
    while close <= end + 1:
        r = run_window(grid, origin, close, use_fair=use_fair, gamma=gamma)
        if r is not None:
            results.append(r)
        close += 900
    return results


def summarize(name, results):
    pnls = np.array([r["pnl"] for r in results])
    qs = np.array([abs(r["q"]) for r in results])
    sharpe = pnls.mean() / pnls.std() if pnls.std() > 0 else float("nan")
    print(f"{name:<11} n={len(results)}  mean={pnls.mean():.4f}  std={pnls.std():.4f}  "
          f"sharpe={sharpe:.3f}  worst={pnls.min():.4f}  |q|={qs.mean():.2f}")


def main() -> None:
    paths = sorted(path for path in DATA_DIR.glob("*.csv") if DAY_FILE.match(path.name))
    if not paths:
        raise SystemExit(f"No daily CSVs in {DATA_DIR}")
    grid, origin = price_grid(paths)
    for g in (0.01, 0.05, 0.1, 0.5, 1.0):
        summarize(f"fair g={g:g}", episodes(grid, origin, use_fair=True, gamma=g))
    summarize("flat 0.5", episodes(grid, origin, use_fair=False))


if __name__ == "__main__":
    main()