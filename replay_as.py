import numpy as np
import pandas as pd
from pathlib import Path
from fetch_kalshi_trades import list_markets

from calibrate_fair_value import price_grid, window_values, SIGMA_WINDOW
from fair_value import fair_value_up
from analyze_brti import DATA_DIR, DAY_FILE

GAMMA = 1
K = 150.0
A = 0.05
MAX_Q = 10



def load_mids(start_ts, end_ts):
    """close_ts -> length-900 array of last trade price, forward filled."""
    mkts = list_markets(min_close=start_ts, max_close=end_ts)
    mids = {}
    for ticker, close in zip(mkts["ticker"], mkts["close_ts"]):
        f = Path("data/edges") / f"{ticker}.parquet"
        if not f.exists():
            continue
        tr = pd.read_parquet(f, columns=["sec", "yes_price"])
        if tr.empty:
            continue
        last = tr.groupby("sec")["yes_price"].last()
        arr = last.reindex(range(900)).ffill().bfill().to_numpy()
        mids[int(close)] = arr
    return mids

def run_window(grid, origin, close, verbose=False, gamma=GAMMA, cut_off = False, mids =None, k=0.0):
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
    if mids is None or close not in mids:
                return None
    mkt = mids[close]
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
        if cut_off and seconds_left <= 60:
            continue
        
        m = float(mkt[t - start])
        center = m + k * (s - m)
        tau = seconds_left
        var_left = sigma_p**2 * tau
        r = center - q * gamma * var_left
        spread = gamma * var_left + (2 / gamma) * np.log(1 + gamma / K)
        bid = max(0.01, min(0.99, r - spread / 2))
        ask = max(0.01, min(0.99, r + spread / 2))
        p_bid = min(1.0, A * np.exp(-K * (m - bid)))
        p_ask = min(1.0, A * np.exp(-K * (ask - m)))
        if bid >= ask:
            continue
        if q < MAX_Q and rng.random() < p_bid:
            q += 1
            cash -= bid
        if q > -MAX_Q and rng.random() < p_ask:
            q -= 1
            cash += ask

        if verbose and seconds_left in (600, 300, 60):
            print(close, seconds_left, round(s, 3), round(bid, 3), round(ask, 3), q)

    pnl = cash + q * outcome
    return {"pnl": pnl, "q": q, "outcome": outcome,"gamma":gamma, "cut_off":cut_off,"k":k}


def episodes(grid, origin, gamma=GAMMA,cut_off = False, mids=None, k=0.0):
    results = []
    end = origin + len(grid) - 1
    close = origin + (900 - origin % 900) % 900
    if close == origin:
        close += 900
    while close <= end + 1:
        r = run_window(grid, origin, close, gamma=gamma,cut_off=cut_off, mids =mids, k=k)
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
    paths = sorted(p for p in DATA_DIR.glob("*.csv") if DAY_FILE.match(p.name))
    grid, origin = price_grid(paths)
    mids = load_mids(origin, origin + len(grid))
    for k in (0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0):
        res = episodes(grid, origin, gamma=0.1, mids=mids, k=k)
        cut = int(len(res) * 0.7)
        summarize(f"train k={k}", res[:cut])
        summarize(f"test  k={k}", res[cut:])


if __name__ == "__main__":
    main()