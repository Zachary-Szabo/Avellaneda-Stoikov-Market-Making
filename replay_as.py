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
    """close_ts -> length-900 arrays: last price (ffilled), min and max trade price (NaN if no trade)."""
    mkts = list_markets(min_close=start_ts, max_close=end_ts)
    mids, lows, highs, trades = {}, {}, {}, {}
    for ticker, close in zip(mkts["ticker"], mkts["close_ts"]):
        f = Path("data/edges") / f"{ticker}.parquet"
        if not f.exists():
            continue
        tr = pd.read_parquet(f, columns=["sec", "yes_price", "count"]).sort_values("sec")
        if tr.empty:
            continue
        g = tr.groupby("sec")["yes_price"]
        c = int(close)
        mids[c] = g.last().reindex(range(900)).ffill().bfill().to_numpy()
        lows[c] = g.min().reindex(range(900)).to_numpy()
        highs[c] = g.max().reindex(range(900)).to_numpy()
        trades[c] = (tr["sec"].to_numpy(), tr["yes_price"].to_numpy(), tr["count"].to_numpy())
    return mids, lows, highs, trades


def run_window(grid, origin, close, verbose=False, gamma=GAMMA, cut_off = False, mids =None, k=0.0, lows =None, highs= None,trades=None, min_vol=1):
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
    q, cash = 0, 0.0
    fair_history = []
    if mids is None or close not in mids or close not in lows:
        return None
    mkt, lows_w, highs_w = mids[close], lows[close], highs[close]
    t_sec, t_px, t_cnt = trades[close]
    n_bid = n_ask = 0
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
        i = t - start
        m = float(mkt[i - 1]) if i > 0 else float(mkt[0])
        center = m + k * (s - m)
        tau = seconds_left
        var_left = sigma_p**2 * tau
        r = center - q * gamma * var_left
        spread = gamma * var_left + (2 / gamma) * np.log(1 + gamma / K)
        bid = max(0.01, min(0.99, r - spread / 2))
        ask = max(0.01, min(0.99, r + spread / 2))
        if bid >= ask: continue
        a, b = np.searchsorted(t_sec, [i, i + 1])     # trades in this second
        px, cnt = t_px[a:b], t_cnt[a:b]
        vol_below = cnt[px < bid].sum()
        vol_above = cnt[px > ask].sum()

        if q < MAX_Q and vol_below >= min_vol:
            q += 1; cash -= bid; n_bid += 1
        if q > -MAX_Q and vol_above >= min_vol:
            q -= 1; cash += ask; n_ask += 1

    pnl = cash + q * outcome
    return {"pnl": pnl, "q": q, "outcome": outcome, "gamma": gamma, "cut_off": cut_off, "k": k, "n_bid": n_bid, "n_ask": n_ask, "min_vol": min_vol}


def episodes(grid, origin, gamma=GAMMA,cut_off = False, mids=None, k=0.0,lows=None, highs = None,trades= None,min_vol = 1):
    results = []
    end = origin + len(grid) - 1
    close = origin + (900 - origin % 900) % 900
    if close == origin:
        close += 900
    while close <= end + 1:
        r = run_window(grid, origin, close, gamma=gamma,cut_off=cut_off, mids =mids, k=k, lows=lows,highs=highs, min_vol=min_vol, trades =trades)
        if r is not None:
            results.append(r)
        close += 900
    return results


def summarize(name, results):
    pnls = np.array([r["pnl"] for r in results])
    qs = np.array([abs(r["q"]) for r in results])
    sharpe = pnls.mean() / pnls.std() if pnls.std() > 0 else float("nan")
    fills = np.array([r["n_bid"] + r["n_ask"] for r in results])
    print(f"{name:<18} n={len(results)}  mean={pnls.mean():.4f}  std={pnls.std():.4f}  "f"sharpe={sharpe:.3f}  worst={pnls.min():.4f}  |q|={qs.mean():.2f}  fills={fills.mean():.1f}")


def main() -> None:
    paths = sorted(p for p in DATA_DIR.glob("*.csv") if DAY_FILE.match(p.name))
    grid, origin = price_grid(paths)
    mids, lows, highs, trades = load_mids(origin, origin + len(grid))
    for min_vol in (1, 10, 50):
        for k in (0.0, 0.3, 0.5, 0.7, 1.0):
            res = episodes(grid, origin, gamma=0.1, mids=mids, lows=lows, highs=highs,
                           trades=trades, min_vol=min_vol, k=k)
            cut = int(len(res) * 0.7)
            summarize(f"train v={min_vol} k={k}", res[:cut])
            summarize(f"test  v={min_vol} k={k}", res[cut:])


if __name__ == "__main__":
    main()