import numpy as np
import pandas as pd
from calibrate_fair_value import price_grid, window_values, SIGMA_WINDOW
from fair_value import fair_value_up
from analyze_brti import DATA_DIR, DAY_FILE
from fetch_kalshi_trades import load_trades

paths = sorted(path for path in DATA_DIR.glob("*.csv") if DAY_FILE.match(path.name))
ticker = "KXBTC15M-26OCT311945-45"
STRIKE = 77405.74
trades = load_trades(ticker= ticker)
grid, origin = price_grid(paths)
close = int(pd.Timestamp("2026-10-31 23:45", tz="UTC").timestamp())
open_ = close - 900

prices = window_values(grid, origin, open_ - SIGMA_WINDOW, close)


print(prices is None, len(prices) if prices is not None else 0)
# 1. Fair value for each second of the window
fair = np.empty(900)
for i in range(900):
    hist = prices[i : SIGMA_WINDOW + i]
    sigma = float(np.std(np.diff(hist), ddof=1))        # dollars per second
    price_now = prices[SIGMA_WINDOW + i]
    seconds_left = 900 - i

    if seconds_left > 60:
        fair[i] = fair_value_up(price_now, STRIKE, sigma, seconds_left)
    else:
        # seconds already locked into the closing 60-second mean
        observed = prices[SIGMA_WINDOW + 840 : SIGMA_WINDOW + i]
        fair[i] = fair_value_up(price_now, STRIKE, sigma, seconds_left, observed_sum=observed.sum(),observed_count=len(observed))
#2. Map each trade to its second in the window
trades["sec"] = (trades["created_time"].astype("int64") // 10**9) - open_
trades = trades[(trades["sec"] >= 0) & (trades["sec"] < 900)].copy()
trades["fair"] = fair[trades["sec"].to_numpy()]
trades["edge"] = trades["yes_price"] - trades["fair"]
# 3. Bucket by fair value
trades["bucket"] = pd.cut(trades["fair"], bins=np.arange(0, 1.01, 0.1))
table = trades.groupby("bucket", observed=True).apply(lambda g: pd.Series({"n_trades": len(g),"volume": g["count"].sum(),"mean_edge": g["edge"].mean(),"vw_edge": np.average(g["edge"], weights=g["count"]),}),include_groups=False)

print(table.round(4))
print("overall vw edge:", round(np.average(trades["edge"], weights=trades["count"]), 4))