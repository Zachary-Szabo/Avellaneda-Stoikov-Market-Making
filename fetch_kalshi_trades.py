"""Fetch Kalshi historical trades for KXBTC15M markets and cache them to CSV."""
import time
from pathlib import Path

import pandas as pd
import requests
import numpy as np
from calibrate_fair_value import price_grid, window_values, SIGMA_WINDOW
from fair_value import fair_value_up
from analyze_brti import DATA_DIR, DAY_FILE

BASE = "https://api.elections.kalshi.com/trade-api/v2"
OUT_DIR = Path("data/kalshi_trades")
paths = sorted(path for path in DATA_DIR.glob("*.csv") if DAY_FILE.match(path.name))

def fetch_from(endpoint, ticker, pause):
    rows, cursor = [], None
    while True:
        params = {"ticker": ticker, "limit": 1000}
        if cursor:
            params["cursor"] = cursor
        resp = requests.get(f"{BASE}/{endpoint}", params=params, timeout=30)
        resp.raise_for_status()
        body = resp.json()
        rows.extend(body.get("trades", []))
        cursor = body.get("cursor")
        if not cursor:
            break
        time.sleep(pause)
    return rows


def fetch_trades(ticker, pause=0.2):
    """Return every trade for one market ticker as a DataFrame."""
    rows = fetch_from("markets/trades", ticker, pause)
    if not rows:  # older markets get moved to the historical archive
        rows = fetch_from("historical/trades", ticker, pause)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["created_time"] = pd.to_datetime(df["created_time"], utc=True, format="ISO8601")
    df["yes_price"] = df["yes_price_dollars"].astype(float)
    df["count"] = df["count_fp"].astype(float)
    return df.sort_values("created_time").reset_index(drop=True)

def load_trades(ticker):
    """Load from cache if present, otherwise fetch and save."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{ticker}.csv"
    if path.exists():
        return pd.read_csv(path, parse_dates=["created_time"])
    df = fetch_trades(ticker)
    df.to_csv(path, index=False)
    return df

def list_markets(series="KXBTC15M", min_close=None, max_close=None, pause=0.2):
    rows, cursor = [], None
    while True:
        params = {"series_ticker": series, "status": "settled", "limit": 1000,
                  "min_close_ts": min_close, "max_close_ts": max_close}
        if cursor:
            params["cursor"] = cursor
        resp = requests.get(f"{BASE}/markets", params=params, timeout=30)
        resp.raise_for_status()
        body = resp.json()
        rows.extend(body.get("markets", []))
        cursor = body.get("cursor")
        if not cursor:
            break
        time.sleep(pause)
    df = pd.DataFrame(rows)
    df["close_ts"] = pd.to_datetime(df["close_time"], utc=True).astype("int64") // 10**9
    return df[["ticker", "floor_strike", "close_ts", "result"]]

def market_edges(ticker, strike, close, grid, origin):
    open_ = close - 900
    prices = window_values(grid, origin, open_ - SIGMA_WINDOW, close)
    if prices is None:
        return None
    fair = np.empty(900)
    for i in range(900):
        hist = prices[i : SIGMA_WINDOW + i]
        sigma = float(np.std(np.diff(hist), ddof=1))
        s_now = prices[SIGMA_WINDOW + i]
        left = 900 - i
        if left > 60:
            fair[i] = fair_value_up(s_now, strike, sigma, left)
        else:
            obs = prices[SIGMA_WINDOW + 840 : SIGMA_WINDOW + i]
            fair[i] = fair_value_up(s_now, strike, sigma, left,
                                    observed_sum=obs.sum(), observed_count=len(obs))
    t = fetch_trades(ticker)          # no CSV cache: too much data
    if t.empty:
        return None
    t["sec"] = (t["created_time"].astype("int64") // 10**9) - open_
    t = t[(t["sec"] >= 0) & (t["sec"] < 900)]
    t = t.assign(fair=fair[t["sec"].to_numpy()])
    t = t.assign(edge=t["yes_price"] - t["fair"], ticker=ticker)
    return t[["ticker", "sec", "fair", "yes_price", "edge", "count", "taker_side"]]

if __name__ == "__main__":
    from pathlib import Path
    grid, origin = price_grid(paths)
    start = int(pd.Timestamp("2026-10-21", tz="UTC").timestamp())
    end = int(pd.Timestamp("2026-11-01", tz="UTC").timestamp())
    markets = list_markets(min_close=start, max_close=end)
    print(len(markets), "markets")

    out = Path("data/edges"); out.mkdir(parents=True, exist_ok=True)
    done = {p.stem for p in out.glob("*.parquet")}
    batch = []
    for k, m in enumerate(markets.itertuples()):
        if m.ticker in done:
            continue
        e = market_edges(m.ticker, m.floor_strike, m.close_ts, grid, origin)
        if e is not None:
            e.to_parquet(out / f"{m.ticker}.parquet")
        if k % 50 == 0:
            print(k, m.ticker)
            