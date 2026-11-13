"""Fetch Kalshi historical trades for KXBTC15M markets and cache them to CSV."""
import time
from pathlib import Path

import pandas as pd
import requests

BASE = "https://api.elections.kalshi.com/trade-api/v2"
OUT_DIR = Path("data/kalshi_trades")


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

    df["created_time"] = pd.to_datetime(df["created_time"], utc=True)
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


if __name__ == "__main__":
    ticker = "KXBTC15M-26OCT311945-45"
    df = load_trades(ticker)
    print(f"{ticker}: {len(df)} trades")
    if not df.empty:
        print(df[["created_time", "yes_price", "count", "taker_side"]].head(10))
        print(df["yes_price"].describe())