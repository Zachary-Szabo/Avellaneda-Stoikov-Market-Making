import numpy as np
import pandas as pd
from pathlib import Path
from fetch_kalshi_trades import list_markets

files = sorted(Path("data/edges").glob("*.parquet"))
cols = ["ticker", "fair", "yes_price", "edge", "count"]
df = pd.concat([pd.read_parquet(f, columns=cols) for f in files], ignore_index=True)
print(f"{len(files)} markets, {len(df):,} trades")
W_GRID = np.round(np.arange(0.0, 1.01, 0.05), 2)


# Taker fee benchmark (makers pay 0 on KXBTC15M as of Nov 2026)
df["taker_fee"] = 0.07 * df["yes_price"] * (1 - df["yes_price"])
df["bucket"] = pd.cut(df["fair"], bins=np.arange(0, 1.01, 0.1))
def summarize(g):
    w = g["count"]
    return pd.Series({
        "n_trades": len(g),
        "volume": w.sum(),
        "vw_edge": np.average(g["edge"], weights=w),
        "vw_taker_fee": np.average(g["taker_fee"], weights=w),
    })
table = df.groupby("bucket", observed=True)[["edge", "taker_fee", "count"]].apply(summarize)
print(table.round(4))


print("\noverall vw edge:", round(np.average(df["edge"], weights=df["count"]), 4))

# Is it consistent across markets, or driven by a few?
per_mkt = df.groupby("ticker").apply(
    lambda g: np.average(g["edge"], weights=g["count"]), include_groups=False)
print("\nper-market edge: mean", round(per_mkt.mean(), 4),
      "| median", round(per_mkt.median(), 4),
      "| % positive", round((per_mkt > 0).mean() * 100, 1))

start = int(pd.Timestamp("2026-10-21", tz="UTC").timestamp())
end = int(pd.Timestamp("2026-11-01", tz="UTC").timestamp())
mkts = list_markets(min_close=start, max_close=end)
outcome = (mkts.set_index("ticker")["result"] == "yes").astype(float)

df["y"] = df["ticker"].map(outcome)
df["close_ts"] = df["ticker"].map(mkts.set_index("ticker")["close_ts"])
df = df.dropna(subset=["y"])

df["brier_mkt"] = (df["yes_price"] - df["y"]) ** 2
df["brier_fair"] = (df["fair"] - df["y"]) ** 2

def brier(g):
    w = g["count"]
    return pd.Series({
        "mkt": np.average(g["brier_mkt"], weights=w),
        "fair": np.average(g["brier_fair"], weights=w),
    })

bt = df.groupby("bucket", observed=True)[["brier_mkt", "brier_fair", "count"]].apply(brier)
bt["fair_minus_mkt"] = bt["fair"] - bt["mkt"]   # negative = your model wins
print(bt.round(5))
print("\noverall  mkt:", round(np.average(df["brier_mkt"], weights=df["count"]), 5),
      "| fair:", round(np.average(df["brier_fair"], weights=df["count"]), 5))

# ---------- Blend: w*model + (1-w)*market --------


def vw_brier(p, y, wts):
    return float(np.average((p - y) ** 2, weights=wts))

def blend_sweep(df, train_frac=0.7):
    order = df.groupby("ticker")["close_ts"].first().sort_values()
    cut = int(len(order) * train_frac)
    train_t = set(order.index[:cut])
    train = df[df["ticker"].isin(train_t)]
    test = df[~df["ticker"].isin(train_t)]

    rows = []
    for w in W_GRID:
        rows.append({
            "w": w,
            "train": vw_brier(w * train["fair"] + (1 - w) * train["yes_price"],
                              train["y"], train["count"]),
            "test": vw_brier(w * test["fair"] + (1 - w) * test["yes_price"],
                             test["y"], test["count"]),
        })
    res = pd.DataFrame(rows)
    best_w = res.loc[res["train"].idxmin(), "w"]
    print(res.to_string(index=False, float_format="%.5f"))
    print(f"\nbest w on train: {best_w}")
    print(f"test Brier at best w: {res.loc[res['w'] == best_w, 'test'].item():.5f}")
    print(f"test Brier market (w=0): {res.loc[res['w'] == 0, 'test'].item():.5f}")
    print(f"test Brier model  (w=1): {res.loc[res['w'] == 1, 'test'].item():.5f}")
    print(f"markets: train={cut}, test={len(order) - cut}")
    return res

##blend_sweep(df)

# ---------- Walk-forward: expanding window, one test day at a time ----------
def walk_forward(df, min_train_days=3):
    df = df.copy()
    df["day"] = pd.to_datetime(df["close_ts"], unit="s", utc=True).dt.date
    days = sorted(df["day"].unique())

    rows = []
    for i in range(min_train_days, len(days)):
        train = df[df["day"] < days[i]]
        test = df[df["day"] == days[i]]

        train_scores = [vw_brier(w * train["fair"] + (1 - w) * train["yes_price"],
                                 train["y"], train["count"]) for w in W_GRID]
        best_w = W_GRID[int(np.argmin(train_scores))]

        def tb(w):
            return vw_brier(w * test["fair"] + (1 - w) * test["yes_price"],
                            test["y"], test["count"])

        rows.append({
            "test_day": days[i],
            "n_mkts": test["ticker"].nunique(),
            "best_w": best_w,
            "mkt": tb(0.0),
            "model": tb(1.0),
            "blend": tb(best_w),
        })

    wf = pd.DataFrame(rows)
    wf["blend_minus_mkt"] = wf["blend"] - wf["mkt"]   # negative = blend wins
    print(wf.to_string(index=False, float_format="%.5f"))
    print(f"\ndays blend beat market: {(wf['blend_minus_mkt'] < 0).sum()} / {len(wf)}")
    print(f"mean blend - mkt: {wf['blend_minus_mkt'].mean():.5f}")
    return wf

walk_forward(df)