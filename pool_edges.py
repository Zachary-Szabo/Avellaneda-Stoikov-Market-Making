import numpy as np
import pandas as pd
from pathlib import Path

files = sorted(Path("data/edges").glob("*.parquet"))
cols = ["ticker", "fair", "yes_price", "edge", "count"]
df = pd.concat([pd.read_parquet(f, columns=cols) for f in files], ignore_index=True)
print(f"{len(files)} markets, {len(df):,} trades")



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