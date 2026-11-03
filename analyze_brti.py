"""Summarize daily BRTI CSVs: 1-second volatility and 15-minute settlement.

Each file in data/brti/YYYY-MM-DD.csv has `time` in unix milliseconds UTC and
`value` as a string. Usage: python analyze_brti.py
"""
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_DIR = Path("data/brti")
OUT = Path("figures")
DAY_FILE = re.compile(r"^\d{4}-\d{2}-\d{2}\.csv$")
SECONDS_PER_DAY = 86_400
FILL_LIMIT = 5
EASTERN = "America/New_York"


def load_day(path: Path) -> pd.Series:
    frame = pd.read_csv(path)
    frame["time"] = pd.to_numeric(frame["time"], errors="coerce")
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame.dropna(subset=["time", "value"])
    frame["time"] = frame["time"].astype(np.int64)
    frame = frame.sort_values("time").drop_duplicates("time", keep="last")

    last = frame.groupby(frame["time"] // 1000, sort=True)["value"].last()
    start = int(pd.Timestamp(path.stem, tz="UTC").timestamp())
    index = np.arange(start, start + SECONDS_PER_DAY, dtype=np.int64)
    series = last.reindex(index).ffill(limit=FILL_LIMIT)
    series.index.name = "second"
    return series


def second_returns(prices: pd.Series) -> pd.Series:
    values = prices.to_numpy(dtype=float)
    returns = np.full(len(values), np.nan)
    previous = values[:-1]
    current = values[1:]
    ok = np.isfinite(previous) & np.isfinite(current) & (previous > 0) & (current > 0)
    step = np.full(len(values) - 1, np.nan)
    step[ok] = np.log(current[ok] / previous[ok])
    returns[1:] = step
    return pd.Series(returns, index=prices.index, name="log_return")


def finite(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values)]


def sample_std(values: np.ndarray) -> float:
    kept = finite(values)
    if len(kept) < 2:
        return float("nan")
    return float(np.std(kept, ddof=1))


def realized_vol(values: np.ndarray) -> float:
    kept = finite(values)
    if len(kept) == 0:
        return float("nan")
    return float(np.sqrt(np.sum(kept * kept)))


def vol_by_hour(returns: pd.Series, tz: str) -> pd.Series:
    stamps = pd.to_datetime(returns.index, unit="s", utc=True).tz_convert(tz)
    frame = pd.DataFrame({
        "r": returns.to_numpy(dtype=float),
        "date": stamps.date,
        "hour": stamps.hour,
    })
    frame = frame[np.isfinite(frame["r"])]
    vols = []
    for (date, hour), group in frame.groupby(["date", "hour"], sort=True):
        vols.append({"date": date, "hour": int(hour), "vol": sample_std(group["r"].to_numpy()) * np.sqrt(900)})
    if not vols:
        return pd.Series(np.nan, index=range(24))
    averaged = pd.DataFrame(vols).groupby("hour")["vol"].mean()
    return averaged.reindex(range(24))


def settlement_stats(prices: pd.Series) -> dict:
    seconds = prices.index.to_numpy(dtype=np.int64)
    values = prices.to_numpy(dtype=float)
    start = int(seconds[0])
    end = int(seconds[-1])
    grid = np.full(end - start + 1, np.nan)
    grid[seconds - start] = values

    first = start + (900 - start % 900) % 900
    moves = []
    above = 0
    used = 0
    skipped_edge = 0
    skipped_gap = 0
    opening = first
    while opening + 899 <= end:
        offset = opening - start
        if offset < 60:
            skipped_edge += 1
            opening += 900
            continue
        target = grid[offset - 60:offset]
        final = grid[offset + 840:offset + 900]
        if not (np.isfinite(target).all() and np.isfinite(final).all()):
            skipped_gap += 1
            opening += 900
            continue
        target_mean = float(target.mean())
        final_mean = float(final.mean())
        if target_mean == 0:
            skipped_gap += 1
            opening += 900
            continue
        used += 1
        above += int(final_mean >= target_mean)
        moves.append(final_mean / target_mean - 1.0)
        opening += 900

    moves = np.asarray(moves, dtype=float)
    return {
        "used": used,
        "above": above,
        "skipped_edge": skipped_edge,
        "skipped_gap": skipped_gap,
        "median_abs_pct": float(np.median(np.abs(moves)) * 100) if len(moves) else float("nan"),
        "std_move": sample_std(moves),
    }


def print_day_table(rows: list[dict]) -> None:
    print("Per day, after a 1-second last-value sample and a 5-second forward fill.")
    print("vol_1s is the std of 1-second log returns. vol_15m = vol_1s * sqrt(900).")
    print("realized_vol is sqrt(sum of squared 1-second log returns). Gaps still missing are not crossed.")
    print(f"{'date':<12} {'missing_s':>10} {'vol_1s':>12} {'vol_15m':>12} {'realized_vol':>14}")
    for row in rows:
        print(
            f"{row['date']:<12} {row['missing']:>10d} {row['vol_1s']:>12.6e} "
            f"{row['vol_15m']:>12.6f} {row['realized_vol']:>14.6f}"
        )


def print_hour_table(utc: pd.Series, eastern: pd.Series) -> None:
    print()
    print("15-minute volatility by hour of day, averaged across days.")
    print("Each hour's number is that hour's 1-second std, scaled by sqrt(900).")
    print(f"{'hour':>4} {'UTC':>12} {'Eastern':>12}")
    for hour in range(24):
        print(f"{hour:>4d} {utc.loc[hour]:>12.6f} {eastern.loc[hour]:>12.6f}")


def print_settlement(stats: dict) -> None:
    used = stats["used"]
    above = stats["above"]
    share = above / used if used else float("nan")
    print()
    print("15-minute windows (:00, :15, :30, :45 UTC).")
    print("target = mean of the 60 seconds before the open; final = mean of the 60 seconds before the close.")
    print(
        f"windows used: {used}  "
        f"(skipped {stats['skipped_edge']} opening before the sample, "
        f"{stats['skipped_gap']} with a missing second)"
    )
    print(f"finished at or above the target: {above} of {used} ({share:.1%})")
    print(f"median absolute % difference: {stats['median_abs_pct']:.4f}%")
    print(f"std of final/target - 1: {stats['std_move']:.6f}")


def save_figure(utc: pd.Series, eastern: pd.Series) -> Path:
    OUT.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    hours = np.arange(24)
    axes[0].bar(hours, utc.to_numpy(), color="C0")
    axes[1].bar(hours, eastern.to_numpy(), color="C1")
    axes[0].set_xlabel("hour (UTC)")
    axes[1].set_xlabel("hour (Eastern)")
    axes[0].set_ylabel("15-minute volatility")
    axes[0].set_title("UTC")
    axes[1].set_title("Eastern")
    for ax in axes:
        ax.set_xticks(range(0, 24, 2))
    fig.suptitle("BRTI 15-minute volatility by hour of day")
    fig.tight_layout()
    dest = OUT / "brti_vol_by_hour.png"
    fig.savefig(dest, dpi=120)
    plt.close(fig)
    return dest


def main() -> None:
    paths = sorted(path for path in DATA_DIR.glob("*.csv") if DAY_FILE.match(path.name))
    if not paths:
        raise SystemExit(f"No daily CSVs in {DATA_DIR}")

    prices = []
    returns = []
    day_rows = []
    for path in paths:
        series = load_day(path)
        rets = second_returns(series)
        vol_1s = sample_std(rets.to_numpy())
        day_rows.append({
            "date": path.stem,
            "missing": int(series.isna().sum()),
            "vol_1s": vol_1s,
            "vol_15m": vol_1s * np.sqrt(900),
            "realized_vol": realized_vol(rets.to_numpy()),
        })
        prices.append(series)
        returns.append(rets)

    combined_prices = pd.concat(prices)
    combined_prices = combined_prices[~combined_prices.index.duplicated(keep="last")].sort_index()
    combined_returns = pd.concat(returns)
    combined_returns = combined_returns[~combined_returns.index.duplicated(keep="last")].sort_index()

    utc = vol_by_hour(combined_returns, "UTC")
    eastern = vol_by_hour(combined_returns, EASTERN)
    stats = settlement_stats(combined_prices)

    print_day_table(day_rows)
    print_hour_table(utc, eastern)
    print_settlement(stats)
    dest = save_figure(utc, eastern)
    print(f"\nsaved {dest}")


if __name__ == "__main__":
    main()
