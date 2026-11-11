"""Score fair_value_up on realized BRTI 15-minute windows. Usage: python calibrate_fair_value.py"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_brti import DATA_DIR, DAY_FILE, load_day
from fair_value import fair_value_up

OUT = Path("figures")
HORIZONS = (600, 300, 120, 60)
HORIZON_NAME = {600: "10 min", 300: "5 min", 120: "2 min", 60: "1 min"}
SIGMA_WINDOW = 3600


def price_grid(paths) -> tuple[np.ndarray, int]:
    series = pd.concat([load_day(path) for path in paths])
    series = series[~series.index.duplicated(keep="last")].sort_index()
    origin = int(series.index[0])
    end = int(series.index[-1])
    grid = np.full(end - origin + 1, np.nan)
    grid[series.index.to_numpy(dtype=np.int64) - origin] = series.to_numpy(dtype=float)
    return grid, origin


def window_values(grid: np.ndarray, origin: int, start: int, stop: int):
    left = start - origin
    right = stop - origin
    if left < 0 or right > len(grid) or right < left:
        return None
    values = grid[left:right]
    if values.size != stop - start or not np.isfinite(values).all():
        return None
    return values


def forecasts(grid: np.ndarray, origin: int) -> dict[int, list[dict]]:
    scored = {horizon: [] for horizon in HORIZONS}
    end = origin + len(grid) - 1
    close = origin + (900 - origin % 900) % 900
    if close == origin:
        close += 900
    while close <= end + 1:
        final = window_values(grid, origin, close - 60, close)
        target = window_values(grid, origin, close - 960, close - 900)
        if final is None or target is None:
            close += 900
            continue
        final_mean = float(final.mean())
        target_mean = float(target.mean())
        outcome = 1.0 if final_mean >= target_mean else 0.0
        for horizon in HORIZONS:
            now = close - horizon
            spot = window_values(grid, origin, now, now + 1)
            history = window_values(grid, origin, now - SIGMA_WINDOW, now + 1)
            observed_sum, observed_count = 0.0, 0
            if spot is None or history is None or len(history) < 2:
                continue
            sigma = float(np.std(np.diff(history), ddof=1))
            if horizon > 60:
                probability = fair_value_up(float(spot[0]), target_mean, sigma, horizon)
            else:
                observed_count = 60 - horizon
                if observed_count == 0:
                    observed_sum = 0.0
                else:
                    observed = window_values(grid, origin, close - 60, now)
                    if observed is None:
                        continue
                    observed_sum = float(observed.sum())
                probability = fair_value_up(
                    float(spot[0]), target_mean, sigma, horizon,
                    observed_sum=observed_sum, observed_count=observed_count,
                )
            scored[horizon].append({"p": probability, "y": outcome, "spot": float(spot[0]), "target": target_mean, "sigma": sigma, "obs_sum": observed_sum, "obs_count": observed_count})
        close += 900
    return scored


def bucket_rows(pairs: list[tuple[float, float]]) -> list[dict]:
    probabilities = np.array([p for p, _ in pairs])
    outcomes = np.array([y for _, y in pairs])
    rows = []
    for index in range(10):
        lower = index / 10
        upper = (index + 1) / 10
        if index == 9:
            chosen = (probabilities >= lower) & (probabilities <= upper)
            label = f"[{lower:.1f}, {upper:.1f}]"
        else:
            chosen = (probabilities >= lower) & (probabilities < upper)
            label = f"[{lower:.1f}, {upper:.1f})"
        count = int(chosen.sum())
        rows.append({
            "label": label,
            "count": count,
            "mean_p": float(probabilities[chosen].mean()) if count else float("nan"),
            "frequency": float(outcomes[chosen].mean()) if count else float("nan"),
        })
    return rows

def baselines(scored: dict[int, list[dict]]) -> dict[int, dict]:
    results = {}
    total, n = 0.0, 0
    fixed_sigma = 0
    for horizon in HORIZONS:
        for r in scored[horizon]: 
            total += r["sigma"]
            n += 1
    fixed_sigma = total/n
    for horizon in HORIZONS:
        records = scored[horizon]
        if not records:
            continue
        p = np.array([r["p"] for r in records])
        y = np.array([r["y"] for r in records])
        brier_model = float(np.mean((p - y) ** 2))
        brier_const = float(np.mean((0.5 - y) ** 2))
        brier_base = float(np.mean((y.mean() - y) ** 2))
        p_fixed = np.array([
            fair_value_up(r["spot"], r["target"], fixed_sigma, horizon,observed_sum=r["obs_sum"], observed_count=r["obs_count"])
            for r in records
        ])
        brier_fixed = float(np.mean((p_fixed - y) ** 2))
        results[horizon] = {
            "model": brier_model,
            "const": brier_const,
            "base_rate": brier_base,
            "fixed_sigma": brier_fixed,
            "skill": 1 - brier_model / brier_const,
        }
    return results


def print_report(scored: dict[int, list[dict]]) -> dict[int, list[dict]]:
    results = baselines(scored)
    print(f"{'horizon':<8} {'n':>6} {'model':>8} {'const':>8} {'base':>8} {'fixed':>8} {'skill':>7}")
    tables = {}
    for horizon in HORIZONS:
        pairs = [(r["p"], r["y"]) for r in scored[horizon]]
        if horizon in results:
            b = results[horizon]
            print(f"{HORIZON_NAME[horizon]:<8} {len(pairs):>6d} {b['model']:>8.4f} {b['const']:>8.4f} "
                  f"{b['base_rate']:>8.4f} {b['fixed_sigma']:>8.4f} {b['skill']:>7.3f}")
        tables[horizon] = bucket_rows(pairs)
    for horizon in HORIZONS:
        print()
        print(f"{HORIZON_NAME[horizon]} calibration")
        print(f"{'bucket':<14} {'n':>6} {'mean_p':>8} {'frequency':>10}")
        for row in tables[horizon]:
            mean_p = "     nan" if row["count"] == 0 else f"{row['mean_p']:8.3f}"
            frequency = "       nan" if row["count"] == 0 else f"{row['frequency']:10.3f}"
            print(f"{row['label']:<14} {row['count']:>6d} {mean_p} {frequency}")
    return tables


def save_figure(tables: dict[int, list[dict]]) -> Path:
    OUT.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect")
    for horizon in HORIZONS:
        rows = [row for row in tables[horizon] if row["count"] > 0]
        ax.plot(
            [row["mean_p"] for row in rows],
            [row["frequency"] for row in rows],
            marker="o",
            label=HORIZON_NAME[horizon],
        )
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("observed frequency")
    ax.set_title("Fair-value calibration")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    dest = OUT / "calibration.png"
    fig.savefig(dest, dpi=120)
    plt.close(fig)
    return dest


def main() -> None:
    paths = sorted(path for path in DATA_DIR.glob("*.csv") if DAY_FILE.match(path.name))
    if not paths:
        raise SystemExit(f"No daily CSVs in {DATA_DIR}")
    grid, origin = price_grid(paths)
    tables = print_report(forecasts(grid, origin))
    dest = save_figure(tables)
    print(f"\nsaved {dest}")


if __name__ == "__main__":
    main()
