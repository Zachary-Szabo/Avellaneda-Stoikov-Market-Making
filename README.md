# mm-sim: Avellaneda-Stoikov market making

Week 1: reproduce Avellaneda & Stoikov (2008), "High-frequency trading in a limit order book", on simulated prices.

## Model
- Reservation price: r = s - q·γ·σ²·(T - t)
- Total spread: γ·σ²·(T - t) + (2/γ)·ln(1 + γ/k)
- Fill chance per step for a quote δ from the mid: A·e^(-kδ)·dt
- State (s, x, q, t) changes each step; settings (σ, γ, k, A, T, dt, s0) live in `mm/params.py`.

## Run
    python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
    pip install -e ".[dev]"
    pytest
    python run_experiments.py   # prints the table and writes figures/

## Results (1,000 sessions, paper parameters)
| Strategy | Spread | Mean P&L | Std P&L | Std final q |
|---|---|---|---|---|
| Inventory | 1.49 | 64.71 | 6.58 | 3.01 |
| Symmetric | 1.49 | 67.92 | 13.55 | 8.60 |

The inventory strategy gives up about 3 in mean P&L to cut P&L risk roughly in half, matching the paper.
