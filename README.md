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

## Inventory caps

The inventory-aware strategy already pulls its quotes back toward zero inventory, but I wanted to see what happens if I also put a hard limit on how much inventory it can hold. With a cap of N, the bot stops quoting the bid when it's already long N and stops quoting the ask when it's short N. I ran 1,000 sessions per setting with the same seed, so the only thing that changes between rows is the cap.

| Max inventory | Avg spread | Mean P&L | Std P&L | Std final inventory | Max abs inventory |
|---------------|-----------|----------|---------|---------------------|-------------------|
| None          | 1.49      | 65.04    | 6.63    | 2.93                | 11                |
| 10            | 1.49      | 65.04    | 6.63    | 2.92                | 10                |
| 5             | 1.49      | 64.89    | 6.62    | 2.62                | 5                 |
| 3             | 1.49      | 63.86    | 6.50    | 1.91                | 3                 |

**What I found:**
- A cap of 10 barely matters. It only clips the one session that got to 11.
- Tighter caps shrink inventory risk a lot. Std of final inventory drops from 2.93 to 1.91 at a cap of 3.
- It costs very little. Mean P&L only falls about 2% (65.04 to 63.86) at the tightest cap, because I skip some fills.
- P&L risk barely changes. The inventory-aware quoting already keeps P&L swings low, so the cap works more like a safety limit on the worst-case position than a way to smooth returns.

**Caveat:** this is one seed. The differences in mean P&L between caps (around 1 point) are small next to the std of about 6.6, so I wouldn't say one cap is better on P&L. The max inventory and inventory std differences are the clear ones.
