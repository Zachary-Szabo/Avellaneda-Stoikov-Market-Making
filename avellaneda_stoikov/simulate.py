"""Monte Carlo simulation of one or many market-making sessions."""
import math
from dataclasses import dataclass

import numpy as np

from .model import fill_probability
from .params import Params
from .strategies import Strategy


@dataclass
class SessionResult:
    pnl: float
    final_q: int
    n_trades: int
    avg_spread: float
    max_q: int

# Simulates one session. Each step: (1) get our bid and ask from the strategy,
# (2) roll for a fill on each side using fill_probability, (3) a bid fill buys one
# (cash down, inventory +1) and an ask fill sells one (cash up, inventory -1),
# (4) move the mid price by a random Brownian step. Final P&L is cash plus
# inventory valued at the last mid price.
def run_session(strategy: Strategy, p: Params, rng: np.random.Generator) -> SessionResult:
    s, x, q = p.s0, 0.0, 0
    n_trades, spread_sum = 0, 0.0
    max_abs_q: int
    for i in range(p.n_steps):
        t = i * p.dt
        bid, ask = strategy.quotes(s, q, t, p)
        spread_sum += ask - bid

        # Distances are measured from the mid, as in the paper.
        delta_b, delta_a = s - bid, ask - s
        if rng.random() < fill_probability(delta_b, p):   # someone sells to us
            q += 1
            x -= bid
            n_trades += 1
        if rng.random() < fill_probability(delta_a, p):   # someone buys from us
            q -= 1
            x += ask
            n_trades += 1
        max_abs_q = max(max_abs_q, abs(q))
        # Price moves after quotes are set (arithmetic Brownian motion, no drift).
        s += p.sigma * math.sqrt(p.dt) * rng.standard_normal()

    return SessionResult(pnl=x + q * s, final_q=q, n_trades=n_trades,max_q= max_abs_q,avg_spread=spread_sum / p.n_steps)


def run_many(strategy: Strategy, p: Params, n: int, seed: int = 0) -> list[SessionResult]:
    rng = np.random.default_rng(seed)
    return [run_session(strategy, p, rng) for _ in range(n)]
