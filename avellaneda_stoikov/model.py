"""Closed-form Avellaneda-Stoikov (2008) quoting rules. Pure functions, no state."""
import math

from .params import Params


def reservation_price(s: float, q: float, t: float, p: Params) -> float:
    """r = s - q * gamma * sigma^2 * (T - t). Long inventory pushes r below the mid."""
    tau = max(p.T - t, 0.0)
    return s - q * p.gamma * p.sigma**2 * tau


def optimal_spread(t: float, p: Params) -> float:
    """Total spread delta_a + delta_b = gamma*sigma^2*(T-t) + (2/gamma)*ln(1 + gamma/k).

    As gamma -> 0 the second term tends to 2/k, so we use that limit for tiny gamma
    to avoid dividing by ~0.
    """
    tau = max(p.T - t, 0.0)
    if p.gamma < 1e-12:
        return 2.0 / p.k
    return p.gamma * p.sigma**2 * tau + (2.0 / p.gamma) * math.log1p(p.gamma / p.k)


def fill_probability(delta: float, p: Params) -> float:
    """Chance a quote `delta` away from the mid fills in one step: A * exp(-k*delta) * dt, capped at 1."""
    return min(1.0, p.A * math.exp(-p.k * delta) * p.dt)
