"""Closed-form Avellaneda-Stoikov (2008) quoting rules. Pure functions, no state."""
import math

from .params import Params

# The center of our bid and ask. It starts at the mid price and shifts away from
# our inventory.
# When we're long (q > 0) it drops below the mid, so our ask is
# easier to fill and our bid harder, pulling inventory back toward 0. The shift
# grows with risk aversion (gamma), volatility (sigma^2), and time left (T - t).
def reservation_price(s: float, q: float, t: float, p: Params) -> float:
    """r = s - q * gamma * sigma^2 * (T - t). Long inventory pushes r below the mid."""
    tau = max(p.T - t, 0.0)
    return s - q * p.gamma * p.sigma**2 * tau

# Total width between our bid and ask, centered on the reservation price.
# It has two parts: an inventory risk term, gamma * sigma^2 * (T - t)
# That shrinks as the session ends, and an order-flow term, (2/gamma) * ln(1 + gamma/k),
# which reflects how fast fills fall off as we quote wider. It doesn't depend on q.
def optimal_spread(t: float, p: Params) -> float:
    """Total spread delta_a + delta_b = gamma*sigma^2*(T-t) + (2/gamma)*ln(1 + gamma/k).

    As gamma -> 0 the second term tends to 2/k, so we use that limit for tiny gamma
    to avoid dividing by ~0.
    """
    tau = max(p.T - t, 0.0)
    if p.gamma < 1e-12:
        return 2.0 / p.k
    return p.gamma * p.sigma**2 * tau + (2.0 / p.gamma) * math.log1p(p.gamma / p.k)

# Chance that a quote delta away from the MID fills in one time step:
# A * exp(-k * delta) * dt, capped at 1. Quotes closer to the mid fill more
# often, and the chance falls off exponentially as we quote wider.
def fill_probability(delta: float, p: Params) -> float:
    """Chance a quote `delta` away from the mid fills in one step: A * exp(-k*delta) * dt, capped at 1."""
    return min(1.0, p.A * math.exp(-p.k * delta) * p.dt)
