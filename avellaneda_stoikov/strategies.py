"""A strategy maps state to (bid, ask)."""
from typing import Protocol, Tuple

from .model import optimal_spread, reservation_price
from .params import Params


class Strategy(Protocol):
    name: str

    def quotes(self, s: float, q: float, t: float, p: Params) -> Tuple[float, float]: ...


class InventoryStrategy:
    """Avellaneda-Stoikov: quote symmetrically around the reservation price."""
    name = "inventory"

    def quotes(self, s, q, t, p):
        r = reservation_price(s, q, t, p)
        half = optimal_spread(t, p) / 2.0
        return r - half, r + half


class SymmetricStrategy:
    """Baseline: same spread, but centered on the mid and ignoring inventory."""
    name = "symmetric"

    def quotes(self, s, q, t, p):
        half = optimal_spread(t, p) / 2.0
        return s - half, s + half
