"""Fixed model settings. State (s, x, q, t) lives in the simulator, not here."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Params:
    sigma: float = 2.0    # price volatility per unit time
    gamma: float = 0.1    # risk aversion
    k: float = 1.5        # how fast fill intensity decays with distance from mid
    A: float = 140.0      # fill intensity at zero distance
    T: float = 1.0        # session length
    dt: float = 0.005     # time step
    s0: float = 100.0     # starting mid price

    @property
    def n_steps(self) -> int:
        return int(round(self.T / self.dt))
