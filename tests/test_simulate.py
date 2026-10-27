import numpy as np

from mm.params import Params
from mm.simulate import run_many
from mm.strategies import InventoryStrategy, SymmetricStrategy

P = Params()


def test_same_seed_is_reproducible():
    a = run_many(InventoryStrategy(), P, 5, seed=42)
    b = run_many(InventoryStrategy(), P, 5, seed=42)
    assert [r.pnl for r in a] == [r.pnl for r in b]


def test_inventory_strategy_has_lower_risk_than_baseline():
    inv = run_many(InventoryStrategy(), P, 300, seed=1)
    sym = run_many(SymmetricStrategy(), P, 300, seed=1)
    assert np.std([r.pnl for r in inv]) < np.std([r.pnl for r in sym])
    assert np.std([r.final_q for r in inv]) < np.std([r.final_q for r in sym])
