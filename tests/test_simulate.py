import numpy as np

from avellaneda_stoikov.params import Params
from avellaneda_stoikov.simulate import run_many
from avellaneda_stoikov.strategies import InventoryStrategy, SymmetricStrategy

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

def test_max_q_exceeds():
    q_test = run_many(InventoryStrategy(), P, 200, seed=1)
    assert max(r.max_abs_q for r in q_test) > 5

##Should fail for now to verify future changes
def test_capped_inventory():
    q_test = run_many(InventoryStrategy(), P, 200, seed=1, max_inventory=5)
    assert max(r.max_abs_q for r in q_test) <= 5