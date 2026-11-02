import numpy as np

from avellaneda_stoikov.params import Params
from avellaneda_stoikov.simulate import run_many, run_session
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


def test_capped_inventory():
    q_test = run_many(InventoryStrategy(), P, 200, seed=1, max_inventory=5)
    assert max(r.max_abs_q for r in q_test) <= 5


def test_record_path_does_not_change_results():
    seed = 7
    off = run_session(InventoryStrategy(), P, np.random.default_rng(seed), record_path=False)
    on = run_session(InventoryStrategy(), P, np.random.default_rng(seed), record_path=True)
    assert off.pnl == on.pnl
    assert off.final_q == on.final_q
    assert off.max_abs_q == on.max_abs_q
    assert on.q_path is not None
    assert len(on.q_path) == P.n_steps