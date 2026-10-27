import math
from dataclasses import replace

import pytest

from mm.model import fill_probability, optimal_spread, reservation_price
from mm.params import Params

P = Params()


def test_flat_inventory_reservation_equals_mid():
    assert reservation_price(100.0, 0, 0.3, P) == pytest.approx(100.0)


def test_long_inventory_shades_below_mid_and_short_above():
    assert reservation_price(100.0, 5, 0.0, P) < 100.0
    assert reservation_price(100.0, -5, 0.0, P) > 100.0


def test_reservation_matches_formula():
    # 100 - 3 * 0.1 * 4 * (1 - 0.25) = 99.1
    assert reservation_price(100.0, 3, 0.25, P) == pytest.approx(99.1)


def test_inventory_effect_vanishes_at_expiry():
    assert reservation_price(100.0, 10, P.T, P) == pytest.approx(100.0)


def test_spread_shrinks_to_fill_term_at_expiry():
    expected = (2 / P.gamma) * math.log(1 + P.gamma / P.k)
    assert optimal_spread(P.T, P) == pytest.approx(expected)
    assert optimal_spread(0.0, P) > optimal_spread(0.9, P)


def test_spread_tends_to_2_over_k_as_gamma_goes_to_zero():
    for g in (1e-6, 1e-9, 0.0):
        assert optimal_spread(0.0, replace(P, gamma=g)) == pytest.approx(2 / P.k, rel=1e-4)


def test_fill_probability_decreases_with_distance_and_is_capped():
    assert fill_probability(0.1, P) > fill_probability(1.0, P)
    assert fill_probability(-100.0, P) == 1.0
