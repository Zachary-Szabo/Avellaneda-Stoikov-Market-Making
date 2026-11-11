from fair_value import fair_value_up


def test_at_the_target_with_time_left_is_even():
    probability = fair_value_up(100.0, 100.0, sigma_per_sec=1.5, seconds_left=600)
    assert abs(probability - 0.5) < 1e-12


def test_large_positive_gap_is_near_one():
    probability = fair_value_up(200.0, 100.0, sigma_per_sec=1.0, seconds_left=600)
    assert probability > 0.99


def test_probability_increases_with_price():
    probabilities = [fair_value_up(price, 100.0, 2.0, 300) for price in (90.0, 100.0, 110.0, 130.0)]
    assert probabilities == sorted(probabilities)
    assert probabilities[0] < probabilities[-1]


def test_probability_stays_in_range_near_expiry():
    cases = [
        fair_value_up(100.0, 100.0, 1.0, 600),
        fair_value_up(80.0, 100.0, 1.0, 120),
        fair_value_up(150.0, 100.0, 1.0, 61),
        fair_value_up(100.0, 100.0, 1.0, 60),
        fair_value_up(110.0, 100.0, 1.0, 1, observed_sum=100.0 * 59, observed_count=59),
        fair_value_up(100.0, 100.0, 1.0, 0, observed_sum=6000.0, observed_count=60),
        fair_value_up(100.0, 101.0, 1.0, 0, observed_sum=6000.0, observed_count=60),
        fair_value_up(100.0, 100.0, 1.0, 1e-6),
    ]
    assert all(0.0 <= probability <= 1.0 for probability in cases)
    assert fair_value_up(100.0, 99.0, 1.0, 0, observed_sum=6000.0, observed_count=60) == 1.0
    assert fair_value_up(100.0, 101.0, 1.0, 0, observed_sum=6000.0, observed_count=60) == 0.0
