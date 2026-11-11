"""Probability that a 15-minute window settles Up under arithmetic Brownian motion."""
import math


def _phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _prob_at_least(mean: float, variance: float, target: float) -> float:
    if variance <= 0.0:
        return 1.0 if mean >= target else 0.0
    return _phi((mean - target) / math.sqrt(variance))


def fair_value_up(s_now, target, sigma_per_sec, seconds_left, observed_sum=0.0, observed_count=0):
    """Probability that the mean of the 60 one-second prices before the close is >= target.

    Price is a driftless Brownian motion: S_{t+h} = s_now + sigma_per_sec * W_h.
    The closing mean uses the seconds S_{T-60}, ..., S_{T-1}.

    With more than 60 seconds left, none of those prices are known. The mean is
    modeled as Normal(s_now, sigma^2 * (seconds_left - 40)). That variance is
    the continuous average (1/60) ∫_{T-60}^{T} S_u du. With a = seconds_left - 60
    and b = seconds_left, Var(∫_a^b W_u du) = b^3/3 - a^2 b + 2 a^3/3
    = 3600 * (seconds_left - 40), so the average has variance sigma^2 * (seconds_left - 40).

    With seconds_left <= 60, observed_count of the 60 prices are already fixed
    and add up to observed_sum. The other r = 60 - observed_count prices are the
    integer horizons a, a+1, ..., a+r-1, where a = seconds_left - r. The case
    used in calibration is a = 0: strictly past seconds are in observed_sum, and
    the current second is horizon 0. Because Cov(W_h, W_k) = min(h, k),

        Var(sum_i S_{h_i}) / sigma^2 = sum_i sum_j min(a+i, a+j)
            = r^2 * a + sum_{i,j < r} min(i, j)
            = r^2 * a + r (r - 1) (2 r - 1) / 6.

    The sum of min(i, j) counts, for each k >= 1, the (r - k)^2 pairs whose min
    is at least k, which is sum_{m=1}^{r-1} m^2. Horizon 0 adds nothing. The
    future sum has mean r * s_now, and the closing mean divides that sum by 60,
    so its variance is the expression above times sigma^2 / 3600.
    """
    if seconds_left > 60:
        variance = sigma_per_sec ** 2 * (seconds_left - 40.0)
        return _prob_at_least(s_now, variance, target)

    remaining = 60 - observed_count
    if remaining < 0:
        raise ValueError("observed_count cannot exceed 60")
    if remaining == 0:
        return _prob_at_least(observed_sum / 60.0, 0.0, target)

    # No time left for a path. Every unsettled second is stuck at the current price.
    first_horizon = seconds_left - remaining
    if first_horizon < 0:
        mean = (observed_sum + remaining * s_now) / 60.0
        return _prob_at_least(mean, 0.0, target)

    var_factor = remaining ** 2 * first_horizon + remaining * (remaining - 1) * (2 * remaining - 1) / 6.0
    mean = (observed_sum + remaining * s_now) / 60.0
    variance = sigma_per_sec ** 2 * var_factor / 3600.0
    return _prob_at_least(mean, variance, target)
