import numpy as np

from sklvq import discriminants
from sklvq.discriminants.relative_distance import RelativeDistance


def test_relative_distance():
    rel_dist = discriminants.grab("relative-distance", None)
    # Test if grab returns the correct class
    assert isinstance(rel_dist, RelativeDistance)

    # Needs to be negative when dist_diff is bigger. Positive when dist_same is bigger and 0 if
    # they are the same:

    rng = np.random.RandomState(0)
    x = rng.random_sample((100, 1))
    y = rng.random_sample((100, 1))

    # Positive when x > y
    assert np.all((x > y) == (rel_dist(x, y) > 0))

    # Negative when y > x
    assert np.all((y > x) == (rel_dist(x, y) < 0))

    # All the same
    assert np.all((rel_dist(x, x) == 0.0) & (rel_dist(y, y) == 0.0))
