import numpy as np

from .test_common import check_init_discriminant


def test_relative_distance():
    rel_dist_class = check_init_discriminant("relative-distance")

    rel_dist = rel_dist_class()

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

    # Potentially raising errors when division by 0.0 or inf happens....

    # Potentially write gradient test cases...
    #   Check if same reverse of diff? Not sure...
