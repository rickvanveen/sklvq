import pytest
import numpy as np

from sklvq import discriminants
from sklvq.discriminants.relative_distance import RelativeDistance



# Check if grab returns correct class, defaults of init (if any), basic workings are correct.
def test_relative_distance():
    rel_dist = discriminants.grab("relative-distance", None)
    # Test if grab returns the correct class
    assert isinstance(rel_dist, RelativeDistance)

