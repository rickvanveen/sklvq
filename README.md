[![ReadTheDocs](https://readthedocs.org/projects/sklvq/badge/?version=latest)](https://sklvq.readthedocs.io/en/latest/?badge=latest)
[![CodeCov](https://codecov.io/gh/rickvanveen/sklvq/branch/develop/graph/badge.svg)](https://codecov.io/gh/rickvanveen/sklvq)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![CodeFactor](https://www.codefactor.io/repository/github/rickvanveen/sklvq/badge/develop)](https://www.codefactor.io/repository/github/rickvanveen/sklvq/overview/develop)
[![CodeQL](https://github.com/rickvanveen/sklvq/workflows/CodeQL/badge.svg)](https://github.com/rickvanveen/sklvq/actions?query=workflow%3ACodeQL)

# sklvq: Scikit Learning Vector Quantization

Documentation with install instruction and examples can be found here: https://sklvq.readthedocs.io/en/latest/

If you find any bugs or have any other suggestions on how to improve the code then please use the issue tracker on
github (https://github.com/rickvanveen/sklvq/issues).

## Tests

```sh
python -m pip install -e '.[tests]'
python -m pytest
```

Default collection is restricted to `tests/` (including doctests).

### Known distance defects

Correctness tests use explicit examples with hand-calculated expected distances
and missing-input gradients. Finite-input gradients are checked by central finite
differences of raw prototype/relevance parameters, without normalization.
Euclidean and squared-Euclidean tests share parametrized examples; adaptive
distances have dedicated tests for relevance factors and localization.
Distance tests use lightweight `SimpleNamespace` fixtures
containing only the state and methods distances read, including the real models'
static relevance-matrix computation methods. Existing model-level tests
separately cover real model initialization, fitting and integration.
The distance suite focuses on internal numerical behavior: fixed answers, gradient
ordering, local matrix selection, rectangular factors, missing observations and
preservation of caller-owned arrays after each distance/gradient call. Local tests
also cover one-sample batches used by stochastic training. It does not duplicate
model input validation or a general-purpose metric API test suite.
Strict expected failures record these existing defects; an unexpected pass
requires updating the regression test:

- `Euclidean` and `SquaredEuclidean` with `force_all_finite="allow-nan"` square
  the sum of observed differences rather than sum their squares. For differences
  `[3, -2, NaN]`, they return `1` instead of `sqrt(13)` and `13`, respectively.
- Both local relevance modes square the NaN quadratic distance a second time:
  the same differences with identity relevance give `169` instead of `13`.
- Prototype-local gradients select omega by class label instead of prototype
  index. Distinct matrices and multiple prototypes per class expose this.
- Global and local adaptive gradients can be nonzero for missing input
  coordinates when relevance has off-diagonal terms, although the masked
  distance is independent of those prototype coordinates.

The Euclidean gradient docstring also describes the squared-distance derivative.
These implementation/documentation defects are left for a separate PR.

Please cite the following publication if you decide using the code in your research

```
@article{vanVeen2021,
    author  = {Rick van Veen and Michael Biehl and Gert-Jan de Vries},
    title   = {sklvq: Scikit Learning Vector Quantization},
    journal = {Journal of Machine Learning Research},
    year    = {2021},
    volume  = {22},
    number  = {231},
    pages   = {1-6},
    url     = {http://jmlr.org/papers/v22/21-0029.html}
}
```
