[![Read the Docs](https://app.readthedocs.org/projects/sklvq/badge/?version=latest)](https://sklvq.readthedocs.io/en/latest/)
[![CodeCov](https://codecov.io/gh/rickvanveen/sklvq/branch/main/graph/badge.svg)](https://codecov.io/gh/rickvanveen/sklvq)
[![CI](https://github.com/rickvanveen/sklvq/actions/workflows/pull-request.yml/badge.svg?branch=main)](https://github.com/rickvanveen/sklvq/actions/workflows/pull-request.yml)
[![CodeFactor](https://www.codefactor.io/repository/github/rickvanveen/sklvq/badge/main)](https://www.codefactor.io/repository/github/rickvanveen/sklvq/overview/main)
[![CodeQL](https://github.com/rickvanveen/sklvq/actions/workflows/codeql.yml/badge.svg?branch=main)](https://github.com/rickvanveen/sklvq/actions/workflows/codeql.yml)

# sklvq: Scikit Learning Vector Quantization

Documentation with install instruction and examples can be found here: https://sklvq.readthedocs.io/en/latest/

If you find any bugs or have any other suggestions on how to improve the code then please use the issue tracker on
github (https://github.com/rickvanveen/sklvq/issues).

## Tests

```sh
uv run --locked --extra tests pytest tests/
```

### Distance tests

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
Regression tests cover missing-value distance formulas, prototype-local matrix
selection and zero prototype updates for missing coordinates.

Known documentation issue: the Euclidean gradient docstring describes the
squared-distance derivative.

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
