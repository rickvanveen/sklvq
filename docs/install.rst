======================
Install and Contribute
======================

Install sklvq
=============

To use the published package in a uv project, run::

    uv add sklvq

To install it in an active Python environment with pip, run::

    python -m pip install sklvq

Develop from source
===================

Install `uv <https://docs.astral.sh/uv/getting-started/installation/>`_, then
clone the repository and create the project environment::

    git clone https://github.com/rickvanveen/sklvq.git
    cd sklvq
    uv sync --locked

Run the commands below from the repository root. uv uses Python 3.12 by
default and installs dependencies from ``uv.lock`` into ``.venv``.
You do not need to activate the environment.

Run an example
--------------

To fit a GLVQ classifier on the Iris dataset and plot its prototypes, run::

    uv run --locked --extra docs python examples/01_basic_usage/plot_01_glvq.py

The ``docs`` extra includes Matplotlib for the gallery examples.

Run tests and checks
--------------------

Run the full test suite on Python 3.12, or select Python 3.11 explicitly::

    uv run --locked --extra tests pytest tests/
    uv run --locked --python 3.11 --extra tests pytest tests/

To run part of the suite, replace ``tests/`` with a directory or test file,
such as ``tests/activations/``. Check lint and formatting with Ruff::

    uv run --locked ruff check src/sklvq/ examples/ tests/
    uv run --locked ruff format --check src/sklvq/ examples/ tests/

Build the documentation
-----------------------

Build the HTML pages and execute the gallery examples with Sphinx::

    uv run --locked --extra docs sphinx-build -b html docs docs/_build/html

Open ``docs/_build/html/index.html`` in a browser. To render equations locally,
install LaTeX and ``dvipng``.
