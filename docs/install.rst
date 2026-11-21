======================
Install and Contribute
======================

This page containst the lists of dependencies and install instructions for two scenarios. Firstly, "General use" for anyone who just whishes to use the algorithms. Secondly, instruction on how to install the package in order to be able to build the docs and run tests.

General Use
===========

This section contains the dependencies and install instructions for regular usage of the sklvq package. If you wish to contribute to the package please see the `Contribute`_ section.

Dependencies
------------

The sklvq toolbox requires the following packages to be installed:

* numpy
* scipy
* scikit-learn

Installation
------------
Sklvq can be installed using pip::

    pip install sklvq

Another option is to clone the repository and install it with pip::

    git clone https://github.com/rickvanveen/sklvq.git
    cd sklvq
    pip install .

Or install using pip from GitHub directly::

    pip install -U git+https://github.com/rickvanveen/sklvq.git

Contribute
==========

You can contribute to this code through pull requests on GitHub. Please, make sure that your code is coming with unit tests to ensure full coverage and continuous integration in the API. Follow the instruction below in order to install all necessary dependencies for development.

Dependencies
------------

In addition to the regular dependencies, sklvq requires a number of packages for testing and building the documentation:

Testing:
    * pytest
    * pytest-cov

Documentation:
    * sphinx
    * sphinx-gallery
    * sphinx_rtd_theme
    * numpydoc
    * matplotlib

Installation
------------

The package can be cloned using the following commands::

    git clone https://github.com/rickvanveen/sklvq.git
    cd sklvq

Install `uv <https://docs.astral.sh/uv/getting-started/installation/>`_, then
create the development environment from the committed lockfile::

    uv sync --locked --extra tests --extra docs

This installs the package, test and documentation dependencies, and Ruff.
The default development Python is 3.12; CI also tests Python 3.11.
After changing dependencies, run ``uv lock`` and commit ``uv.lock``.


Running Tests
-------------

Run tests from the repository root::

    uv run --locked --extra tests pytest tests/

Check lint and formatting without changing files::

    uv run --locked ruff check src/sklvq/ examples/ tests/
    uv run --locked ruff format --check src/sklvq/ examples/ tests/

Building Docs
-------------

Build HTML documentation and execute the gallery examples from the repository root::

    uv run --locked --extra docs make -C docs html

This will generate a build folder from which the index.html can be opened locally. Other options
are also available see the 'Makefile' in the doc folder.
