#!/usr/bin/env bash

# TravisCI default virtualenv
python --version
pip install codecov
pip install -e ".[tests]"

# Version conflict error resolved?
pip install --upgrade pytest

