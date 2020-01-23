#!/usr/bin/env bash

# TravisCI default virtualenv
python --version
pip install codecov
pip install -e ".[tests]"

