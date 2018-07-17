#!/usr/bin/env bash
# This script was adapted from scikit-learn's install script for travis CI. It
# will create a conda environment with everything you need for development.

# License: 3-clause BSD

set -e

ENVNAME="${ENVNAME:-wolpert_devel}"

DISTRIB="${DISTRIB:-conda}"
PYTHON_VERSION="${PYTHON_VERSION:-3.6.2}"
INSTALL_MKL="${INSTALL_MKL:-true}"
NUMPY_VERSION="${NUMPY_VERSION:-1.14.2}"
SCIPY_VERSION="${SCIPY_VERSION:-1.0.0}"
PANDAS_VERSION="${PANDAS_VERSION:-0.20.3}"
SKLEARN_VERSION="${SKLEARN_VERSION:-0.19.1}"
COVERAGE=true
CHECK_PYTEST_SOFT_DEPENDENCY="${CHECK_PYTEST_SOFT_DEPENDENCY:-true}"
TEST_DOCSTRINGS="${TEST_DOCSTRINGS:-true}"


if [[ "$DISTRIB" == "conda" ]]; then
    deactivate || true

    conda update --yes conda

    TO_INSTALL="python=$PYTHON_VERSION pip pytest pytest-cov \
                numpy=$NUMPY_VERSION scipy=$SCIPY_VERSION scikit-learn=$SKLEARN_VERSION"

    if [[ "$INSTALL_MKL" == "true" ]]; then
        TO_INSTALL="$TO_INSTALL mkl"
    else
        TO_INSTALL="$TO_INSTALL nomkl"
    fi

    if [[ -n "$PANDAS_VERSION" ]]; then
        TO_INSTALL="$TO_INSTALL pandas=$PANDAS_VERSION"
    fi

    conda env remove -y -n $ENVNAME || true

    conda create -n $ENVNAME --yes $TO_INSTALL
    source activate $ENVNAME

    pip install nose==1.3.7
    # for python 3.4, conda does not have recent pytest packages
    if [[ "$PYTHON_VERSION" == "3.4" ]]; then
        pip install pytest==3.5
    fi
fi

if [[ "$COVERAGE" == "true" ]]; then
    pip install coverage codecov
fi

if [[ "$TEST_DOCSTRINGS" == "true" ]]; then
    pip install sphinx sphinx-autobuild numpydoc sphinx_rtd_theme sphinx_paramlinks  # numpydoc requires sphinx
fi

if [[ "$RUN_FLAKE8" == "true" ]]; then
    conda install flake8 -y
fi

pip install --editable .
