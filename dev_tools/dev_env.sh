#!/usr/bin/env bash
# This script was adapted from scikit-learn's install script for travis CI. It
# will create a conda environment with everything you need for development. Most
# of the dependencies here are for building scikit learn from scratch.

# License: 3-clause BSD

set -e

ENVNAME="${ENVNAME:-wolpert_devel}"

DISTRIB="${DISTRIB:-conda}"
PYTHON_VERSION="${PYTHON_VERSION:-3.6.2}"
INSTALL_MKL="${INSTALL_MKL:-true}"
NUMPY_VERSION="${NUMPY_VERSION:-1.14.2}"
SCIPY_VERSION="${SCIPY_VERSION:-1.0.0}"
PANDAS_VERSION="${PANDAS_VERSION:-0.20.3}"
CYTHON_VERSION="${CYTHON_VERSION:-0.26.1}"
PYAMG_VERSION="${PYAMG_VERSION:-3.3.2}"
PILLOW_VERSION="${PILLOW_VERSION:-4.3.0}"
COVERAGE=true
CHECK_PYTEST_SOFT_DEPENDENCY="${CHECK_PYTEST_SOFT_DEPENDENCY:-true}"
TEST_DOCSTRINGS="${TEST_DOCSTRINGS:-true}"


if [[ "$DISTRIB" == "conda" ]]; then
    deactivate || true

    conda update --yes conda

    TO_INSTALL="python=$PYTHON_VERSION pip pytest pytest-cov \
                numpy=$NUMPY_VERSION scipy=$SCIPY_VERSION \
                cython=$CYTHON_VERSION"

    if [[ "$INSTALL_MKL" == "true" ]]; then
        TO_INSTALL="$TO_INSTALL mkl"
    else
        TO_INSTALL="$TO_INSTALL nomkl"
    fi

    if [[ -n "$PANDAS_VERSION" ]]; then
        TO_INSTALL="$TO_INSTALL pandas=$PANDAS_VERSION"
    fi

    if [[ -n "$PYAMG_VERSION" ]]; then
        TO_INSTALL="$TO_INSTALL pyamg=$PYAMG_VERSION"
    fi

    if [[ -n "$PILLOW_VERSION" ]]; then
        TO_INSTALL="$TO_INSTALL pillow=$PILLOW_VERSION"
    fi


    conda env remove -y -n $ENVNAME || true

    conda create -n $ENVNAME --yes $TO_INSTALL
    source activate $ENVNAME

    # for python 3.4, conda does not have recent pytest packages
    if [[ "$PYTHON_VERSION" == "3.4" ]]; then
        pip install pytest==3.5
    fi

elif [[ "$DISTRIB" == "ubuntu" ]]; then
    # At the time of writing numpy 1.9.1 is included in the travis
    # virtualenv but we want to use the numpy installed through apt-get
    # install.
    deactivate
    # Create a new virtualenv using system site packages for python, numpy
    # and scipy
    virtualenv --system-site-packages testvenv
    source testvenv/bin/activate
    pip install pytest pytest-cov cython==$CYTHON_VERSION

elif [[ "$DISTRIB" == "scipy-dev-wheels" ]]; then
    # Set up our own virtualenv environment to avoid travis' numpy.
    # This venv points to the python interpreter of the travis build
    # matrix.
    virtualenv --python=python ~/testvenv
    source ~/testvenv/bin/activate
    pip install --upgrade pip setuptools

    echo "Installing numpy and scipy master wheels"
    dev_url=https://7933911d6844c6c53a7d-47bd50c35cd79bd838daf386af554a83.ssl.cf2.rackcdn.com
    pip install --pre --upgrade --timeout=60 -f $dev_url numpy scipy pandas cython
    pip install pytest pytest-cov
fi

if [[ "$COVERAGE" == "true" ]]; then
    pip install coverage codecov
fi

if [[ "$TEST_DOCSTRINGS" == "true" ]]; then
    pip install sphinx numpydoc  # numpydoc requires sphinx
fi

if [[ "$RUN_FLAKE8" == "true" ]]; then
    conda install flake8 -y
fi

cd sklearn && pip install .
# if [[ "$SKIP_TESTS" == "true" && "$CHECK_PYTEST_SOFT_DEPENDENCY" != "true" ]]; then
#     echo "No need to build scikit-learn"
# else
#     # Build scikit-learn in the install.sh script to collapse the verbose
#     # build output in the travis output when it succeeds.
#     python --version
#     python -c "import numpy; print('numpy %s' % numpy.__version__)"
#     python -c "import scipy; print('scipy %s' % scipy.__version__)"
#     python -c "\
# try:
#     import pandas
#     print('pandas %s' % pandas.__version__)
# except ImportError:
#     pass
# "
# fi

cd .. && pip install --editable .
