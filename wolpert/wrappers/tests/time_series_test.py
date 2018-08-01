import numpy as np
import pytest

from copy import deepcopy

from sklearn import datasets
from sklearn.utils.testing import assert_equal, assert_array_equal
from sklearn.model_selection import ParameterGrid
from sklearn.linear_model import LinearRegression, LogisticRegression

from wolpert.wrappers.time_series import (TimeSeriesSplit, TimeSeriesWrapper,
                                          TimeSeriesStackableTransformer)

from .utils import check_estimator


def assert_splits_equal(result, expected):
    result = list(result)
    for i, (train_expected, test_expected) in enumerate(expected):
        train_result, test_result = result[i]
        assert_array_equal(train_result, train_expected)
        assert_array_equal(test_result, test_expected)


def test_split_class():
    # with default values
    arr = np.arange(3)
    ts = TimeSeriesSplit()
    assert_splits_equal(ts.split(arr),
                        [[[0], [1]],
                         [[0, 1], [2]]])

    # with offset
    arr = np.arange(5)
    ts = TimeSeriesSplit(offset=2)
    assert_splits_equal(ts.split(arr),
                        [[[0], [3]],
                         [[0, 1], [4]]])

    # with min_train_size
    arr = np.arange(3)
    ts = TimeSeriesSplit(min_train_size=2)
    assert_splits_equal(ts.split(arr),
                        [[[0, 1], [2]]])

    # with max_train_size
    arr = np.arange(3)
    ts = TimeSeriesSplit(max_train_size=1)
    assert_splits_equal(ts.split(arr),
                        [[[0], [1]],
                         [[1], [2]]])

    # with test_set_size
    arr = np.arange(5)
    ts = TimeSeriesSplit(test_set_size=2)
    assert_splits_equal(ts.split(arr),
                        [[[0], [1, 2]],
                         [[0, 1, 2], [3, 4]]])

    # min/max train_size validation
    arr = np.arange(3)
    ts = TimeSeriesSplit(min_train_size=10, max_train_size=2)
    splits = ts.split(arr)
    with pytest.raises(ValueError) as _:
        next(splits)


RANDOM_SEED = 8939

X, y = datasets.make_classification(random_state=RANDOM_SEED, n_samples=100)

META_ESTIMATOR_PARAMS = {'offset': [0, 1],
                         'test_set_size': [10, 20],
                         'min_train_size': [30, 50],
                         'max_train_size': [None, 60]}


def _check_estimator(estimator, **fit_params):
    # basic checks
    check_estimator(estimator, X, y, **fit_params)

    Xt, indexes = estimator.blend(X, y, **fit_params)


def test_regression():
    # tests regression with various parameter settings

    meta_params = {'method': ['auto', 'predict']}
    meta_params.update(META_ESTIMATOR_PARAMS)

    regressor = LinearRegression()

    for params in ParameterGrid(meta_params):
        blended_reg = TimeSeriesStackableTransformer(regressor, **params)
        _check_estimator(blended_reg)


def test_classification():
    # tests classification with various parameter settings
    clf = LogisticRegression(random_state=RANDOM_SEED)
    meta_params = {'method': ['auto', 'predict',
                              'predict_proba']}
    meta_params.update(META_ESTIMATOR_PARAMS)

    for params in ParameterGrid(meta_params):
        blended_clf = TimeSeriesStackableTransformer(clf, **params)
        _check_estimator(blended_clf)


def test_wrapper():
    dummy_estimator = "dummy"

    wrapper_params = deepcopy(META_ESTIMATOR_PARAMS)
    wrapper_params["default_method"] = ["auto", "predict", "predict_proba"]

    for params in ParameterGrid(wrapper_params):
        wrapper = TimeSeriesWrapper(**params)

        default_method = params.pop("default_method")

        for method in [None, "auto", "predict_proba"]:
            wrapped_est = wrapper.wrap_estimator(
                dummy_estimator, method=method)

            direct_wrapped_est = TimeSeriesStackableTransformer(
                dummy_estimator, method=method, **params)

            # checks that method is chosen appropriately
            if method is None:
                assert_equal(wrapped_est.method, default_method)
            else:
                assert_equal(wrapped_est.method, method)

            # check that both transformers are the same
            for k in ["estimator", "offset", "test_set_size",
                      "min_train_size", "max_train_size",
                      "n_cv_jobs"]:
                assert_equal(getattr(wrapped_est, k),
                             getattr(direct_wrapped_est, k))
