from copy import deepcopy

from sklearn.utils.testing import (assert_equal, assert_array_equal,
                                   assert_almost_equal)
from sklearn import datasets
from sklearn.model_selection import ParameterGrid
from sklearn.linear_model import RidgeClassifier, LinearRegression
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.ensemble import RandomForestClassifier

from wolpert.wrappers import HoldoutStackableTransformer, HoldoutWrapper

from .utils import check_estimator

iris = datasets.load_iris()
X, y = iris.data[:, 1:3], iris.target

RANDOM_SEED = 8939

META_ESTIMATOR_PARAMS = {'holdout_size': [.1, .2, .5],
                         'random_state': [132]}


def _check_estimator(estimator, **fit_params):
    # basic checks
    check_estimator(estimator, X, y, **fit_params)

    # checks that the transformed dataset is roughly the same size as
    # holdout_size parameter
    Xt, _ = estimator.blend(X, y, **fit_params)
    assert_almost_equal(estimator.holdout_size, Xt.shape[0] / float(X.shape[0]))


def test_regression():
    # tests regression with various parameter settings

    meta_params = {'method': ['auto', 'predict']}
    meta_params.update(META_ESTIMATOR_PARAMS)

    regressors = [LinearRegression(), LinearSVR(random_state=RANDOM_SEED)]

    for reg in regressors:
        for params in ParameterGrid(meta_params):
            blended_reg = HoldoutStackableTransformer(reg, **params)
            _check_estimator(blended_reg)


def test_classification():
    # tests classification with various parameter settings

    testcases = [{'clf': RandomForestClassifier(random_state=RANDOM_SEED),
                  'extra_params': {'method': ['auto', 'predict',
                                              'predict_proba']}},
                 {'clf': LinearSVC(random_state=RANDOM_SEED),
                  'extra_params': {'method': ['auto', 'predict',
                                              'decision_function']}},
                 {'clf': RidgeClassifier(random_state=RANDOM_SEED),
                  'extra_params': {'method': ['auto', 'predict']}}]

    for testcase in testcases:
        clf = testcase['clf']

        meta_params = deepcopy(testcase['extra_params'])
        meta_params.update(META_ESTIMATOR_PARAMS)

        for params in ParameterGrid(meta_params):
            blended_clf = HoldoutStackableTransformer(clf, **params)
            _check_estimator(blended_clf)


WRAPPER_PARAMS = {"default_method": ["auto", "predict_proba"],
                  "holdout_size": [.1, .2],
                  "random_state": [10, 20, 30],
                  "fit_to_all_data": [True, False]}


def test_wrapper():
    dummy_estimator = "dummy"

    for params in ParameterGrid(WRAPPER_PARAMS):
        wrapper = HoldoutWrapper(**params)

        default_method = params.pop("default_method")

        for method in [None, "auto", "predict_proba"]:
            wrapped_est = wrapper.wrap_estimator(
                dummy_estimator, method=method)

            direct_wrapped_est = HoldoutStackableTransformer(
                dummy_estimator, method=method, **params)

            # checks that method is chosen appropriately
            if method is None:
                assert_equal(wrapped_est.method, default_method)
            else:
                assert_equal(wrapped_est.method, method)

            # check that both transformers are the same
            for k in ["estimator", "holdout_size", "random_state",
                      "fit_to_all_data"]:
                assert_equal(getattr(wrapped_est, k),
                             getattr(direct_wrapped_est, k))
