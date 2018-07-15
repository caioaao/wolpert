from copy import deepcopy

import numpy as np

from sklearn.utils.testing import assert_equal, assert_array_equal
from sklearn import datasets
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.linear_model import RidgeClassifier, LinearRegression
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.ensemble import RandomForestClassifier

from wolpert.wrappers import CVStackableTransformer, CVWrapper

from .utils import check_estimator

iris = datasets.load_iris()
X, y = iris.data[:, 1:3], iris.target

RANDOM_SEED = 8939

META_ESTIMATOR_PARAMS = {'cv': [2, StratifiedKFold()],
                         'n_cv_jobs': [1, 2]}

META_ESTIMATOR_FIT_PARAMS = [{}, {"sample_weight": np.ones(y.shape)}]


def _check_estimator(estimator, **fit_params):
    check_estimator(estimator, X, y, **fit_params)

    # checks that result from blend has the same length as input
    estimator.blend(X, y)


def test_regression():
    # tests regression with various parameter settings

    meta_params = {'method': ['auto', 'predict']}
    meta_params.update(META_ESTIMATOR_PARAMS)

    regressors = [LinearRegression(), LinearSVR(random_state=RANDOM_SEED)]

    for reg in regressors:
        for params in ParameterGrid(meta_params):
            blended_reg = CVStackableTransformer(reg, **params)
            for fit_params in META_ESTIMATOR_FIT_PARAMS:
                _check_estimator(blended_reg, **fit_params)


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
            blended_clf = CVStackableTransformer(clf, **params)
            for fit_params in META_ESTIMATOR_FIT_PARAMS:
                _check_estimator(blended_clf, **fit_params)


WRAPPER_PARAMS = {"default_method": ["auto", "predict_proba"],
                  "cv": [1, 2],
                  "n_cv_jobs": [1, 2, 3]}


def test_wrapper():
    dummy_estimator = "dummy"

    for params in ParameterGrid(WRAPPER_PARAMS):
        wrapper = CVWrapper(**params)

        default_method = params.pop("default_method")

        for method in [None, "auto", "predict_proba"]:
            wrapped_est = wrapper.wrap_estimator(
                dummy_estimator, method=method)

            direct_wrapped_est = CVStackableTransformer(
                dummy_estimator, method=method, **params)

            # checks that method is chosen appropriately
            if method is None:
                assert_equal(wrapped_est.method, default_method)
            else:
                assert_equal(wrapped_est.method, method)

            # check that both transformers are the same
            for k in ["estimator", "cv", "n_cv_jobs"]:
                assert_equal(getattr(wrapped_est, k),
                             getattr(direct_wrapped_est, k))
