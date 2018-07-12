from copy import deepcopy

import numpy as np

from sklearn.utils.testing import assert_equal, assert_array_equal
from sklearn import datasets
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.linear_model import RidgeClassifier, LinearRegression
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.ensemble import RandomForestClassifier

from wolpert.wrappers import CVStackableTransformer

iris = datasets.load_iris()
X, y = iris.data[:, 1:3], iris.target

RANDOM_SEED = 8939

META_ESTIMATOR_PARAMS = {'cv': [2, StratifiedKFold()],
                         'n_cv_jobs': [1, 2]}

META_ESTIMATOR_FIT_PARAMS = [{}, {"sample_weight": np.ones(y.shape)}]


def _check_estimator(estimator, **fit_params):
    # checks that we can fit_transform to the data
    Xt = estimator.fit_transform(X, y, **fit_params)

    # checks that we get a column vector
    assert_equal(Xt.ndim, 2)

    # checks that `fit` is available
    estimator.fit(X, y, **fit_params)

    # checks that we can transform the data after it's fitted
    Xt2 = estimator.transform(X)

    # checks that transformed data is always a column vector
    assert_equal(Xt.ndim, 2)

    # checks that transform is equal to fit_transform
    assert_array_equal(Xt, Xt2)

    # checks for determinism: every `transform` should yield the same result
    for i in range(10):
        assert_array_equal(Xt2, estimator.transform(X))

    # checks that `blend` is availabe
    Xt_blend = estimator.blend(X, y, **fit_params)

    # checks that blended data is always a column vector
    assert_equal(Xt_blend.ndim, 2)


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


def test_transformer_from_classification():
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
