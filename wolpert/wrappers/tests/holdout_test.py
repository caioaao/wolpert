from copy import deepcopy

from sklearn.utils.testing import (assert_equal, assert_array_equal,
                                   assert_almost_equal)
from sklearn import datasets
from sklearn.model_selection import ParameterGrid
from sklearn.linear_model import RidgeClassifier, LinearRegression
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.ensemble import RandomForestClassifier

from wolpert.wrappers import HoldoutStackableTransformer

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
    Xt = estimator.blend(X, y, **fit_params)
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
