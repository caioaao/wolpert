from sklearn.utils.testing import assert_array_equal
from sklearn import datasets
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone

from wolpert.pipeline import StackingLayer, make_stack_layer
from wolpert.wrappers import CVStackableTransformer

iris = datasets.load_iris()
X, y = iris.data[:, 1:3], iris.target

RANDOM_SEED = 8939

STACK_LAYER_PARAMS = {'n_jobs': [1, 2]}


def _check_restack(X, Xorig):
    # checks that original data is appended to the rest of the features
    assert_array_equal(Xorig, X[:, -Xorig.shape[1]:])


def _check_layer(l, restack):
    l_ = clone(l)

    # check that we can fit_transform the data
    Xt = l_.fit_transform(X, y)
    if restack:
        _check_restack(Xt, X)

    # check that we can transform the data
    Xt = l_.transform(X)
    if restack:
        _check_restack(Xt, X)

    # check that `fit` is accessible
    l_ = clone(l)
    l_.fit(X, y)

    # check that we can blend the data
    Xt = l_.blend(X, y)
    if restack:
        _check_restack(Xt, X)

    # check that `fit_blend` is accessible
    l_ = clone(l)
    Xt = l_.fit_blend(X, y)
    if restack:
        _check_restack(Xt, X)

    # check that `fit_blend` fits the layer
    l_ = clone(l)
    Xt0 = l_.fit_blend(X, y)

    Xt = l_.blend(X, y)
    if restack:
        _check_restack(Xt, X)

    # check results match
    assert_array_equal(Xt0, Xt)


def test_layer_regression():
    base_regs = [
        ('lr', CVStackableTransformer(
            LinearRegression())),
        ('svr', CVStackableTransformer(
            LinearSVR(random_state=RANDOM_SEED)))]

    for params in ParameterGrid(STACK_LAYER_PARAMS):
        # assert constructor
        reg_layer = StackingLayer(base_regs, **params)
        _check_layer(reg_layer, False)


def test_layer_classification():
    base_clfs = [
        ('rf1', CVStackableTransformer(RandomForestClassifier(
            random_state=RANDOM_SEED, criterion='gini'))),
        ('rf2', CVStackableTransformer(RandomForestClassifier(
            random_state=RANDOM_SEED, criterion='entropy')))]

    for params in ParameterGrid(STACK_LAYER_PARAMS):
        # assert constructor
        clf_layer = StackingLayer(base_clfs, **params)
        _check_layer(clf_layer, False)


STACK_LAYER_FULL_PARAMS = {'cv': [3, StratifiedKFold()],
                           'restack': [False, True],
                           'method': ['auto', 'predict', 'predict_proba'],
                           'n_jobs': [1, 2],
                           'n_cv_jobs': [1, 2]}


def test_layer_helper_constructor():
    base_estimators = [LinearRegression(), LinearRegression()]
    for params in ParameterGrid(STACK_LAYER_FULL_PARAMS):
        if params['n_jobs'] != 1 and params['n_cv_jobs'] != 1:
            continue  # nested parallelism is not supported

        if params['method'] is 'predict_proba':
            continue

        reg_layer = make_stack_layer(*base_estimators, **params)
        _check_layer(reg_layer, params["restack"])
