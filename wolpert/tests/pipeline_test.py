from sklearn.utils.testing import assert_array_equal
from sklearn import datasets
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone, BaseEstimator

from wolpert.pipeline import StackingLayer, StackingPipeline, make_stack_layer
from wolpert.wrappers import CVStackableTransformer, CVWrapper, HoldoutWrapper

iris = datasets.load_iris()
X, y = iris.data[:, 1:3], iris.target

RANDOM_SEED = 8939

STACK_LAYER_PARAMS = {'restack': [False, True],
                      'n_jobs': [1, 2]}

STACK_LAYER_CV_PARAMS = {
    'default_method': ['auto', 'predict', 'predict_proba'],
    'cv': [3, StratifiedKFold()],
    'n_cv_jobs': [1, 2]}

STACK_LAYER_HOLDOUT_PARAMS = {
    'default_method': ['auto', 'predict', 'predict_proba'],
    'holdout_size': [.5],
    'random_state': [435],
    'fit_to_all_data': [True, False]}


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
    Xt, indexes = l_.blend(X, y)
    if restack:
        _check_restack(Xt, X[indexes])

    # check that `fit_blend` is accessible
    l_ = clone(l)
    Xt, indexes = l_.fit_blend(X, y)
    if restack:
        _check_restack(Xt, X[indexes])

    # check that `fit_blend` fits the layer
    l_ = clone(l)
    Xt0, indexes0 = l_.fit_blend(X, y)

    Xt, indexes = l_.blend(X, y)
    if restack:
        _check_restack(Xt, X[indexes])

    # check results match
    assert_array_equal(Xt0, Xt)
    assert_array_equal(indexes0, indexes)


def test_layer_regression():
    base_regs = [
        ('lr', CVStackableTransformer(
            LinearRegression())),
        ('svr', CVStackableTransformer(
            LinearSVR(random_state=RANDOM_SEED)))]

    for params in ParameterGrid(STACK_LAYER_PARAMS):
        if not params["restack"]:
            continue
        params.pop("restack")

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
        if params["restack"]:
            continue
        params.pop("restack")

        # assert constructor
        clf_layer = StackingLayer(base_clfs, **params)

        _check_layer(clf_layer, False)


def test_layer_helper_constructor():
    base_estimators = [LinearRegression(), LinearRegression()]

    for layer_params in ParameterGrid(STACK_LAYER_PARAMS):
        # test with string as wrapper
        reg_layer = make_stack_layer(*base_estimators, blending_wrapper="cv",
                                     **layer_params)
        _check_layer(reg_layer, layer_params["restack"])

        # test with wrapper object
        for wrapper_params in ParameterGrid(STACK_LAYER_CV_PARAMS):
            if layer_params['n_jobs'] != 1 and wrapper_params['n_cv_jobs'] != 1:
                continue  # nested parallelism is not supported

            if wrapper_params['default_method'] is 'predict_proba':
                continue

            wrapper = CVWrapper(**wrapper_params)
            reg_layer = make_stack_layer(
                *base_estimators, blending_wrapper=wrapper, **layer_params)
            _check_layer(reg_layer, layer_params["restack"])

        # test with string as wrapper
        reg_layer = make_stack_layer(
            *base_estimators, blending_wrapper="holdout", **layer_params)

        # TODO check restack on holdout. it's working, but _check_layer must be
        # refactored to handle it
        _check_layer(reg_layer, False)

        # test with wrapper object
        for wrapper_params in ParameterGrid(STACK_LAYER_HOLDOUT_PARAMS):
            if wrapper_params['default_method'] is 'predict_proba':
                continue

            wrapper = HoldoutWrapper(**wrapper_params)
            reg_layer = make_stack_layer(
                *base_estimators, blending_wrapper=wrapper, **layer_params)
            # TODO check restack on holdout. it's working, but _check_layer
            # must be refactored to handle it
            _check_layer(reg_layer, False)


class IdentityEstimator(BaseEstimator):
    def __init__(self):
        self.last_fit_params = None

    def fit(self, X, y, *args, **kwargs):
        self.last_fit_params = (X, y)
        return self

    def predict(self, X, *args, **kwargs):
        return X


def test_pipeline():
    for blending_wrapper in ['cv', 'holdout']:
        l0 = make_stack_layer(LinearRegression(), LinearRegression(),
                              blending_wrapper=blending_wrapper)
        identity_est = IdentityEstimator()
        reg = StackingPipeline([("layer-0", clone(l0)),
                                ("l1", identity_est)])

        X_layer0, indexes = l0.fit_blend(X, y)
        reg.fit(X, y)

        # second layer must receives the blending results from the first one
        assert_array_equal(X_layer0, identity_est.last_fit_params[0])
        assert_array_equal(y[indexes], identity_est.last_fit_params[1])

        preds_l0 = l0.transform(X)
        preds_pipeline = reg.predict(X)

        # identity estimator must return the same result as first layer
        assert_array_equal(preds_l0, preds_pipeline)

    # check that pipeline also provides `fit_blend` when final estimator does it
    l0 = make_stack_layer(LinearRegression(), LinearRegression())
    reg = StackingPipeline([("layer-0", clone(l0)), ("layer-1", clone(l0))])

    l1 = clone(l0)
    Xt, indexes = l0.fit_blend(X, y)
    Xt, indexes = l1.fit_blend(Xt, y[indexes])

    Xt_pipeline, indexes_pipeline = reg.fit_blend(X, y)
    assert_array_equal(Xt, Xt_pipeline)
    assert_array_equal(indexes, indexes_pipeline)

    # check that pipeline also provides `fit_blend` when final estimator does it
    Xt, indexes = l0.blend(X, y)
    Xt, indexes = l1.blend(Xt, y[indexes])

    Xt_pipeline, indexes_pipeline = reg.blend(X, y)
    assert_array_equal(Xt, Xt_pipeline)
    assert_array_equal(indexes, indexes_pipeline)
