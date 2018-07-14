import numpy as np

from .sklearn_pipeline import (Pipeline, FeatureUnion, _apply_weight,
                               _name_estimators)
from sklearn.externals.joblib import Parallel, delayed
from sklearn.preprocessing import FunctionTransformer
from .wrappers import _choose_wrapper


def _blend_one(transformer, X, y, weight, **fit_params):
    res = transformer.blend(X, y, **fit_params)
    return _apply_weight(res, weight)


def _fit_blend_one(transformer, X, y, weight, **fit_params):
    Xt = transformer.fit_blend(X, y, **fit_params)
    return _apply_weight(Xt, weight), transformer


class StackingLayer(FeatureUnion):
    """Creates a single layer for the stacked ensemble.

    This works similarly to scikit learn's ``FeatureUnion`` class, with the
    only difference that it also exposes methods for blending all estimators
    for building stacked ensembles.

    All transformers must implement ``blend`` or, in other words, all
    transformers must be wrapped with a class that inherits from
    ``BaseStackableTransformer``.

    Some precautions must be taken for this to work properly: when calling
    ``StackingLayer`` constructor directly, make sure all estimators are
    wrapped with the exact same wrapper.

    Parameters of the transformers may be set using its name and the parameter
    name separated by a '__'. A transformer may be replaced entirely by
    setting the parameter with its name to another transformer,
    or removed by setting to ``None``.

    Parameters
    ----------
    transformer_list : list of (string, transformer) tuples
        List of transformer objects to be applied to the data. The first
        half of each tuple is the name of the transformer.

    n_jobs : int, optional
        Number of jobs to run in parallel (default 1).

    transformer_weights : dict, optional
        Multiplicative weights for features per transformer.
        Keys are transformer names, values the weights.

    See also
    --------
    wolpert.pipeline.make_stack_layer : convenience function for simplified
        layer construction.

    Examples
    --------
    >>> from sklearn.naive_bayes import GaussianNB
    >>> from sklearn.svm import SVR
    >>> from wolpert.wrappers import CVStackableTransformer
    >>>
    >>> reg1 = CVStackableTransformer(GaussianNB(priors=None),
    ...                               method='predict')
    >>> reg2 = CVStackableTransformer(SVR(), method='predict')
    >>>
    >>> StackingLayer([("gaussiannb", reg1),
    ...                ("svr", reg2)])
    ...                        # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
        StackingLayer(n_jobs=1,
        transformer_list=[('gaussiannb',
                           CVStackableTransformer(cv=3,
                                                  estimator=GaussianNB(...),
                                                  method='predict',
                                                  n_cv_jobs=1)),
                          ('svr',
                           CVStackableTransformer(cv=3,
                                                  estimator=SVR(...),
                                                  method='predict',
                                                  n_cv_jobs=1))],
        transformer_weights=None)
    """
    def _validate_one_transformer(self, t):
        if not hasattr(t, "blend"):
            raise TypeError("All transformers should implement 'blend'."
                            " '%s' (type %s) doesn't" %
                            (t, type(t)))
        return super(StackingLayer, self)._validate_one_transformer(t)

    @property
    def _blend_one(self):
        return _blend_one

    @property
    def _fit_blend_one(self):
        return _fit_blend_one

    def blend(self, X, y, **fit_params):
        """Transform dataset by calling ``blend`` on each transformer and concatenating
        the results.

        Parameters
        ----------
        X : array-like or sparse matrix, shape=(n_samples, n_features)
            Input data used to build forests. Use ``dtype=np.float32`` for
            maximum efficiency.

        y : array-like, shape = [n_samples]
            Target values.

        **fit_params : parameters to be passed to the base estimator.

        Returns
        -------
        X_transformed : sparse matrix, shape=(n_samples, n_out)
            Transformed dataset.

        """
        self._validate_transformers()
        Xs = Parallel(n_jobs=self.n_jobs)(
            delayed(self._blend_one)(trans, X, y, weight, **fit_params)
            for name, trans, weight in self._iter())

        if not Xs:
            # All transformers are None
            return np.zeros((X.shape[0], 0))

        return self._stack_results(Xs)

    def fit_blend(self, X, y, weight=None, **fit_params):
        """Fit to and transform dataset by calling ``fit_blend`` on each transformer
        and concatenating the results.

        Parameters
        ----------
        X : array-like or sparse matrix, shape=(n_samples, n_features)
            Input data used to build forests. Use ``dtype=np.float32`` for
            maximum efficiency.

        y : array-like, shape = [n_samples]
            Target values.

        **fit_params : parameters to be passed to the base estimator.

        Returns
        -------
        X_transformed : sparse matrix, shape=(n_samples, n_out)
            Transformed dataset.

        """
        self._validate_transformers()

        result = Parallel(n_jobs=self.n_jobs)(
            delayed(self._fit_blend_one)(trans, X, y, weight, **fit_params)
            for name, trans, weight in self._iter())

        if not result:
            # All transformers are None
            return np.zeros((X.shape[0], 0))
        Xs, transformers = zip(*result)
        self._update_transformer_list(transformers)

        return self._stack_results(Xs)


class StackingPipeline(Pipeline):
    """The stacked model.

    All steps but the last one must implement ``blend``.
    """
    def __init__(self, steps, memory=None):
        super(StackingPipeline, self).__init__(steps, memory)

    @property
    def _fit_transform_one(self):
        return _fit_blend_one

    def _validate_transformers(self):
        super(StackingPipeline, self)._validate_steps()
        names, estimators = zip(*self.steps)

        # validate estimators
        transformers = estimators[:-1]

        for t in transformers:
            if t is None:
                continue
            if hasattr(t, "blend"):
                raise TypeError("All intermediate steps should be "
                                "transformers and implement blend."
                                " '%s' (type %s) doesn't" % (t, type(t)))


def _identity(x):
    return x


def _identity_transformer():
    """Contructs a transformer that returns its input unchanged"""
    return FunctionTransformer(_identity, accept_sparse=True)


def _wrap_estimators(named_estimators, method='auto', blending_type="cv",
                     **blending_opts):
    wrapper = _choose_wrapper(blending_type)
    return [(name, wrapper(
        est, method=method, **blending_opts))
            for name, est in named_estimators]


def make_stack_layer(*estimators, **kwargs):
    """Creates a single stack layer to be used in a stacked ensemble.

    Parameters
    ----------
    *estimators : list of estimators to be wrapped and used in a layer

    method : string, optional (default='auto')
        This method will be called on the estimator to produce the output of
        transform. If the method is ``auto``, will try to invoke, for each
        estimator, ``predict_proba``, ``decision_function`` or ``predict``
        in that order.

    restack: bool, optional (default=False)
        Wether to repeat the layer input in the output.

    n_jobs : int, optinal (default=1)
        Number of jobs to be passed to ``StackingLayer``. Each job will be
        responsible for blending one of the estimators.

    blending_type: string, optional (default='cv')
        The strategy to be used when blending. Possible values are 'cv' and
        'holdout'.

    **blending_opts: arguments to be passed to the blending wrapper

    Examples
    --------
    >>> from sklearn.naive_bayes import GaussianNB
    >>> from sklearn.svm import SVR
    >>> make_stack_layer(GaussianNB(priors=None), SVR(),
    ...                  method='predict')
    ...                        # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
        StackingLayer(n_jobs=1,
        transformer_list=[('gaussiannb',
                           CVStackableTransformer(cv=3,
                                                  estimator=GaussianNB(...),
                                                  method='predict',
                                                  n_cv_jobs=1)),
                          ('svr',
                           CVStackableTransformer(cv=3,
                                                  estimator=SVR(...),
                                                  method='predict',
                                                  n_cv_jobs=1))],
        transformer_weights=None)

    Returns
    -------
    l : StackingLayer
    """
    method = kwargs.pop('method', 'auto')
    restack = kwargs.pop('restack', False)
    n_jobs = kwargs.pop('n_jobs', 1)
    blending_type = kwargs.pop('blending_type', "cv")
    blending_opts = kwargs

    named_estimators = _name_estimators(estimators)

    transformer_list = _wrap_estimators(
        named_estimators, method=method, blending_type=blending_type,
        **blending_opts)

    if restack:
        wrapper = _choose_wrapper(blending_type)
        transformer_list.append(
            ('identity-transformer', wrapper(_identity_transformer(),
                                             method='transform',
                                             **blending_opts)))

    return StackingLayer(transformer_list, n_jobs=n_jobs)
