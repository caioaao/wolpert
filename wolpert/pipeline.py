import numpy as np

from sklearn.externals import six
from sklearn.externals.joblib import Parallel, delayed
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import FunctionTransformer
from sklearn.utils.metaestimators import if_delegate_has_method
from sklearn.utils.validation import check_memory

from .sklearn_pipeline import (Pipeline, FeatureUnion, _apply_weight,
                               _name_estimators, _stack_results)
from .wrappers import _choose_wrapper


def _blend_one(transformer, X, y, weight, **fit_params):
    res, indexes = transformer.blend(X, y, **fit_params)
    return _apply_weight(res, weight), indexes


def _fit_blend_one(transformer, X, y, weight, **fit_params):
    Xt, indexes = transformer.fit_blend(X, y, **fit_params)
    return _apply_weight(Xt, weight), transformer, indexes


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

    @staticmethod
    def _validate_xs(Xs):
        if not Xs:
            raise ValueError(
                "No support for all transformers as None.")

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
        X_transformed, indexes : tuple of (sparse matrix, array-like)
            `X_transformed` is the transformed dataset.
            `indexes` is the indexes of the transformed data on the input.
        """
        self._validate_transformers()
        res = Parallel(n_jobs=self.n_jobs)(
              delayed(self._blend_one)(trans, X, y, weight, **fit_params)
              for name, trans, weight in self._iter())

        Xs, indexes = zip(*res)

        StackingLayer._validate_xs(Xs)

        return _stack_results(Xs), indexes[0]

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
        X_transformed, indexes : tuple of (sparse matrix, array-like)
            `X_transformed` is the transformed dataset.
            `indexes` is the indexes of the transformed data on the input.
        """
        self._validate_transformers()

        result = Parallel(n_jobs=self.n_jobs)(
            delayed(self._fit_blend_one)(trans, X, y, weight, **fit_params)
            for name, trans, weight in self._iter())

        Xs = [Xt for Xt, _, _ in result]
        transformers = [t for _, t, _ in result]
        indexes = [i for _, _, i in result]

        StackingLayer._validate_xs(Xs)

        self._update_transformer_list(transformers)

        return _stack_results(Xs), indexes[0]


class StackingPipeline(Pipeline):
    """A pipeline of ``StackingLayer``s with a final estimator.

    During ``fit``, sequentially apply ``fit_blend`` to each ``StackingLayer``
    and feeds the transformed data into the next layer. Finally fits the final
    estimator to the last transformed data.

    When generating predictions, calls ``transform`` on each layer sequentially
    before feeding the data to the final estimator.

    Parameters
    ----------
    steps : list of (string, estimator) tuples
        List of (name, object) tuples that are chained, in the order in which
        they are chained, with the last object an estimator. All objects
        besides the last one must inherit from ``BaseStackableTransformer``.

    memory : None, str or object with the joblib.Memory interface, optional
        Used to cache the fitted transformers of the pipeline. By default,
        no caching is performed. If a string is given, it is the path to
        the caching directory. Enabling caching triggers a clone of
        the transformers before fitting. Therefore, the transformer
        instance given to the pipeline cannot be inspected
        directly. Use the attribute ``named_steps`` or ``steps`` to
        inspect estimators within the pipeline. Caching the
        transformers is advantageous when fitting is time consuming.

    Examples
    --------
    >>> from sklearn.naive_bayes import GaussianNB
    >>> from sklearn.svm import SVR
    >>> from sklearn.linear_model import LinearRegression
    >>> layer0 = make_stack_layer(GaussianNB(priors=None), SVR())
    >>> final_estimator = LinearRegression()
    >>> StackingPipeline([("l0", layer0), ("final", final_estimator)])
    ...                        # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    StackingPipeline(memory=None,
             steps=[('l0', StackingLayer(...)),
                    ('final', LinearRegression(...))])
    """
    def __init__(self, steps, memory=None):
        super(StackingPipeline, self).__init__(steps, memory)

    @property
    def _fit_transform_one(self):
        return _fit_blend_one

    @property
    def _fit_blend_one(self):
        return _fit_blend_one

    def _validate_steps(self):
        super(StackingPipeline, self)._validate_steps()
        names, estimators = zip(*self.steps)

        # validate estimators
        transformers = estimators[:-1]

        for t in transformers:
            if t is None:
                continue
            if not hasattr(t, "blend"):
                raise TypeError("All intermediate steps should implement blend."
                                " '%s' (type %s) doesn't" % (t, type(t)))

    def _fit(self, X, y=None, **fit_params):
        # shallow copy of steps - this should really be steps_
        self.steps = list(self.steps)
        self._validate_steps()
        # Setup the memory
        memory = check_memory(self.memory)

        fit_blend_one_cached = memory.cache(self._fit_blend_one)

        fit_params_steps = dict((name, {}) for name, step in self.steps
                                if step is not None)
        for pname, pval in six.iteritems(fit_params):
            step, param = pname.split('__', 1)
            fit_params_steps[step][param] = pval
        Xt = X
        indexes = np.arange(X.shape[0])
        for step_idx, (name, transformer) in enumerate(self.steps[:-1]):
            if transformer is None:
                pass
            else:
                if hasattr(memory, 'cachedir') and memory.cachedir is None:
                    # we do not clone when caching is disabled to preserve
                    # backward compatibility
                    cloned_transformer = transformer
                else:
                    cloned_transformer = clone(transformer)
                # Fit or load from cache the current transfomer
                Xt, fitted_transformer, indexes = fit_blend_one_cached(
                    cloned_transformer, Xt, y[indexes], None,
                    **fit_params_steps[name])
                # Replace the transformer of the step with the fitted
                # transformer. This is necessary when loading the transformer
                # from the cache.
                self.steps[step_idx] = (name, fitted_transformer)

        if self._final_estimator is None:
            return Xt, {}, indexes

        return Xt, fit_params_steps[self.steps[-1][0]], indexes

    def fit(self, X, y=None, **fit_params):
        """Fit the model

        Fit all the transforms one after the other and transform the
        data, then fit the transformed data using the final estimator.

        Parameters
        ----------
        X : iterable
            Training data. Must fulfill input requirements of first step of the
            pipeline.

        y : iterable, default=None
            Training targets. Must fulfill label requirements for all steps of
            the pipeline.

        **fit_params : dict of string -> object
            Parameters passed to the ``fit`` method of each step, where
            each parameter name is prefixed such that parameter ``p`` for step
            ``s`` has key ``s__p``.

        Returns
        -------
        self : Pipeline
            This estimator
        """
        Xt, fit_params, indexes = self._fit(X, y, **fit_params)

        if self._final_estimator is not None:
            self._final_estimator.fit(Xt, y[indexes], **fit_params)

        return self

    def fit_transform(self, X, y=None, **fit_params):
        """Fit the model and transform with the final estimator

        Fits all the transforms one after the other and transforms the
        data, then uses fit_transform on transformed data with the final
        estimator.

        Parameters
        ----------
        X : iterable
            Training data. Must fulfill input requirements of first step of the
            pipeline.

        y : iterable, default=None
            Training targets. Must fulfill label requirements for all steps of
            the pipeline.

        **fit_params : dict of string -> object
            Parameters passed to the ``fit`` method of each step, where
            each parameter name is prefixed such that parameter ``p`` for step
            ``s`` has key ``s__p``.

        Returns
        -------
        Xt : array-like, shape = [n_samples, n_transformed_features]
            Transformed samples
        """
        self.fit(X, y, **fit_params)
        return self.transform(X)

    @if_delegate_has_method(delegate='_final_estimator')
    def fit_predict(self, X, y=None, **fit_params):
        """Applies fit_predict of last step in pipeline after transforms.

        Helper function. Same result as calling ``fit()`` followed by
        ``predict``.

        Parameters
        ----------
        X : iterable
            Training data. Must fulfill input requirements of first step of
            the pipeline.

        y : iterable, default=None
            Training targets. Must fulfill label requirements for all steps
            of the pipeline.

        **fit_params : dict of string -> object
            Parameters passed to the ``fit`` method of each step, where
            each parameter name is prefixed such that parameter ``p`` for step
            ``s`` has key ``s__p``.

        Returns
        -------
        y_pred : array-like

        """
        self.fit(X, y, **fit_params)
        return self.predict(X)

    @if_delegate_has_method(delegate='_final_estimator')
    def fit_blend(self, X, y=None, **fit_params):
        """Applies fit_blend of last step in pipeline after transforms.

        Applies fit_blends of a pipeline to the data, followed by the
        fit_blend method of the final estimator in the pipeline. Valid
        only if the final estimator implements fit_blend.

        Parameters
        ----------
        X : iterable
            Training data. Must fulfill input requirements of first step of
            the pipeline.

        y : iterable, default=None
            Training targets. Must fulfill label requirements for all steps
            of the pipeline.

        **fit_params : dict of string -> object
            Parameters passed to the ``fit`` method of each step, where
            each parameter name is prefixed such that parameter ``p`` for step
            ``s`` has key ``s__p``.

        Returns
        -------
        X_transformed, indexes : tuple of (sparse matrix, array-like)
            `X_transformed` is the transformed dataset.
            `indexes` is the indexes of the transformed data on the input.

        """
        Xt, fit_params, indexes = self._fit(X, y, **fit_params)

        return self._final_estimator.fit_blend(Xt, y[indexes], **fit_params)

    @if_delegate_has_method(delegate='_final_estimator')
    def blend(self, X, y=None, **fit_params):
        """Apply blends, and blends with the final estimator

        Parameters
        ----------
        X : iterable
            Training data. Must fulfill input requirements of first step of
            the pipeline.

        y : iterable, default=None
            Training targets. Must fulfill label requirements for all steps
            of the pipeline.

        **fit_params : dict of string -> object
            Parameters passed to the ``fit`` method of each step, where
            each parameter name is prefixed such that parameter ``p`` for step
            ``s`` has key ``s__p``.

        Returns
        -------
        X_transformed, indexes : tuple of (sparse matrix, array-like)
            `X_transformed` is the transformed dataset.
            `indexes` is the indexes of the transformed data on the input.

        """
        Xt, indexes = X, np.arange(X.shape[0])
        for _, transform in self.steps:
            if transform is not None:
                Xt, indexes = transform.blend(Xt, y[indexes], **fit_params)
        return Xt, indexes

    def score(self, X, y=None, **validation_args):
        """Scores the model using scikit learn's ``cross_validate``

        Blends the whole dataset and then uses the final estimator to produce a
        score

        Parameters
        ----------
        X : iterable
            Training data. Must fulfill input requirements of first step of
            the pipeline.

        y : iterable, default=None
            Training targets. Must fulfill label requirements for all steps
            of the pipeline.

        **score_args : dictionary
            Arguments to be passed to ``cross_validate``

        Returns
        -------
        scores : dict of float arrays of shape=(n_splits,)
            Array of scores of the estimator for each run of the cross validation.

            A dict of arrays containing the score/time arrays for each scorer is
            returned. The possible keys for this ``dict`` are:

                ``test_score``
                    The score array for test scores on each cv split.
                ``train_score``
                    The score array for train scores on each cv split.
                    This is available only if ``return_train_score`` parameter
                    is ``True``.
                ``fit_time``
                    The time for fitting the estimator on the train
                    set for each cv split.
                ``score_time``
                    The time for scoring the estimator on the test set for each
                    cv split. (Note time for scoring on the train set is not
                    included even if ``return_train_score`` is set to ``True``
                ``estimator``
                    The estimator objects for each cv split.
                    This is available only if ``return_estimator`` parameter
                    is set to ``True``.

        """
        Xt, indexes = X, np.arange(X.shape[0])

        for _, transform in self.steps[:-1]:
            if transform is not None:
                Xt, indexes = transform.blend(Xt, y[indexes])

        return cross_validate(self._final_estimator, X, y, **validation_args)

def _identity(x):
    return x


def _identity_transformer():
    """Contructs a transformer that returns its input unchanged"""
    return FunctionTransformer(_identity, accept_sparse=True)


def _wrap_estimators(named_estimators, blending_wrapper="cv"):
    wrapper = _choose_wrapper(blending_wrapper)
    return [(name, wrapper.wrap_estimator(est))
            for name, est in named_estimators]


def make_stack_layer(*estimators, **kwargs):
    """Creates a single stack layer to be used in a stacked ensemble.

    Parameters
    ----------
    *estimators : list
        List of estimators to be wrapped and used in a layer

    restack: bool, optional (default=False)
        Wether to repeat the layer input in the output.

    n_jobs : int, optinal (default=1)
        Number of jobs to be passed to ``StackingLayer``. Each job will be
        responsible for blending one of the estimators.

    blending_wrapper: string or Wrapper object, optional (default='cv')
        The strategy to be used when blending. Possible string values are 'cv'
        and 'holdout'. If a wrapper object is passed, it will be used instead.

    Examples
    --------

    >>> from sklearn.naive_bayes import GaussianNB
    >>> from sklearn.svm import SVR
    >>> make_stack_layer(GaussianNB(priors=None), SVR())
    ...                        # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
        StackingLayer(n_jobs=1,
        transformer_list=[('gaussiannb',
                           CVStackableTransformer(cv=3,
                                                  estimator=GaussianNB(...),
                                                  method='auto',
                                                  n_cv_jobs=1)),
                          ('svr',
                           CVStackableTransformer(cv=3,
                                                  estimator=SVR(...),
                                                  method='auto',
                                                  n_cv_jobs=1))],
        transformer_weights=None)

    Returns
    -------

    l : StackingLayer

    """
    restack = kwargs.pop('restack', False)
    n_jobs = kwargs.pop('n_jobs', 1)
    blending_wrapper = kwargs.pop('blending_wrapper', "cv")

    if kwargs:
        raise ValueError("Invalid parameters: %s" % kwargs.keys())

    named_estimators = _name_estimators(estimators)

    transformer_list = _wrap_estimators(
        named_estimators, blending_wrapper=blending_wrapper)

    if restack:
        wrapper = _choose_wrapper(blending_wrapper)
        transformer_list.append(
            ('restacker', wrapper.wrap_estimator(
                _identity_transformer(), method='transform')))

    return StackingLayer(transformer_list, n_jobs=n_jobs)
