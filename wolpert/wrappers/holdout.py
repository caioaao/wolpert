"""Stacked ensemble wrapper using holdout strategy"""

# Author: Caio Oliveira <caioaao@gmail.com>
# License: BSD 3 clause

from sklearn.base import clone
from sklearn.model_selection import train_test_split

from .base import BaseStackableTransformer, BaseWrapper


class HoldoutStackableTransformer(BaseStackableTransformer):
    """Transformer to turn estimators into meta-estimators for model stacking

    During blending, trains on one part of the dataset and generates
    predictions for the other part. This makes it more robust against leaks,
    but the subsequent layers will have less data to train on.

    Beware that the ``blend`` method will return a dataset that is smaller than
    the original one.

    Parameters
    ----------
    estimator : predictor
        The estimator to be blended.

    method : string, optional (default='auto')
        This method will be called on the estimator to produce the output of
        transform. If the method is ``auto``, will try to invoke, for each
        estimator, ``predict_proba``, ``decision_function`` or ``predict``
        in that order.

    holdout_size : float, optional (default=.1)
        Fraction of the dataset to be ignored for training. The holdout_size
        will be the size of the blended dataset.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, ``random_state`` is the seed used by the random number
        generator; If ``RandomState`` instance, ``random_state`` is the random
        number generator; If ``None``, the random number generator is the
        ``RandomState`` instance used by ``np.random``.

    fit_to_all_data : bool, optional (default=False)
        When true, will fit the final estimator to the whole dataset. If not,
        fits only to the non-holdout set. This only affects the ``fit`` and
        ``fit_blend`` steps.

    Examples
    --------
    >>> from sklearn.naive_bayes import GaussianNB
    >>> from wolpert.wrappers import HoldoutStackableTransformer
    >>> HoldoutStackableTransformer(GaussianNB(priors=None),
    ...                             holdout_size=.2,
    ...                             method='predict_proba')
    ...     # doctest: +NORMALIZE_WHITESPACE
    HoldoutStackableTransformer(estimator=GaussianNB(priors=None),
                                fit_to_all_data=False,
                                holdout_size=0.2,
                                method='predict_proba',
                                random_state=None)

    """
    def __init__(self, estimator, method='auto', holdout_size=.1,
                 random_state=None, fit_to_all_data=False):
        super(HoldoutStackableTransformer, self).__init__(estimator, method)
        self.holdout_size = holdout_size
        self.random_state = random_state
        self.fit_to_all_data = fit_to_all_data

    def _split_data(self, X, y):
        return train_test_split(
            X, y, test_size=self.holdout_size, random_state=self.random_state)

    def _fit_blend(self, X, y, fit_to_all_data, **fit_params):
        X_train, X_holdout, y_train, y_holdout = self._split_data(X, y)

        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X_train, y_train, **fit_params)

        preds = self._estimator_function(X_holdout)

        if preds.ndim == 1:
            preds = preds.reshape(-1, 1)

        if fit_to_all_data:
            fitted_estimator = self.estimator_.fit(X, y, **fit_params)
        else:
            fitted_estimator = self.estimator_

        self.estimator_ = None

        return fitted_estimator, preds

    def blend(self, X, y, **fit_params):
        """Transform dataset using cross validation.

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
        _, preds = self._fit_blend(X, y, False, **fit_params)
        return preds

    def fit(self, X, y, **fit_params):
        """Fit the estimator to the training set.

        If self.fit_to_all_data is true, will fit to whole dataset. If not,
        will only fit to the part not in the holdout set during blending.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples]
            Target values.

        **fit_params : parameters to be passed to the base estimator.

        Returns
        -------
        self : object

        """
        self.estimator_ = clone(self.estimator)

        if self.fit_to_all_data:
            X_train, _, y_train, _ = self._split_data(X, y)

            self.estimator_.fit(X_train, y_train, **fit_params)
        else:
            self.estimator_.fit(X, y, **fit_params)

        return self

    def fit_blend(self, X, y, **fit_params):
        """Fit to and transform dataset.

        If self.fit_to_all_data is true, will fit to whole dataset. If not,
        will only fit to the part not in the holdout set during blending.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples]
            Target values.

        **fit_params : parameters to be passed to the base estimator.

        Returns
        -------
        X_transformed : sparse matrix, shape=(n_samples, n_out)
            Transformed dataset.
        """
        self.estimator_, preds = self._fit_blend(
            X, y, self.fit_to_all_data, **fit_params)
        return preds


class HoldoutWrapper(BaseWrapper):
    """Helper class to wrap estimators with ``CVStackableTransformer``

    Parameters
    ----------

    default_method : string, optional (default='auto')
        This method will be called on the estimator to produce the output of
        transform. If the method is ``auto``, will try to invoke, for each
        estimator, ``predict_proba``, ``decision_function`` or ``predict``
        in that order.

    holdout_size : float, optional (default=.1)
        Fraction of the dataset to be ignored for training. The holdout_size
        will be the size of the blended dataset.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, ``random_state`` is the seed used by the random number
        generator; If ``RandomState`` instance, ``random_state`` is the random
        number generator; If ``None``, the random number generator is the
        ``RandomState`` instance used by ``np.random``.

    fit_to_all_data : bool, optional (default=False)
        When true, will fit the final estimator to the whole dataset. If not,
        fits only to the non-holdout set. This only affects the ``fit`` and
        ``fit_blend`` steps.

    Examples
    --------
    """

    def __init__(self, default_method='auto', holdout_size=.1,
                 random_state=None, fit_to_all_data=False):
        super(CVWrapper, self).__init__(default_method)
        self.holdout_size = holdout_size
        self.random_state = random_state
        self.fit_to_all_data = fit_to_all_data

    def wrap_estimator(self, estimator, method=None, **kwargs):
        """Wraps an estimator and returns a transformer that is suitable for stacking.

        Parameters
        ----------
        estimator : predictor
            The estimator to be blended.

        method : string, optional (default='auto')
            This method will be called on the estimator to produce the output
            of transform. If the method is ``auto``, will try to invoke, for
            each estimator, ``predict_proba``, ``decision_function`` or
            ``predict`` in that order.

        Returns
        -------
        t : CVStackableTransformer
        """
        method = method or self.default_method

        return HoldoutStackableTransformer(
            estimator, method=method, holdout_size=self.holdout_size,
            random_state=self.random_state,
            fit_to_all_data=self.fit_to_all_data)
