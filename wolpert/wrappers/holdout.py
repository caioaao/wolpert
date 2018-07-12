"""Stacked ensemble wrapper using holdout strategy"""

# Author: Caio Oliveira <caioaao@gmail.com>
# License: BSD 3 clause

from sklearn.base import (BaseEstimator, TransformerMixin, MetaEstimatorMixin,
                          clone)
from sklearn.model_selection import train_test_split


class HoldoutStackableTransformer(BaseEstimator, MetaEstimatorMixin,
                                  TransformerMixin):
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

    holdout_size : float, optional (default=.1)
        Fraction of the dataset to be used for training.

    method : string, optional (default='auto')
        This method will be called on the estimator to produce the output of
        transform. If the method is ``auto``, will try to invoke, for each
        estimator, ``predict_proba``, ``decision_function`` or ``predict``
        in that order.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, ``random_state`` is the seed used by the random number
        generator; If ``RandomState`` instance, ``random_state`` is the random
        number generator; If ``None``, the random number generator is the
        ``RandomState`` instance used by ``np.random``.

    Examples
    --------
    >>> from sklearn.naive_bayes import GaussianNB
    >>> from wolpert.wrappers import HoldoutStackableTransformer
    >>> HoldoutStackableTransformer(GaussianNB(priors=None),
    ...                             holdout_size=.2,
    ...                             method='predict_proba')
    ...     # doctest: +NORMALIZE_WHITESPACE
    HoldoutStackableTransformer(estimator=GaussianNB(priors=None),
                                holdout_size=0.2,
                                method='predict_proba',
                                random_state=None)

    """
    def __init__(self, estimator, holdout_size=.1, method='auto',
                 random_state=None):
        self.estimator = estimator
        self.method = method
        self.holdout_size = holdout_size
        self.random_state = random_state

    def fit(self, X, y=None, **fit_params):
        """Fit the estimator to the whole training set.

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
        self.estimator_.fit(X, y, **fit_params)
        return self

    @property
    def _estimator_function_name(self):
        if self.method == 'auto':
            if getattr(self.estimator_, 'predict_proba', None):
                method = 'predict_proba'
            elif getattr(self.estimator_, 'decision_function', None):
                method = 'decision_function'
            else:
                method = 'predict'
        else:
            method = self.method

        return method

    @property
    def _estimator_function(self):
        return getattr(self.estimator_, self._estimator_function_name)

    def transform(self, *args, **kwargs):
        """Transform the whole dataset.

        Parameters
        ----------
        X : array-like or sparse matrix, shape=(n_samples, n_features)
            Input data to be transformed. Use ``dtype=np.float32`` for maximum
            efficiency. Sparse matrices are also supported, use sparse
            ``csr_matrix`` for maximum efficiency.

        Returns
        -------
        X_transformed : sparse matrix, shape=(n_samples, n_out)
            Transformed dataset.

        """
        preds = self._estimator_function(*args, **kwargs)

        if preds.ndim == 1:
            preds = preds.reshape(-1, 1)

        return preds

    def fit_transform(self, X, y=None, **fit_params):
        """Fit estimator to the entire training set and transforms it.

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
        return self.fit(X, y, **fit_params).transform(X)

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
        self.estimator_ = clone(self.estimator)
        X_train, X_holdout, y_train, y_holdout = train_test_split(
            X, y, test_size=self.holdout_size, random_state=self.random_state)
        self.estimator_.fit(X_train, y_train)

        preds = self._estimator_function(X_holdout)

        self.estimator_ = None

        if preds.ndim == 1:
            preds = preds.reshape(-1, 1)

        return preds
