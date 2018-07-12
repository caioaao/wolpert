"""Stacked ensemble wrapper using holdout strategy"""

# Author: Caio Oliveira <caioaao@gmail.com>
# License: BSD 3 clause

from sklearn.base import clone
from sklearn.model_selection import train_test_split

from .base import BaseStackableTransformer


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
        Fraction of the dataset to be used for training.

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
    def __init__(self, estimator, method='auto', holdout_size=.1,
                 random_state=None):
        super(HoldoutStackableTransformer, self).__init__(estimator, method)
        self.holdout_size = holdout_size
        self.random_state = random_state

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
