"""Stacked ensemble wrapper using cross validation"""

# Author: Caio Oliveira <caioaao@gmail.com>
# License: BSD 3 clause

from sklearn.base import (BaseEstimator, TransformerMixin, MetaEstimatorMixin,
                          clone)
from sklearn.model_selection import cross_val_predict


class CVStackableTransformer(BaseEstimator, MetaEstimatorMixin,
                             TransformerMixin):
    """Transformer to turn estimators into meta-estimators for model stacking

    This class uses the k-fold predictions to "blend" the estimator. This
    allows the subsequent layers to use all the data for training. The drawback
    is that, as the metaestimators will be re-trained using the whole training
    set, the train and test set for subsequent layers won't be generated from
    the same probability distribution. Either way, this method still proves
    useful in practice.

    Parameters
    ----------
    estimator : predictor
        The estimator to be blended.

    cv : int, cross-validation generator or an iterable, optional (default=3)
        Determines the cross-validation splitting strategy to be used for
        generating features to train the next layer on the stacked ensemble or,
        more specifically, during ``blend``.

        Possible inputs for cv are:

        - None, to use the default 3-fold cross-validation,
        - integer, to specify the number of folds.
        - An object to be used as a cross-validation generator.
        - An iterable yielding train/test splits.

        For integer/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass,
        ``sklearn.model_selection.StratifiedKFold`` is used. In all other
        cases, ``sklearn.model_selection.KFold`` is used.

    method : string, optional (default='auto')
        This method will be called on the estimator to produce the output of
        transform. If the method is ``auto``, will try to invoke, for each
        estimator, ``predict_proba``, ``decision_function`` or ``predict``
        in that order.

    n_cv_jobs : int, optional (default=1)
        Number of jobs to be passed to ``cross_val_predict`` during
        ``blend``.

    Examples
    --------
    >>> from sklearn.naive_bayes import GaussianNB
    >>> from wolpert.wrappers import CVStackableTransformer
    >>> CVStackableTransformer(GaussianNB(priors=None), cv=5,
    ...                        method='predict_proba')
    ...     # doctest: +NORMALIZE_WHITESPACE
    CVStackableTransformer(cv=5, estimator=GaussianNB(priors=None),
                           method='predict_proba', n_cv_jobs=1)
    """
    def __init__(self, estimator, cv=3, method='auto', n_cv_jobs=1):
        self.estimator = estimator
        self.cv = cv
        self.method = method
        self.n_cv_jobs = n_cv_jobs

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
        preds = cross_val_predict(self.estimator_, X, y, cv=self.cv,
                                  method=self._estimator_function_name,
                                  n_jobs=self.n_cv_jobs, fit_params=fit_params)
        self.estimator_ = None

        if preds.ndim == 1:
            preds = preds.reshape(-1, 1)

        return preds
