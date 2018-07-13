from sklearn.base import (BaseEstimator, TransformerMixin, MetaEstimatorMixin,
                          clone)


class BaseStackableTransformer(BaseEstimator, MetaEstimatorMixin,
                               TransformerMixin):
    def __init__(self, estimator, method='auto'):
        self.estimator = estimator
        self.method = method

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
