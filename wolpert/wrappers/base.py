import abc
import textwrap

import numpy as np

from sklearn.base import (BaseEstimator, TransformerMixin, MetaEstimatorMixin,
                          clone)
from sklearn.externals import six
from sklearn.metrics import (r2_score, median_absolute_error,
                             mean_absolute_error, mean_squared_error,
                             mean_squared_log_error, accuracy_score, f1_score,
                             roc_auc_score, average_precision_score,
                             precision_score, recall_score, log_loss,
                             explained_variance_score, brier_score_loss)
from sklearn.metrics.cluster import (adjusted_rand_score, homogeneity_score,
                                     completeness_score, v_measure_score,
                                     mutual_info_score,
                                     adjusted_mutual_info_score,
                                     normalized_mutual_info_score,
                                     fowlkes_mallows_score)


class BaseStackableTransformer(BaseEstimator, MetaEstimatorMixin,
                               TransformerMixin):
    """Base class for wrappers. Shouldn't be used directly, but inherited by
    specialized wrappers.

    Parameters
    ----------
    estimator : predictor
        The estimator to be blended.

    method : string, optional (default='auto')
        This method will be called on the estimator to produce the output of
        transform. If the method is ``auto``, will try to invoke, for each
        estimator, ``predict_proba``, ``decision_function`` or ``predict``
        in that order.

    scoring : string, callable, dict or None (default=None)
        If not ``None``, will save scores generated by the scoring object on
        the ``scores_`` attribute each time `blend` is called.

    verbose : bool (default=False)
        When true, prints scores to stdout. `scoring` must not be ``None``.

    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, estimator, method='auto', scoring=None, verbose=False):
        self.estimator = estimator
        self.method = method
        self.scoring = scoring
        self.verbose = verbose

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

    @abc.abstractmethod
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
        X_transformed, indexes : tuple of (sparse matrix, array-like)
            `X_transformed` is the transformed dataset.
            `indexes` is the indexes of the transformed data on the input.
        """
        pass

    @abc.abstractmethod
    def fit_blend(self, X, y, **fit_params):
        """Transform dataset using cross validation and fits the estimator.

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
        pass

    def fit(self, X, y=None, **fit_params):
        """Fit the estimator.

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


class BaseWrapper(object):
    __metaclass__ = abc.ABCMeta
    """Factory class used to wrap estimators.

    Parameters
    ----------

    default_method : string, optional (default='auto')
        This method will be called on the estimator to produce the output of
        transform. If the method is ``auto``, will try to invoke, for each
        estimator, ``predict_proba``, ``decision_function`` or ``predict``
        in that order.

    default_scoring : string, callable, dict or None (default=None)
        If not ``None``, will save scores generated by the scoring object on the
        ``scores_`` attribute each time `blend` is called.

    verbose : bool (default=False)
        When true, prints scores to stdout. `scoring` must not be ``None``.

    """

    def __init__(self, default_method="auto", default_scoring=None,
                 verbose=False):
        self.default_method = default_method
        self.default_scoring = default_scoring
        self.verbose = verbose

    @abc.abstractmethod
    def wrap_estimator(self, estimator, method=None, **kwargs):
        """Wraps an estimator and returns a stackable transformer.

        Parameters
        ----------
        estimator : predictor
            The estimator to be blended.

        method : string or None, optional (default=None)
            If not ``None``, his method will be called on the estimator instead
            of ``default_method`` to produce the output of transform. If the
            method is ``auto``, will try to invoke, for each estimator,
            ``predict_proba``, ``decision_function`` or ``predict`` in that
            order.

        Returns
        -------
        t : Transformer that implements the interface defined by
            ``BaseStackableTransformer``

        """
        pass


METRICS = dict(explained_variance=explained_variance_score,
               f1=f1_score, r2=r2_score,
               median_absolute_error=median_absolute_error,
               mean_absolute_error=mean_absolute_error,
               mean_squared_error=mean_squared_error,
               mean_squared_log_error=mean_squared_log_error,
               accuracy=accuracy_score, roc_auc=roc_auc_score,
               precision=precision_score,
               recall=recall_score,
               average_precision=average_precision_score,
               log_loss=log_loss,
               brier_score_loss=brier_score_loss,
               # Cluster metrics that use supervised evaluation
               adjusted_rand_score=adjusted_rand_score,
               homogeneity_score=homogeneity_score,
               completeness_score=completeness_score,
               v_measure_score=v_measure_score,
               mutual_info_score=mutual_info_score,
               adjusted_mutual_info_score=adjusted_mutual_info_score,
               normalized_mutual_info_score=normalized_mutual_info_score,
               fowlkes_mallows_score=fowlkes_mallows_score)


def _scoring_fn(scoring):
    if isinstance(scoring, six.string_types):
        scoring = METRICS[scoring]
    return scoring


def _scoring_dict(scoring, idx=0):
    name = "score" if idx == 0 else "score%d" % idx
    return {name: _scoring_fn(scoring)}


def _scores(ytrue, ypreds, scoring):
    if isinstance(scoring, dict):
        scoring = {name: _scoring_fn(scoring)
                   for name, scoring in scoring.items()}
    elif isinstance(scoring, six.string_types) or callable(scoring):
        scoring = _scoring_dict(scoring, 0)
    else:
        dicts = [_scoring_dict(s, i) for i, s in
                 enumerate(scoring)]
        scoring = {}
        for d in dicts:
            scoring.update(d)

    return {name: score(ytrue, ypreds)
            for name, score in scoring.items()}


def _dict_to_str(d):
    return ', '.join(["%s=%s" % (k, v) for k, v in d.items()])


def _wrap_text(t, linewidth, subsequent_indent_spaces):
    return '\n'.join(
        textwrap.wrap(t, width=linewidth, drop_whitespace=True,
                      subsequent_indent=' ' * subsequent_indent_spaces))


def _print_scores(estimator, scores):
    linewidth = np.get_printoptions()['linewidth']

    params = estimator.get_params()
    params.pop('estimator')
    scores_lines = [_wrap_text(" - scores %d: %s" % (i, _dict_to_str(score)),
                               linewidth, 4)
                    for i, score in enumerate(scores)]

    print(_wrap_text("[BLEND] %s" % _dict_to_str(params),
                     linewidth, 8))
    print('\n'.join(scores_lines))
