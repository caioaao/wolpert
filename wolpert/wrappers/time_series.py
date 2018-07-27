import numpy as np
import scipy.sparse as sp

from sklearn.utils import indexable
from sklearn.utils.validation import _num_samples
from sklearn.model_selection._validation import _fit_and_predict
from sklearn.model_selection._split import check_cv
from sklearn.base import is_classifier, clone
from sklearn.preprocessing import LabelEncoder
from sklearn.externals.joblib import Parallel, delayed, logger


from .base import BaseStackableTransformer, BaseWrapper


def _cross_val_predict(estimator, X, y=None, groups=None, cv=None, n_jobs=1,
                       verbose=0, fit_params=None, pre_dispatch='2*n_jobs',
                       method='predict'):
    """Generate cross-validated estimates for each input data point

    Copied from scikit learn, with the only difference that it doesn't check for
    permutations.

    Parameters
    ----------
    estimator : estimator object implementing 'fit' and 'predict'
        The object to use to fit the data.

    X : array-like
        The data to fit. Can be, for example a list, or an array at least 2d.

    y : array-like, optional, default: None
        The target variable to try to predict in the case of
        supervised learning.

    groups : array-like, with shape (n_samples,), optional
        Group labels for the samples used while splitting the dataset into
        train/test set.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 3-fold cross validation,
        - integer, to specify the number of folds in a `(Stratified)KFold`,
        - An object to be used as a cross-validation generator.
        - An iterable yielding train, test splits.

        For integer/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

    n_jobs : integer, optional
        The number of CPUs to use to do the computation. -1 means
        'all CPUs'.

    verbose : integer, optional
        The verbosity level.

    fit_params : dict, optional
        Parameters to pass to the fit method of the estimator.

    pre_dispatch : int, or string, optional
        Controls the number of jobs that get dispatched during parallel
        execution. Reducing this number can be useful to avoid an
        explosion of memory consumption when more jobs get dispatched
        than CPUs can process. This parameter can be:

            - None, in which case all the jobs are immediately
              created and spawned. Use this for lightweight and
              fast-running jobs, to avoid delays due to on-demand
              spawning of the jobs

            - An int, giving the exact number of total jobs that are
              spawned

            - A string, giving an expression as a function of n_jobs,
              as in '2*n_jobs'

    method : string, optional, default: 'predict'
        Invokes the passed method name of the passed estimator. For
        method='predict_proba', the columns correspond to the classes
        in sorted order.

    Returns
    -------
    predictions : ndarray
        This is the result of calling ``method``

    See also
    --------
    cross_val_score : calculate score for each CV split

    cross_validate : calculate one or more scores and timings for each CV split

    Notes
    -----
    In the case that one or more classes are absent in a training portion, a
    default score needs to be assigned to all instances for that class if
    ``method`` produces columns per class, as in {'decision_function',
    'predict_proba', 'predict_log_proba'}.  For ``predict_proba`` this value is
    0.  In order to ensure finite output, we approximate negative infinity by
    the minimum finite float value for the dtype in other cases.

    Examples
    --------
    >>> from sklearn import datasets, linear_model
    >>> from sklearn.model_selection import cross_val_predict
    >>> diabetes = datasets.load_diabetes()
    >>> X = diabetes.data[:150]
    >>> y = diabetes.target[:150]
    >>> lasso = linear_model.Lasso()
    >>> y_pred = cross_val_predict(lasso, X, y)

    """
    X, y, groups = indexable(X, y, groups)

    cv = check_cv(cv, y, classifier=is_classifier(estimator))

    if method in ['decision_function', 'predict_proba', 'predict_log_proba']:
        le = LabelEncoder()
        y = le.fit_transform(y)

    # We clone the estimator to make sure that all the folds are
    # independent, and that it is pickle-able.
    parallel = Parallel(n_jobs=n_jobs, verbose=verbose,
                        pre_dispatch=pre_dispatch)
    prediction_blocks = parallel(delayed(_fit_and_predict)(
        clone(estimator), X, y, train, test, verbose, fit_params, method)
        for train, test in cv.split(X, y, groups))

    # Concatenate the predictions
    predictions = [pred_block_i for pred_block_i, _ in prediction_blocks]
    test_indices = np.concatenate([indices_i
                                   for _, indices_i in prediction_blocks])

    # Check for sparse predictions
    if sp.issparse(predictions[0]):
        predictions = sp.vstack(predictions, format=predictions[0].format)
    else:
        predictions = np.concatenate(predictions)
    return predictions, test_indices


class TimeSeriesSplit:
    """Time Series cross-validator

    Provides train/test indices to split time series data samples
    that are observed at fixed time intervals, in train/test sets.
    In each split, test indices must be higher than before, and thus shuffling
    in cross validator is inappropriate.

    This cross-validation object is a variation of :class:`KFold`.
    In the kth split, it returns first k folds as train set and the
    (k+1)th fold as test set.

    Note that unlike standard cross-validation methods, successive
    training sets are supersets of those that come before them.

    Read more in the :ref:`User Guide <cross_validation>`.

    Parameters
    ----------
    min_train_size : int, optional (default=1)
        Minimum size for a single training set.

    max_train_size : int, optional (default=None)
        Maximum size for a single training set.

    offset : integer, optional (default=0)
        Number of rows to skip after the last train split rows

    drop : integer, optional (default=0)
        Number of rows to skip in the beginning

    test_set_size : integer, optional (default=1)
        Size of the test set. This will also be the amount of rows added to the
        training set at each iteration

    """
    def __init__(self, offset=0, test_set_size=1, drop=0, min_train_size=1,
                 max_train_size=None):
        self.test_set_size = test_set_size
        self.offset = offset
        self.drop = drop
        self.min_train_size = min_train_size
        self.max_train_size = max_train_size

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        y : array-like, shape (n_samples,)
            Always ignored, exists for compatibility.

        groups : array-like, with shape (n_samples,), optional
            Always ignored, exists for compatibility.

        Yields
        ------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.
        """
        if self.max_train_size and self.min_train_size > self.max_train_size:
            raise ValueError("min_train_size cannot be bigger than max_train_size")

        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        indices = np.arange(n_samples)
        train_ends = range(self.min_train_size,
                           n_samples - (self.test_set_size + self.offset),
                           self.test_set_size)

        for train_end in train_ends:
            test_start = train_end + self.offset
            test_end = train_end + self.offset + self.test_set_size
            if self.max_train_size and self.max_train_size < train_end:
                train_start = train_end - self.max_train_size
            else:
                train_start = 0
            yield (indices[train_start:train_end],
                   indices[test_start:test_end])


class TimeSeriesStackableTransformer(BaseStackableTransformer):
    """Transformer to turn estimators into meta-estimators for model stacking

    Each split is composed by a train set containing the first ``t`` rows in the
    data set and a test set composed of rows ``t+k`` to ``t+k+n``, where ``k``
    and ``n`` are the `offset` and `test_set_size` parameters.

    Parameters
    ----------
    estimator : predictor
        The estimator to be blended.

    method : string, optional (default='auto')
        This method will be called on the estimator to produce the output of
        transform. If the method is ``auto``, will try to invoke, for each
        estimator, ``predict_proba``, ``decision_function`` or ``predict``
        in that order.

    offset : integer, optional (default=0)
        Number of rows to skip after the last train split rows

    test_set_size : integer, optional (default=1)
        Size of the test set. This will also be the amount of rows added to the
        training set at each iteration

    min_train_size : int, optional (default=1)
        Minimum size for a single training set.

    max_train_size : int, optional (default=None)
        Maximum size for a single training set.

    n_cv_jobs : int, optional (default=1)
        Number of jobs to be passed to ``cross_val_predict`` during
        ``blend``.

    """
    def __init__(self, estimator, method='auto', offset=0, test_set_size=1,
                 min_train_size=1, max_train_size=None, n_cv_jobs=1):
        super(TimeSeriesStackableTransformer, self).__init__(estimator, method)
        self.offset = offset
        self.test_set_size = test_set_size
        self.min_train_size = min_train_size
        self.max_train_size = max_train_size
        self.n_cv_jobs = n_cv_jobs

    def _splitter(self):
        return TimeSeriesSplit(
            offset=self.offset, test_set_size=self.test_set_size,
            min_train_size=self.min_train_size,
            max_train_size=self.max_train_size)

    def blend(self, X, y, **fit_params):
        """Transform dataset using time series split.

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
        ts = self._splitter()

        self.estimator_ = clone(self.estimator)
        Xt, indexes = _cross_val_predict(self.estimator_, X, y, cv=ts,
                                         n_jobs=self.n_cv_jobs,
                                         method=self._estimator_function_name)
        self.estimator_ = None

        if Xt.ndim == 1:
            Xt = Xt.reshape(-1, 1)
        return Xt, indexes

    def fit_blend(self, X, y, **fit_params):
        """Transform dataset using cross validation and fits the estimator to the
        entire dataset.

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
        blend_results = self.blend(X, y, **fit_params)
        self.fit(X, y, **fit_params)
        return blend_results


class TimeSeriesWrapper(BaseWrapper):
    """Helper class to wrap estimators with ``TimeSeriesStackableTransformer``

    Parameters
    ----------

    default_method : string, optional (default='auto')
        This method will be called on the estimator to produce the output of
        transform. If the method is ``auto``, will try to invoke, for each
        estimator, ``predict_proba``, ``decision_function`` or ``predict``
        in that order.

    offset : integer, optional (default=0)
        Number of rows to skip after the last train split rows

    test_set_size : integer, optional (default=1)
        Size of the test set. This will also be the amount of rows added to the
        training set at each iteration

    min_train_size : int, optional (default=1)
        Minimum size for a single training set.

    max_train_size : int, optional (default=None)
        Maximum size for a single training set.

    n_cv_jobs : int, optional (default=1)
        Number of jobs to be passed to ``cross_val_predict`` during
        ``blend``.

    """

    def __init__(self, default_method='auto', offset=0, test_set_size=1,
                 min_train_size=1, max_train_size=None, n_cv_jobs=1):
        super(TimeSeriesWrapper, self).__init__(default_method)
        self.offset = offset
        self.test_set_size = test_set_size
        self.min_train_size = min_train_size
        self.max_train_size = max_train_size
        self.n_cv_jobs = n_cv_jobs

    def wrap_estimator(self, estimator, method=None, **kwargs):
        """Wraps an estimator and returns a transformer that is suitable for stacking.

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
        t : TimeSeriesStackableTransformer

        """
        method = method or self.default_method

        return TimeSeriesStackableTransformer(
            estimator, method=method, offset=self.offset,
            test_set_size=self.test_set_size,
            min_train_size=self.min_train_size,
            max_train_size=self.max_train_size,
            n_cv_jobs=self.n_cv_jobs)
