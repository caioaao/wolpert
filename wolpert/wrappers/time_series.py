import numpy as np

from sklearn.base import clone
from sklearn.utils import indexable
from sklearn.utils.validation import _num_samples
from sklearn.model_selection._split import _BaseKFold

from .base import BaseStackableTransformer, BaseWrapper


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
                           n_samples - (self.test_set_size + self.offset) + 1)

        for train_end in train_ends:
            test_start = train_end + self.offset
            test_end = train_end + self.offset + self.test_set_size
            if self.max_train_size and self.max_train_size <= test_start:
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

    n_splits : int, optional (default=None)
        Number of splits. Must be at least 1. If ``None``, will use all
        available data.

    min_train_size : int, optional (default=1)
        Minimum size for a single training set.

    max_train_size : int, optional (default=None)
        Maximum size for a single training set.
    """
    def __init__(self, estimator, method='auto', offset=0, test_set_size=1,
                 n_splits=None, min_train_size=1, max_train_size=None):
        super(TimeSeriesStackableTransformer, self).__init__(estimator, method)
        self.offset = offset
        self.test_set_size = test_set_size
        self.n_splits = n_splits
        self.min_train_size = min_train_size
        self.max_train_size = max_train_size

