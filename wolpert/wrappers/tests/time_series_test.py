import numpy as np
import pytest

from sklearn.utils.testing import assert_array_equal

from wolpert.wrappers.time_series import TimeSeriesSplit


def assert_splits_equal(a, b):
    for (train_a, test_a), (train_b, test_b) in zip(a, b):
        assert_array_equal(train_a, train_b)
        assert_array_equal(test_a, test_b)


def test_split_class():
    # with default values
    arr = np.arange(3)
    ts = TimeSeriesSplit()
    assert_splits_equal(ts.split(arr),
                        [[[0], [1]],
                         [[0, 1], [2]]])

    # with offset
    arr = np.arange(5)
    ts = TimeSeriesSplit(offset=2)
    assert_splits_equal(ts.split(arr),
                        [[[0], [3]],
                         [[0, 1], [4]]])

    # with min_train_size
    arr = np.arange(3)
    ts = TimeSeriesSplit(min_train_size=2)
    assert_splits_equal(ts.split(arr),
                        [[[0, 1], [2]]])

    # with max_train_size
    arr = np.arange(3)
    ts = TimeSeriesSplit(max_train_size=1)
    assert_splits_equal(ts.split(arr),
                        [[[0], [1]],
                         [[1], [2]]])

    # with test_set_size
    arr = np.arange(3)
    ts = TimeSeriesSplit(test_set_size=2)
    assert_splits_equal(ts.split(arr),
                        [[[0], [1, 2]]])

    # min/max train_size validation
    arr = np.arange(3)
    ts = TimeSeriesSplit(min_train_size=10, max_train_size=2)
    splits = ts.split(arr)
    with pytest.raises(ValueError) as _:
        next(splits)



