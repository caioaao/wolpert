import numpy as np

from sklearn.utils.testing import (assert_equal, assert_array_equal,
                                   assert_array_almost_equal,
                                   assert_almost_equal)
from sklearn.metrics import log_loss, roc_auc_score

from wolpert.wrappers.base import _scores

RANDOM_STATE = 498595


def test_scores():
    np.random.seed(RANDOM_STATE)
    ytrue = np.random.randint(2, size=10)
    ypreds = np.random.rand(10)

    # check that it works with a single predefined score
    logloss = _scores(ytrue, ypreds, 'log_loss')['score']
    assert_almost_equal(logloss, 1.2694, decimal=4)

    # check that works with metric
    logloss = _scores(ytrue, ypreds, log_loss)['score']
    assert_almost_equal(logloss, 1.2694, decimal=4)

    # check that it works with list
    scores = _scores(ytrue, ypreds, ('log_loss', roc_auc_score))
    assert_array_equal(["score", "score1"], list(scores.keys()))
    assert_array_almost_equal([1.2694, 0.24], list(scores.values()),
                              decimal=4)

    # check that it works with dict
    scores = _scores(ytrue, ypreds, {"logloss": 'log_loss',
                                     "roc_auc": roc_auc_score})
    assert_array_equal(["logloss", "roc_auc"], list(scores.keys()))
    assert_array_almost_equal([1.2694, 0.24], list(scores.values()),
                              decimal=4)
