import numpy as np

from wolpert.wrappers import (CVStackableTransformer,
                              HoldoutStackableTransformer,
                              TimeSeriesStackableTransformer)
from sklearn.linear_model import LinearRegression
from sklearn.utils.testing import assert_almost_equal


CONSTRUCTORS = [CVStackableTransformer, HoldoutStackableTransformer,
                TimeSeriesStackableTransformer]


def test_scoring():
    scores = [[{'s1': 31.24, 's2': 731.9532}],
              [{'s1': 31.24, 's2': 975.9376}],
              [{'s1': 20., 's2': 400.},
               {'s1': 31.24, 's2': 975.9376}]]

    X = np.asarray([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
    y = np.asarray([12., -8., 3.24])
    for Constructor, expected_scores in zip(CONSTRUCTORS, scores):
        # checks for scoring
        reg = Constructor(LinearRegression(),
                          method='predict',
                          scoring={'s1': 'median_absolute_error',
                                   's2': 'mean_squared_error'})
        reg.blend(X, y)
        for i, score in enumerate(reg.scores_):
            assert_almost_equal(score['s1'], expected_scores[i]['s1'])
            assert_almost_equal(score['s2'], expected_scores[i]['s2'])
