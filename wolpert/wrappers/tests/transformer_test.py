import numpy as np

from wolpert.wrappers import (CVStackableTransformer,
                              HoldoutStackableTransformer,
                              TimeSeriesStackableTransformer)
from sklearn.linear_model import LinearRegression
from sklearn.utils.testing import assert_almost_equal

from mock import patch


CONSTRUCTORS = [CVStackableTransformer, HoldoutStackableTransformer,
                TimeSeriesStackableTransformer]

REGRESSOR = LinearRegression()

SCORING_PARAM = {'s1': 'median_absolute_error',
                 's2': 'mean_squared_error'}

EXPECTED_SCORES = [[{'s1': 31.24, 's2': 731.9532}],
                   [{'s1': 31.24, 's2': 975.9376}],
                   [{'s1': 20., 's2': 400.},
                    {'s1': 31.24, 's2': 975.9376}]]

X = np.asarray([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
y = np.asarray([12., -8., 3.24])


def _check_scores(results, expected):
    for i, score in enumerate(results):
        assert_almost_equal(score['s1'], expected[i]['s1'])
        assert_almost_equal(score['s2'], expected[i]['s2'])


def test_scoring():
    for Constructor, expected_scores in zip(CONSTRUCTORS, EXPECTED_SCORES):
        # checks for scoring
        reg = Constructor(REGRESSOR, method='predict', scoring=SCORING_PARAM)
        reg.fit_blend(X, y)
        _check_scores(reg.scores_, expected_scores)


def test_verbosity():
    for Constructor, expected_scores in zip(CONSTRUCTORS, EXPECTED_SCORES):
        reg = Constructor(REGRESSOR, method='predict', scoring=SCORING_PARAM,
                          verbose=True)
        with patch('wolpert.wrappers.base._print_scores') as mocked_print:
            reg.fit_blend(X, y)
            (called_reg, resulting_scores), _  = mocked_print.call_args
            assert(reg == called_reg)
            _check_scores(resulting_scores, expected_scores)


def test_blend_after_fit():
    for Constructor in CONSTRUCTORS:
        reg = Constructor(REGRESSOR, method='predict')
        reg.fit(X, y)
        reg.blend(X, y)
        reg.transform(X)

