from sklearn.utils.testing import assert_equal, assert_array_equal


def check_estimator(estimator, X, y, **fit_params):
    # checks that we can fit_transform to the data
    Xt = estimator.fit_transform(X, y, **fit_params)

    # checks that we get a column vector
    assert_equal(Xt.ndim, 2)

    # checks that `fit` is available
    estimator.fit(X, y, **fit_params)

    # checks that we can transform the data after it's fitted
    Xt2 = estimator.transform(X)

    # checks that transformed data is always a column vector
    assert_equal(Xt.ndim, 2)

    # checks that transform is equal to fit_transform
    assert_array_equal(Xt, Xt2)

    # checks for determinism: every `transform` should yield the same result
    for i in range(10):
        assert_array_equal(Xt2, estimator.transform(X))

    # checks that `blend` is availabe
    Xt_blend, indexes = estimator.blend(X, y, **fit_params)

    # checks that blended data is always a column vector
    assert_equal(Xt_blend.ndim, 2)

    # checks that indexes have the same amount of indexes as transformed data
    assert_equal(Xt_blend.shape[0], indexes.shape[0])

    # checks that `fit_blend` is available
    Xt_fit_blend, indexes = estimator.fit_blend(X, y, **fit_params)

    # checks that fit_blend and blend returns the same results
    assert_array_equal(Xt_blend, Xt_fit_blend)

    # checks that indexes have the same amount of indexes as transformed data
    assert_equal(Xt_blend.shape[0], indexes.shape[0])

