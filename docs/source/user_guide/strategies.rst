.. _user_guide__strategies:

Stacking strategies
===================

Currently wolpert supports two strategies, each with its pros and cons:

Stacking with cross validation
------------------------------

.. currentmodule:: wolpert

This strategy, implemented in the class :class:`wrappers.CVStackableTransformer` uses the predictions from cross validation to build the next data set. This means all the samples from the training set will be available to the next layer:

.. doctest::

   >>> import numpy as np
   >>> from wolpert.wrappers import CVStackableTransformer
   >>> from sklearn.linear_model import LogisticRegression
   >>> X = np.asarray([[1, 2], [3, 4], [5, 6], [7, 8]])
   >>> y = np.asarray([0, 1, 0, 1])
   >>> wrapped_clf = CVStackableTransformer(LogisticRegression(random_state=1), cv=2)
   >>> wrapped_clf.fit_blend(X, y)  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
   (array([[0.52444526, 0.47555474],
           [0.48601904, 0.51398096],
           [0.15981917, 0.84018083],
           [0.08292124, 0.91707876]]), ...)

.. note::
   The first argument returned by ``blend`` / ``fit_blend`` is the transformed training set and the second one is the indexes of the rows present on this transformed data, but don't worry about the second argument now.

As you can see, the data transformed by blending has the same number of rows as the input. For this estimator, this should **always** be true. This is good because we'll have more data to train the subsequent layers, but it comes with a downside: as we fit the layer to the whole training set after blending, the probability distribution for the transformed data set will change from train to test. But don't worry too much: in practice the results are still good.

In multi-layer stackings, this may be the only choice. This is because if we choose another strategy, our training set will become exponentially smaller from layer to layer.


Stacking with holdout set
-------------------------

.. currentmodule:: wolpert

When the training set is too big, using a k-fold split may be too slow. For this cases, we have :class:`wrappers.HoldoutStackableTransformer`. This strategy splits the data into two sets: training and holdout. The models are trained using the training set and outputs predictions for the holdout set. This means we'll have fewer rows to train the subsequent layers. See the following example:

.. doctest::

   >>> import numpy as np
   >>> from wolpert.wrappers import HoldoutStackableTransformer
   >>> from sklearn.linear_model import LogisticRegression
   >>> X = np.asarray([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
   >>> y = np.asarray([0, 1, 0, 1, 1])
   >>> wrapped_clf = HoldoutStackableTransformer(LogisticRegression(random_state=1),
   ...                                           holdout_size=.5)
   >>> wrapped_clf.fit_blend(X, y)  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    (array([[0.34674758, 0.65325242],
            [0.0649691 , 0.9350309 ],
            [0.21229721, 0.78770279]]),
     array([1, 4, 2]))

As you can see from the indexes array, only predictions for rows 1, 2 and 4 were returned on the transformed data set. This will be faster than :class:`wrappers.CVStackableTransformer` and, if :meth:`fit_to_all_data <wrappers.HoldoutStackableTransformer>` is set to ``False``, train and test sets will come from the same probability distribution.

Stacking with time series
-------------------------

.. currentmodule:: wolpert

When dealing with time series data, extra care must be taken to avoid leakages. :class:`wrappers.TimeSeriesStackableTransformer` handles part of this issue by making splits that never violate the original ordering of the data or, in other words, indexes on the training set will always be smaller than indexes on the test set for all splits.

It works by walking in an ascending order, growing the training set on each split and predicting on the data after the training set. It's almost the same as `sklearn's TimeSeriesSplit <http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html>`_, but with some knobs that we found more useful. Here's an example:

.. doctest::

   >>> import numpy as np
   >>> from wolpert.wrappers import TimeSeriesStackableTransformer
   >>> from sklearn.linear_model import LogisticRegression
   >>> X = np.asarray([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
   >>> y = np.asarray([0, 1, 0, 1, 1])
   >>> wrapped_clf = TimeSeriesStackableTransformer(LogisticRegression(random_state=1),
   ...                                              min_train_size=2)
   >>> wrapped_clf.fit_blend(X, y)  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    (array([[0.15981917, 0.84018083],
            [0.74725218, 0.25274782]]),
     array([2, 3]))

These were the splits used to generate the blended data set:

#. Train on indexes ``0`` and ``1``, predict for index ``2``;
#. Train on indexes ``0``, ``1`` and ``2``, predict for index ``3``.

This resembles the `leave-one-out cross validation <https://en.wikipedia.org/wiki/Cross-validation_(statistics)#Leave-one-out_cross-validation>`_, but :class:`wrappers.TimeSeriesStackableTransformer` provides other options, so make sure to check its documentation. For example, to make a blended dataset that resembles `leave-p-out cross-validation <https://en.wikipedia.org/wiki/Cross-validation_(statistics)#Leave-p-out_cross-validation>`_, all you have to do is change the :paramref:`wrappers.TimeSeriesStackableTransformer.test_set_size`:

.. doctest::

   >>> import numpy as np
   >>> from wolpert.wrappers import TimeSeriesStackableTransformer
   >>> from sklearn.linear_model import LogisticRegression
   >>> X = np.asarray([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
   >>> y = np.asarray([0, 1, 0, 1, 1, 0])
   >>> wrapped_clf = TimeSeriesStackableTransformer(LogisticRegression(random_state=1),
   ...                                              min_train_size=2,
   ...                                              test_set_size=2)
   >>> wrapped_clf.fit_blend(X, y)  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    (array([[0.15981917, 0.84018083],
            [0.08292124, 0.91707876]]),
     array([2, 3]))

.. note::

   Notice that in the last example the last sample was dropped from the transformed data. This is because, when the remaining samples are not enough to satisfy the test set size constraint, they are dropped.
