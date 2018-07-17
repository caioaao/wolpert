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

TODO
