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
   The first argument returned by ``blend`` / ``fit_blend`` is the transformed training set and the second one are the indexes of the rows present on this transformed data, but don't worry about the second argument now.

As you can see, the data transformed by blending has the same number of rows as the input.


Stacking with holdout set
-------------------------

