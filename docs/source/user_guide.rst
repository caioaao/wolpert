.. _user_guide:
   
User Guide
==========

Stacked generalization is another method of combining estimators to reduce
their biases [W1992]_ by combining several estimators (possibly non-linearly)
stacked together in layers. Each layer will contain estimators and their
predictions are used as features to the next layer.

As stacked generalization is a generic framework for combining supervised
estimators, it works with regression and classification problems. The API
reflects that, so it's the same for both categories.

The intent of this user guide is to serve both as an introduction to stacked
generalization and as a manual for using the framework.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   user_guide/intro
   user_guide/basic_usage
   user_guide/strategies
   user_guide/restack

.. topic:: References

 .. [W1992] D. H. Wolpert, "Stacked Generalization", Neural Networks, Vol. 5, No. 5, 1992.
