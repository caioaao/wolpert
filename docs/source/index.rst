Wolpert, a stacked generalization framework
===========================================

Wolpert is a `scikit-learn <http://scikit-learn.org>`_ compatible framework for easily building stacked ensembles. It supports:

* Different stacking strategies
* Multi-layer models
* Different weights for each transformer
* Easy to make it distributed

Quickstart
==========

Install
-------

The easiest way to install is using pip. Just run ``pip install wolpert`` and you're ready to go.

Build a simple model
--------------------

First we need the layers of our model. The simplest way is using the helper function
:func:`wolpert.pipeline.make_stack_layer`::

     from sklearn.ensemble import RandomForestClassifier
     from sklearn.svm import SVC
     from sklearn.neighbors import KNNClassifer
     from wolpert.pipeline import make_stack_layer
     
     layer0 = make_stack_layer(SVC(), KNNClassifier(),
                               RandomForestClassifier(),
                               blending_wrapper='holdout')

This function will wrap each estimator on
:class:`wolpert.wrappers.HoldoutStackableTransformer` and join them with
:class:`wolpert.pipeline.StackingLayer`. This final class inherits from scikit
learn's `FeatureUnion <http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.FeatureUnion.html>`_,
so all the methods are inherited here. Now, for finishing our model, we must add
a final estimator on top of that layer. We do this by using
:class:`wolpert.pipeline.StackingPipeline`::

     from wolpert.pipeline import StackingPipeline
     from sklearn.linear_model import LogisticRegression
     
     clf = StackingPipeline([('l0', layer0),
                             ('l1', LogisticRegression())])

And that's it! And this also inherits a scikit learn class: the `Pipeline <http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html>`_, so it works just the same::

    clf.fit(Xtrain, ytrain)
    ypreds = clf.predict_proba(Xtest)

This is just the basic example, but there are several ways of building a stacked ensemble with this framework. Make sure to check the :ref:`User Guide <user_guide>` to know more.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   user_guide
   api_docs
     
Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
