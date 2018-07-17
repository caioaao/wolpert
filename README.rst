|build-status| |docs| |package-status|

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

Building a simple model
-----------------------

First we need the layers of our model. The simplest way is using the helper function `make_stack_layer <http://wolpert.readthedocs.io/en/latest/generated/wolpert.pipeline.html#wolpert.pipeline.make_stack_layer>`_:

.. testcode::

     from sklearn.ensemble import RandomForestClassifier
     from sklearn.svm import SVC
     from sklearn.neighbors import KNeighborsClassifier
     from sklearn.linear_model import LogisticRegression
     from wolpert.pipeline import make_stack_layer, StackingPipeline
     
     layer0 = make_stack_layer(SVC(), KNeighborsClassifier(),
                               RandomForestClassifier(),
                               blending_wrapper='holdout')
                               
     clf = StackingPipeline([('l0', layer0),
                             ('l1', LogisticRegression())])

And that's it! And ``StackingPipeline`` inherits a scikit learn class: the `Pipeline <http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html>`_, so it works just the same::

    clf.fit(Xtrain, ytrain)
    ypreds = clf.predict_proba(Xtest)

This is just the basic example, but there are several ways of building a stacked ensemble with this framework. Make sure to check the `User Guide <http://wolpert.readthedocs.io/en/latest/user_guide.html>`_ to know more.

.. |build-status| image:: https://circleci.com/gh/caioaao/wolpert.png?style=shield
    :alt: CircleCI
    :scale: 100%
    :target: https://circleci.com/gh/caioaao/wolpert

.. |docs| image:: https://readthedocs.org/projects/wolpert/badge/?verion=latest
    :alt: Documentation status
    :scale: 100%
    :target: https://wolpert.readthedocs.io/en/latest/?badge=latest

.. |package-status| image:: https://badge.fury.io/py/wolpert.svg
    :alt: PyPI version
    :scale: 100%
    :target: https://badge.fury.io/py/wolpert
