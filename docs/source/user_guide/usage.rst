..  _user_guide__usage:

Using wolpert to build stacked ensembles
========================================

.. currentmodule:: wolpert

We'll build a stacked ensemble for a classification task. Let's start by loading our data:

.. testcode::

   from sklearn.datasets import make_classification
   RANDOM_STATE = 888
   X, y = make_classification(n_samples=1000, random_state=RANDOM_STATE)

Now let's choose some base models to build our first layer. We'll go with a KNN, SVM, random forest and extremely randomized trees, all available on scikit learn. It's worth noting that any scikit learn compatible model can be used here:

.. testcode::

   from sklearn.neighbors import KNeighborsClassifier
   from sklearn.svm import SVC
   from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

   knn = KNeighborsClassifier()
   svc = SVC(random_state=RANDOM_STATE, probability=True)
   rf = RandomForestClassifier(random_state=RANDOM_STATE)
   et = ExtraTreesClassifier(random_state=RANDOM_STATE)

Now let's test each classifier alone and see what we get. We'll use a cross validation with 3 folds for this and evaluate using log loss.

.. testcode::

   import numpy as np
   from sklearn.model_selection import StratifiedKFold
   from sklearn.metrics import log_loss

   def evaluate(clf, clf_name, X, y):
       kfold = StratifiedKFold(n_splits=3, random_state=RANDOM_STATE)
       scores = []
       for train_idx, test_idx in kfold.split(X, y):
           ypreds =  clf.fit(X[train_idx], y[train_idx]).predict_proba(X[test_idx])
           scores.append(log_loss(y[test_idx], ypreds))
       print("Logloss for %s: %.5f (+/- %.5f)" % (clf_name, np.mean(scores), np.std(scores)))
       return scores

   evaluate(knn, "KNN classifier", X, y)
   evaluate(rf, "Random Forest", X, y)
   evaluate(svc, "SVM classifier", X, y)
   evaluate(et, "ExtraTrees", X, y)

.. testoutput::

    Logloss for KNN classifier: 0.65990 (+/- 0.10233)
    Logloss for Random Forest: 0.47338 (+/- 0.21536)
    Logloss for SVM classifier: 0.24082 (+/- 0.02127)
    Logloss for ExtraTrees: 0.53194 (+/- 0.08710)

The best model here is the SVM. We now have a baseline to build our stacked ensemble.

The first thing that needs to be decided is the stacking strategy we'll use. The dataset is pretty small, so it's ok to go for a cross validation strategy.

.. note::
   To know more about the strategies implemented in wolpert, read the :ref:`strategies chapter <user_guide__strategies>`.

The easiest way to do so is using the helper function :func:`pipeline.make_stack_layer`. This function takes a list of steps to be used to build a layer and the blending wrapper.

.. testcode::

   from wolpert import make_stack_layer

   layer0 = make_stack_layer(knn, rf, svc, et, blending_wrapper='cv')

Ok, now that we have our first layer, let's put a very simple model on top of it and see how it goes. For validating the meta estimator, we must first generate the blended dataset (see :meth:`pipeline.StackingLayer.fit_blend` for more info):

.. testcode::

   Xt, t_indexes = layer0.fit_blend(X, y)
   yt = y[t_indexes]

Now we can build our meta estimator and evaluate it:

.. testcode::

   from sklearn.linear_model import LogisticRegression

   meta = LogisticRegression(random_state=RANDOM_STATE)

   evaluate(meta, "Meta estimator", Xt, yt)

.. testoutput::

   Logloss for Meta estimator: 0.22706 (+/- 0.02656)

Notice the score is already better than our best classifier on the first layer. Now let's construct the final model. To do this we'll use the class :class:`pipeline.StackingPipeline`. This acts like scikit learn's `Pipeline <http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html>`_: each output from a step is piped to the next step. The difference is that ``StackingPipeline`` will use blending when fitting the models to a dataset.

.. testcode::

   from wolpert import StackingPipeline

   stacked_clf = StackingPipeline([("l0", layer0), ("meta", meta)])

.. note::

   The final class has a helper method for evaluating it, called :meth:`score <pipeline.StackingPipeline.score>`, but it depends on scikit learn's `cross_validate <http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html>`_ function and this function doesn't allow us to pass the method we want it to call on the estimator, always calling ``predict``. Here's an example:

   .. testcode::

      stacked_clf.fit(X, y)
      scores = stacked_clf.score(X, y, scoring='neg_log_loss', cv=3)

      print("Logloss for Stacked classifier: %.5f (+/- %.5f)" % (-np.mean(scores["test_score"]),
                                                                 np.std(scores["test_score"])))

   .. testoutput::

      Logloss for Stacked classifier: 0.28145 (+/- 0.03106)

   Notice the score is worse than our handcrafted evaluation.

Now let's see how we can improve our model.

Multi-layer stacked ensemble
----------------------------

Let's try a simple approach: we'll grab the best two models from the first layer and create a second one. We'll also use :ref:`restacking <user_guide__intro_restacking>` on the first layer. The final meta estimator will remain the same.

.. testcode::

   layer0_clfs = [KNeighborsClassifier(),
                  SVC(random_state=RANDOM_STATE, probability=True),
                  RandomForestClassifier(random_state=RANDOM_STATE),
                  ExtraTreesClassifier(random_state=RANDOM_STATE)]

   layer1_clfs = [SVC(random_state=RANDOM_STATE, probability=True),
                  RandomForestClassifier(random_state=RANDOM_STATE)]



   layer0 = make_stack_layer(*layer0_clfs, blending_wrapper="cv", restack=True)
   layer1 = make_stack_layer(*layer1_clfs, blending_wrapper="cv")

   # first let's build the pipeline without the final estimator to see its
   # performance
   transformer = StackingPipeline([("layer0", layer0), ("layer1", layer1)])
   Xt, t_indexes = transformer.fit_blend(X, y)

   evaluate(meta, "Meta classificator with two layers", X, y)

.. testoutput::

   Logloss for Meta classificator with two layers: 0.28145 (+/- 0.03106)

Well, it didn't help. Let's keep the old model then. There are some reasons for this: maybe our model is too complex for the dataset, so a single layer is better.

Model selection
---------------

We can access all attributes on all estimators just like in a regular scikit learn pipeline. With that we can follow the same steps for model selection:

.. testcode::

   from sklearn.model_selection import GridSearchCV

   param_grid = {
       "l0__svc__method": ["predict_proba", "decision_function"],
       "l0__svc__estimator__C": [.1, 1., 10]}

   clf_cv = GridSearchCV(stacked_clf, param_grid, scoring="neg_log_loss", n_jobs=-1)
   clf_cv.fit(X, y)
   test_scores = clf_cv.cv_results_["mean_test_score"]
   print("Logloss for best model on CV: %.5f (+/- %.5f)" % (-test_scores.mean(), test_scores.std()) )

.. testoutput::

   Logloss for best model on CV: 0.22491 (+/- 0.00259)

.. note::

   Remember that this score should be compared to the one from the :meth:`score <wolpert.StackingPipeline.score>` method.

Wrappers API
------------

Up until now we relied on the default arguments for wrapping our models. To have more control over this arguments, one can use the :mod:`wolpert.wrappers` API. Let's build our model now using a 10-fold cross validation. For this, we'll use the :class:`CVWrapper <wolpert.wrappers.CVWrapper>` helper class.

.. testcode::

   from wolpert.wrappers import CVWrapper

   cv_wrapper = CVWrapper(cv=10, n_cv_jobs=-1)

The main method for this class is :meth:`wrap_estimator <wolpert.wrappers.CVWrapper.wrap_estimator>`, that receives an estimator and returns it wrapped with a class that exposes the methods ``blend`` and ``fit_blend``. We can also pass it to the :paramref:`wolpert.make_stack_layer.blending_wrapper` argument and it will be used to wrap all the estimators on the layer:

.. testcode::

   layer0 = make_stack_layer(knn, rf, svc, et, blending_wrapper=cv_wrapper)
   stacked_clf = StackingPipeline([("l0", layer0), ("meta", meta)])

Just out of curiosity, here's the model performance:

.. testcode::

   Xt, t_indexes = layer0.fit_blend(X, y)

   evaluate(meta, "Meta classificator with CV=10 on first layer", Xt, y[t_indexes])

.. testoutput::

   Logloss for Meta classificator with CV=10 on first layer: 0.22241 (+/- 0.03292)

Inner estimators performance
----------------------------

Sometimes it's useful to keep track of the performance of each estimator inside an ensemble. To do so, every wrapper exposes a parameter called ``scoring``. It works simmilarly to scikit learn's `scoring parameter <http://scikit-learn.org/stable/modules/cross_validation.html#computing-cross-validated-metrics>`_, but it uses the metrics functions directly instead of a scorer. We do so because we want to avoid retraining models inside an ensemble, as it's already an expensive computation as it is.

When ``scoring`` is set, everytime a blend happens, it will store the scoring results in the ``scores_`` parameter. It's a list of dicts where each key is the name of the score used. If not supplied, the name will be ``score`` with an integer suffix.

Each metric may be a string (for the builtin metrics) or a function that receives the true labels and the predicted labels and outputs a single floating number, denoting the score for this pair. The ``scoring`` parameters accepts a single metric, a list of metrics or a dict where the key is the metric name and the value is the metric itself.

.. testcode::

   import numpy as np

   from wolpert.wrappers import CVStackableTransformer
   from sklearn.linear_model import LinearRegression
   from sklearn.metrics import mean_squared_error

   X = np.asarray([[1, 2], [3, 4], [5, 6], [7, 8]])
   y = np.asarray([0, 1, 0, 1])

   # With a single metric
   cvs = CVStackableTransformer(
       LinearRegression(), scoring='mean_absolute_error')
   cvs.fit_blend(X, y)
   print(cvs.scores_)

   # a list of metrics
   cvs = CVStackableTransformer(
       LinearRegression(), scoring=['mean_absolute_error',
                                    mean_squared_error])
   cvs.fit_blend(X, y)
   print(cvs.scores_)

   # a dict of metrics
   cvs = CVStackableTransformer(
       LinearRegression(), scoring={'mae': 'mean_absolute_error',
                                    'mse': mean_squared_error})
   cvs.fit_blend(X, y)
   import pprint
   pprint.pprint(cvs.scores_)


.. testoutput::
   :options: +ELLIPSIS

   [{'score': 1.380...}]
   [{'score': 1.380..., 'score1': 2.294...}]
   [{'mae': 1.380..., 'mse': 2.294...}]

We can also use the ``verbose`` parameter to keep track of the models performances. It will print the results to stdout.

.. testcode::

   cvs = CVStackableTransformer(
       LinearRegression(), scoring='mean_absolute_error', verbose=True)
   cvs.fit_blend(X, y)

.. testoutput::
   :options: +ELLIPSIS

   [BLEND] cv=3, estimator=<class
           'sklearn.linear_model.base.LinearRegression'>,
           estimator__copy_X=True, estimator__fit_intercept=True,
           estimator__n_jobs=1, estimator__normalize=False, method=auto,
           n_cv_jobs=1, scoring=mean_absolute_error, verbose=True
    - scores 0: score=1.380...
