# -*- coding: utf-8 -*-
'''
A module of tools to evaluate predictive models.
'''
import numpy as np
from sklearn import cross_validation


class EvalError(Exception):
    '''Errors in evaluating the model, for example
    a missing predict function.'''
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return self.msg


class GenModel(object):
    '''
    A general (binary classification) model class, mostly for
    testing and explanatory purposes.

    :param function fit:
        the fit function of the model
    :param function predict:
        the prediction function of the model
    :param function predict_proba:
        the probability prediction function of the model,
        computes the probability that an observation belongs to the first class

    '''

    def __init__(self, fit=None, predict=None, predict_proba=None):
        self.fit = fit
        self.predict = predict
        self.predict_proba = predict_proba


class Evaluate(object):
    '''
    Evaluate a predictive binary classification model
    on a choice of metrics including accuracy, AUC,
    precision and recall. The model is assumed to have functions

    * *fit* (for cross validation): takes as input the training set
      features and responses and fits the model to the training set
    * *predict* (for accuracy, precision, recall
      and cross validation): takes as input a list of observations and
      outputs a list of predictions
    * *predict_proba* (for AUC and drawing the ROC curve): takes as input
      a list of observations and outputs a list of probabilities that the
      response belongs to the first class

    For example, we make a dummy test set and model:

    >>> model = GenModel(predict = lambda x : ['blue' for i in x])
    >>> test_features = range(20)
    >>> test_responses = np.array(
    ...     ['blue' if i%2==0 else 'green' for i in range(20)]
    ...     )

    and make an `Evaluate` object to test its accuracy.

    >>> eval = Evaluate(model)
    >>> eval.accuracy(test_features=test_features,
    ...     test_responses=test_responses)
    0.5

    :param model:
        the model to evaluate, described above
    :param array-like data_features:
        the features of the data to evaluate the model on,
        generally the training set features
    :param array-like data_responses:
        the responses of the data to evaluate the model on

    '''

    def __init__(self, model=None, data_features=None, data_responses=None):
        self.model = model
        self.data_features = data_features
        self.data_responses = data_responses

    def cv_split(self, k=10):
        '''
        Make k folds of the data using stratified cross validation.

        :param int k:
            The number of folds to divide the data into
        '''

        self.cv = cross_validation.StratifiedKFold(
            self.data_responses,
            n_folds=k
            )

    def compute_scores(self, test_features=None, test_responses=None):
        '''
        Compute the metric scores for the model on the
        cross-validation folds of the training data
        or on the test data, storing the results in a dataframe.

        Add a new class attribute

        * *scores*,
            the results of the computations as a
            `pandas.core.frame.DataFrame` object

        :param array-like test_features:
            The features of the test data. If none, evaluate the model
            on the cv folds.
        :param array-like test_responses:
            The responses of the test data. If none, evaluate the model
            on the cv folds.
        '''

        self.accuracies = []
        for i, (train, test) in enumerate(self.cv):
            self.model.fit(
                self.data_features[train],
                self.data_responses[train]
                )
            predictions = self.model.predict(self.data_features[test])
            self.accuracies.append(
                self.accuracy(
                    test_responses=self.data_responses[test],
                    predictions=predictions
                    )
                )

    def accuracy(self,
                 test_responses,
                 test_features=None,
                 predictions=None,
                 train=None):
        '''
        Find the accuracy of the model on a given test set.

        :param array-like test_responses:
            the responses of the test set
        :param array-like test_features:
            the features of the test set to predict on.
            If None, use the pre-made `predictions`
        :param array-like predictions:
            the predictions to test.
            If None, use `test_features` to predict on
        :param array-like train:
            the data set to train the model on (optional)
        '''
        # check if the model and it's predict function exist
        if not self.model:
            raise EvalError('Accuracy: no model to evaluate!')
        if not self.model.predict:
            raise EvalError('Accuracy: model has no predict function!')

        # train the model if a training set is given and fit function exists
        if train:
            if not self.model.fit:
                raise EvalError('Accuracy: training set' +
                                ' given but model has no fit function!')
            else:
                self.model.fit(train)

        if not predictions:
            # make the prediction
            predictions = self.model.predict(test_features)

        # make sure the lengths of predictions and test_responses match up
        if not len(predictions) == len(test_responses):
            raise EvalError('The numbers of predictions' +
                            ' and test responses do not match up!')

        # calculate and return the accuracy
        return np.mean(test_responses == predictions)
