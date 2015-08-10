# -*- coding: utf-8 -*-
'''
A module of tools to evaluate predictive models.
'''
import numpy as np
import pandas as pd
from sklearn import cross_validation
from sklearn import metrics as skmetrics


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
    :param pos_label:
        the class to be considered positive for auc, precision and recall;
        if None, the first class is picked
    '''

    def __init__(self,
                 model=None,
                 data_features=None,
                 data_responses=None,
                 pos_label=None):
        self.model = model
        self.data_features = data_features
        self.data_responses = data_responses
        self.classes = list(set(self.data_responses))
        if len(self.classes) > 2:
            raise EvalError('More than two classes!')
        if pos_label is None:
            self.pos_label = self.data_responses.iloc[0]
        else:
            self.pos_label = pos_label

    def cv_split(self, k=10):
        '''
        Make k folds of the data using stratified cross validation.

        :param int k:
            The number of folds to divide the data into
        '''

        self.k = k
        self.cv = cross_validation.StratifiedKFold(
            self.data_responses,
            n_folds=k
            )

    def compute_scores(
            self,
            test_features=None,
            test_responses=None,
            metrics=['accuracy', 'auc', 'precision', 'recall', 'f1']
            ):
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

        # if only one metric is passed as a string, turn it into a list
        if not type(metrics) is list:
            metrics = [metrics]

        # build the scores dataframe
        self.scores = self.build_scores_df()

        for i, (train, test) in enumerate(self.cv):

            # fit the model
            self.model.fit(
                self.data_features[train],
                self.data_responses[train]
                )

            # make the predictions if there's a metric besides auc
            if not metrics == ['auc']:
                predictions = self.model.predict(self.data_features[test])

            # make probability predictions if auc is a metric
            if 'auc' in metrics:
                prob_preds = self.model.predict_proba(self.data_features[test])

            if 'accuracy' in metrics:
                acc = self.accuracy(
                    test_responses=self.data_responses[test],
                    predictions=predictions
                    )
                self.scores.loc['accuracy']['fold'][i] = acc

            if 'auc' in metrics:

                # add fpr, tpr and thresh to roc list
                fpr, tpr, thresh = skmetrics.roc_curve(
                    y_true=self.data_responses[test],
                    y_score=prob_preds[:, 1],
                    pos_label=self.pos_label
                    )
                try:
                    self.roc.append([fpr, tpr, thresh])
                except:
                    self.roc = [[fpr, tpr, thresh]]

                # compute the auc
                self.scores.loc['auc']['fold'][i] = skmetrics.auc(fpr, tpr)

            if 'precision' in metrics:
                sum_prec = 0.0
                for cl in self.classes:
                    prec = skmetrics.precision_score(
                        y_true=self.data_responses[test],
                        y_pred=predictions,
                        pos_label=cl
                        )
                    sum_prec += prec
                    ind = ('precision', 'class {}'.format(cl))
                    self.scores.xs(ind)['fold'][i] = prec
                self.scores.xs(('precision', 'avg'))['fold'][i] = sum_prec/2

            if 'recall' in metrics:
                sum_rec = 0.0
                for cl in self.classes:
                    rec = skmetrics.recall_score(
                        y_true=self.data_responses[test],
                        y_pred=predictions,
                        pos_label=cl
                        )
                    sum_rec += rec
                    ind = ('recall', 'class {}'.format(cl))
                    self.scores.xs(ind)['fold'][i] = rec
                self.scores.xs(('recall', 'avg'))['fold'][i] = sum_rec/2

            if 'f1' in metrics:
                sum_f1 = 0.0
                for cl in self.classes:
                    f1 = skmetrics.f1_score(
                        y_true=self.data_responses[test],
                        y_pred=predictions,
                        pos_label=cl
                        )
                    sum_f1 += f1
                    ind = ('f1', 'class {}'.format(cl))
                    self.scores.xs(ind)['fold'][i] = f1
                self.scores.xs(('f1', 'avg'))['fold'][i] = sum_f1/2

            # add min, max, mean, std
            for i in self.scores.index:
                self.scores.xs(i)['min'] = np.min(self.scores.xs(i)['fold'])
                self.scores.xs(i)['max'] = np.max(self.scores.xs(i)['fold'])
                self.scores.xs(i)['mean'] = np.mean(self.scores.xs(i)['fold'])
                self.scores.xs(i)['std'] = np.std(self.scores.xs(i)['fold'])

    def build_scores_df(self):
        '''Build the scores dataframe.'''

        metrics_level_1 = pd.MultiIndex.from_tuples([('accuracy',), ('auc',)])
        metrics_level_2 = pd.MultiIndex.from_product([
            ['precision', 'recall', 'f1'],
            ['class {}'.format(cl) for cl in self.classes]+['avg']
            ])
        fold_columns = pd.MultiIndex.from_tuples(
            [('fold', i) for i in xrange(self.k)]
            )
        agg_columns = pd.MultiIndex.from_tuples(
            [(agg, '') for agg in ['min', 'max', 'mean', 'std']]
            )
        scores = pd.DataFrame(
            columns=fold_columns | agg_columns,
            index=metrics_level_1 | metrics_level_2
            )
        return scores

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

        if predictions is None:
            # make the prediction
            predictions = self.model.predict(test_features)

        # make sure the lengths of predictions and test_responses match up
        if not len(predictions) == len(test_responses):
            raise EvalError('The numbers of predictions' +
                            ' and test responses do not match up!')

        # calculate and return the accuracy
        return np.mean(test_responses == predictions)

    def compute_curves(self):
        '''Plot ROC and precision recall curves.'''
        pass
