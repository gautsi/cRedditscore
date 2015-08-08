# -*- coding: utf-8 -*-
'''
A module of tools to train and package
predictive models for the quality of comments on reddit.
'''

import pandas as pd
from sklearn import cross_validation
from sklearn import pipeline
from sklearn.feature_extraction import text
from sklearn import naive_bayes
import pickle as pk


def get_quality(score, low_thresh=0, high_thresh=15):
    '''
    Get the quality (good, bad, neutral) of a score
    based on the score thresholds.

    For example,

    >>> get_quality(score=15)
    'neutral'
    >>> get_quality(score=2, low_thresh=5, high_thresh=20)
    'bad'
    >>> get_quality(score=2, low_thresh=-10, high_thresh=1)
    'good'

    :param int score: the score of the comment
    :param int low_thresh: the low threshold of a neutral comment
    :param int high_thresh: the high threshold of a neutral comment
    :returns: the quality of the comment (good, bad or neutral)
    :rtype: string
    '''

    if score > high_thresh:
        qual = 'good'
    elif score < low_thresh:
        qual = 'bad'
    else:
        qual = 'neutral'

    return qual


class TermFreqModel(object):
    '''
    A Naive Bayes model predicting the quality of
    comments on Reddit. The data is input as a pandas dataframe
    with columns

    * *comment_id*: a unique identifier for each comment
    * *score* (int): the score of the comment
    * *content*: the comment itself

    For example, we make a small comments data set

    >>> import pandas as pd
    >>> test_df = pd.DataFrame([
    ...     [1, 1, "That's cool", 1],
    ...     [2, -3, 'boo you', 2],
    ...     [3, 4, 'I love you', 2],
    ...     [3, 16, 'I love you', 4],
    ...     ], columns=['comment_id', 'score', 'content', 'timestamp'])

    and build a `TermFreqModel` object from it.

    >>> tfm = TermFreqModel(comments_df = test_df)

    :param pandas.core.frame.DataFrame comments_df:
        the dataframe containing the comment data
    :param int low_thresh:
        the lower bound for the score of a neutral comment.
        Anything lower is considered a bad comment
    :param int high_thresh:
        the upper bound for the score of a neutral comment.
        Anything higher is considered a good comment

    '''

    def __init__(self, comments_df, low_thresh=0, high_thresh=15):
        self.comments_df = comments_df
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh

        self.setup_data()

    def setup_data(self):
        '''
        Set up the data for model training.
        In detail:

        * Remove all but the most recent observation for each comment
        * Add the quality feature to the data, which will be our outcome
          variable
        * Separate out the good and bad comments. This will be the data
          we train the model on.

        This function adds a new class attribute

        * *good_bad_df*:
            the dataframe containing only
            the most recent observations of the
            good and bad comments
        '''
        # Pick only the most recent observation of each comment
        self.comments_data_set = self.most_recent_obs(self.comments_df)

        # Add the quality feature to the data
        self.add_qual_feature(self.comments_data_set)

        # Get the good and bad comments
        self.good_df, self.bad_df = self.get_good_bad(self.comments_data_set)

        # Combine good and bad comments
        self.good_bad_df = pd.concat([self.good_df, self.bad_df])

    def train_test(self, test_size=0.2):
        '''
        Split the data into train and test sets.

        This function adds new class attributes

        * *X_train* and *X_test*,
            The features of the training and test parts of the data set
        * *y_train* and *y_test*,
            The outcomes of the training and test parts of the data set

        :param int test_size:
            the percentage of data points to hold out for testing
        '''

        train_test = cross_validation.train_test_split(
            self.good_bad_df.content,
            self.good_bad_df.qual,
            test_size=test_size, random_state=0
            )

        self.X_train, self.X_test, self.y_train, self.y_test = train_test

    def make_model(self,
                   test_size=0.2,
                   ngram_range=(1, 4),
                   max_features=1000
                   ):
        '''
        Make a new class attribute

        * *model*:
            the Naive Bayes model as an *sklearn.pipeline.Pipeline* object

        For example, we make a small comments data set,

        >>> import pandas as pd
        >>> test_df = pd.DataFrame([
        ...     [1, 1, "That's cool", 1],
        ...     [2, -3, 'boo you', 2],
        ...     [3, 4, 'I love you', 2],
        ...     [3, 16, 'I love you', 4],
        ...     ], columns=['comment_id', 'score', 'content', 'timestamp'])

        build a `TermFreqModel` object from it,

        >>> tfm = TermFreqModel(test_df)

        and train a model on it predictive of the quality of a comment.

        >>> tfm.train_test(test_size=0.1)
        >>> tfm.make_model(ngram_range=(1, 3), max_features=10)
        >>> tfm.fit()
        >>> prediction = tfm.model.predict(['Thanks for a great post!'])
        >>> prediction in ['good', 'bad']
        True

        :param int test_size:
            the percentage of data points to hold out for testing
        :param tuple ngram_range:
            the range of n for ngrams to include as features
        :param int max_features:
            the maximum number of features to include
        '''

        # Make the count vectorizer
        self.cvec = text.CountVectorizer(
            ngram_range=ngram_range,
            stop_words='english',
            max_features=max_features
            )

        # Make the pipeline
        self.model = pipeline.Pipeline([
            ('vect', self.cvec),
            ('tfidf', text.TfidfTransformer(use_idf=False)),
            ('gnb', naive_bayes.MultinomialNB()),
            ])

    def fit(self):
        '''Fit the model.'''
        self.model.fit(self.X_train, self.y_train)

    def dump_model(self, pickle_name='text_mnb_model'):
        '''
        Dump the model object to file with `pickle`.

        :param string pickle_name:
            the name of the file to dump the object to
        '''
        write_file = open(pickle_name, 'w')
        pk.dump(self.model, write_file)

    def get_good_bad(self, df):
        '''
        Get the good and bad comments in a data set.

        :param pandas.core.frame.DataFrame df:
            the comments data set

        :return:
            the dataframes containing
            only the good and bad comments from `df`
        :rtype: pandas.core.frame.DataFrame, pandas.core.frame.DataFrame
        '''

        good_df = self.comments_data_set[
            self.comments_data_set.qual == 'good'
            ]

        bad_df = self.comments_data_set[
            self.comments_data_set.qual == 'bad'
            ]

        return good_df, bad_df

    def add_qual_feature(self, df):
        '''
        Add the comment quality feature to the data
        as a new column named `qual`.
        This will be our outcome variable.

        :param pandas.core.frame.DataFrame df:
            the data frame to add the `qual` column to
        '''

        qual_func = lambda x: get_quality(
            score=x,
            low_thresh=self.low_thresh,
            high_thresh=self.high_thresh)

        df['qual'] = df.score.apply(qual_func)

    def most_recent_obs(self, df):
        '''
        Select the most recent observation of each comment in
        a data set.

        :param pandas.core.frame.DataFrame df:
            The data set of comments

        :returns:
            the data set containing only the most
            recent observation of each comment in `df`
        :rtype: pandas.core.frame.DataFrame
        '''

        return df.groupby(df.comment_id).last()

    def get_data(self):
        '''
        Get the full data set of the model.

        :returns:
            the full data set underlying the model
        :rtype: pandas.core.frame.DataFrame
        '''

        return self.comments_data_set
