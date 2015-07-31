# -*- coding: utf-8 -*-
'''
A module of tools to train, test, validate and package
predictive models for the quality of comments on reddit.
'''

import pandas as pd
from sklearn import cross_validation
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB


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
        The dataframe containing the comment data
    :param int low_thresh:
        The lower bound for the score of a neutral comment.
        Anything lower is considered a bad comment.
    :param int high_thresh:
        The upper bound for the score of a neutral comment.
        Anything higher is considered a good comment.
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
        '''Split the data into train and test sets.'''

        train_test = cross_validation.train_test_split(
            self.good_bad_df.content,
            self.good_bad_df.qual,
            test_size=test_size, random_state=0
            )

        self.X_train, self.X_test, self.y_train, self.y_test = train_test

    def make_model(self, test_size=0.2):
        '''
        Split the data into train and test sets and
        train the Naive Bayes model.
        '''

        self.train_test(test_size)

        # Make the count vectorizer
        self.cvec = CountVectorizer(
            ngram_range=(1, 4),
            stop_words='english',
            max_features=1000
            )

        # Make the pipeline
        self.model = Pipeline([
            ('vect', self.cvec),
            ('tfidf', TfidfTransformer(use_idf=False)),
            ('gnb', MultinomialNB()),
            ])

        # Fit the model
        self.model.fit(self.X_train, self.y_train)

    def get_good_bad(self, df):
        '''
        Get the good and bad comments in a data set.

        :param pandas.core.frame.DataFrame df:
            The comments data set.

        :return:
            The dataframes containing
            only the good and bad comments from `df`.
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
        Add the comment quality feature to the data.
        This will be our outcome variable.
        '''

        df['qual'] = df.score.apply(lambda x: self.get_quality(x))

    def get_quality(self, score):
        '''
        Get the quality (good, bad, neutral) of a score
        based on the score thresholds.

        For example, we make a small comment data set and get
        the quality of the first comment:

        >>> import pandas as pd
        >>> test_df = pd.DataFrame([
        ...     [1, 1, "That's cool", 1],
        ...     [2, -3, "boo you", 2],
        ...     [3, 4, "I love you", 2],
        ...     [3, 16, "I love you", 4],
        ...     ], columns=['comment_id', 'score', 'content', 'timestamp'])
        >>> tfm = TermFreqModel(comments_df = test_df)
        >>> tfm.get_quality(test_df.score[0])
        'neutral'

        :param int score: The score of the comment.
        :returns: the quality of the comment (good, bad or neutral)
        :rtype: string
        '''

        if score > self.high_thresh:
            qual = 'good'
        elif score < self.low_thresh:
            qual = 'bad'
        else:
            qual = 'neutral'

        return qual

    def most_recent_obs(self, df):
        '''
        Select the most recent observation of each comment in
        a data set.

        :param pandas.core.frame.DataFrame df:
            The data set of comments

        :returns:
            The data set containing only the most
            recent observation of each comment in `df`
        :rtype: pandas.core.frame.DataFrame
        '''

        return df.groupby(df.comment_id).last()

    def get_data(self):
        '''
        Get the full data set of the model.

        :returns:
            The full data set underlying the model.
        :rtype: pandas.core.frame.DataFrame
        '''

        return self.comments_data_set
