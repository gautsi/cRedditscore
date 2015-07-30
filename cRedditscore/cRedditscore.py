# -*- coding: utf-8 -*-
"""
A module of tools to train, test, validate and package
predictive models for the quality of comments on reddit.
"""


class TermFreqModel(object):
    """
    A Naive Bayes model predicting the quality of
    comments on Reddit. The data is input as a pandas dataframe
    with columns

    * *comment_id*: a unique identifier for each comment
    * *score* (int): the score of the comment
    * *content*: the comment itself

    :param pandas.core.frame.DataFrame comments_df:
        The dataframe containing the comment data
    :param int low_thresh:
        The lower bound for the score of a neutral comment.
        Anything lower is considered a bad comment.
    :param int high_thresh:
        The upper bound for the score of a neutral comment.
        Anything higher is considered a good comment.
    """

    def __init__(self, comments_df, low_thresh=0, high_thresh=15):
        self.comments_df = comments_df
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh

        # Pick only the most recent observation of each comment
        self.comments_most_recent = self.most_recent_obs(self.comments_df)
        self.add_qual_feature(self.comments_most_recent)

    def add_qual_feature(self, df):
        """
        Add the comment quality feature to the data.
        This will be our outcome variable.
        """

        df['qual'] = df.score.apply(lambda x: self.get_quality(x))

    def get_quality(self, score):
        """
        Get the quality (good, bad, neutral) of a score
        based on the score thresholds.
        """

        if score > self.high_thresh:
            qual = 'good'
        elif score < self.low_thresh:
            qual = 'bad'
        else:
            qual = 'neutral'

        return qual

    def most_recent_obs(self, df):
        """
        Select the most recent observation of each comment in
        a data set.

        :param pandas.core.frame.DataFrame df:
            The data set of comments

        :returns:
            The data set containing only the most
            recent observation of each comment in `df`
        :rtype: pandas.core.frame.DataFrame
        """

        return df.groupby(df.comment_id).last()
