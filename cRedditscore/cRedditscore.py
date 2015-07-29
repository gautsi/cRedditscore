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

    For example, suppose we

    :param pandas.core.frame.DataFrame comments_df:
        The dataframe containing the comment data

    """

    def __init__(self, comments_df):
        self.comments_df = comments_df

        # Pick only the most recent observation of each comment
        self.comments_most_recent = self.most_recent_obs(self.comments_df)

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
