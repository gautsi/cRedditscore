# -*- coding: utf-8 -*-
"""A module of tools to collect comment data from Reddit"""

import praw as pr
import random


class collection(object):
    """
    A class to handle the collecting and storing of comment data from Reddit.

    In practice, `collection` requires the name of a Reddit web app to
    connect to the Reddit api. By default, no web app is passed to the class,
    in which case the class is useless.

    The database for storing comments has a table `subm_table` for storing
    submission data with columns

    * *submission_id*,
    * *subreddit*, and
    * *timestamp*

    and a table `comm_table` for storing comment data with columns

    * *submission_id*,
    * *subm_title*,
    * *subm_content*,
    * *subm_created*,
    * *subm_created_local*,
    * *subm_score*,
    * *subm_author*,
    * *subm_num_comments*,
    * *comment_id*,
    * *user_id*,
    * *prev_comment_id*,
    * *created*,
    * *created_local*,
    * *timestamp*,
    * *content*,
    * *subreddit*,
    * *score*,
    * *ups*,
    * *downs*, and
    * *controversiality*.

    :param string user_agent:
        the name of the web app to connect to the api through
    :param list subreddits:
        the list of subreddits to collect from
    :param sqlalchemy.sql.schema.Table subm_table:
        the submissions sql table
    :param sqlalchemy.sql.schema.Table comm_table:
                the comments sql table

    For example,

    >>> col = collection()
    >>> col.subreddits[0]
    'dataisbeautiful'
    """

    def __init__(self,
                 user_agent=None,
                 subreddits=['dataisbeautiful',
                             'programming',
                             'technology',
                             'python',
                             'cpp',
                             'funny',
                             'news',
                             'science'],
                 subm_table=None,
                 comm_table=None):

        self.user_agent = user_agent
        self.subreddits = subreddits
        self.subm_table = subm_table
        self.comm_table = comm_table

        if user_agent:
            self.red = pr.Reddit(user_agent=self.user_agent)
        else:
            self.red = None

    def get_random_subm(self):
        """
        Get a random submission from Reddit.

        :returns:
            a random submission from a random subreddit in self.subreddits
            if self.red exists, else None
        :rtype:
            praw.objects.Submission
        """

        if not self.red:
            return None

        random_index = random.randint(0, len(self.subreddits)-1)
        subr = self.red.get_subreddit(self.subreddits[random_index])
        return subr.get_random_submission()

    def is_subm_in_table(self, subm):
        """Check to see if subm is already in self.subm_table"""
        pass
