# -*- coding: utf-8 -*-
"""A module of tools to collect comment data from Reddit"""

import praw as pr


class collection(object):
    """
    A class to handle the collecting and storing of comment data from Reddit.

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
                             'funny'
                             'news',
                             'science']):

        self.user_agent = user_agent
        self.subreddits = subreddits
        if user_agent:
            self.red = pr.Reddit(user_agent=self.user_agent)
        else:
            self.red = None
