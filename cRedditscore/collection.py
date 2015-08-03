# -*- coding: utf-8 -*-
"""A module of tools to collect comment data from Reddit."""

import praw as pr
import random as rn
from sqlalchemy.sql import select
import time as tm
from sqlalchemy.exc import StatementError


class Collect(object):
    """
    A class to handle the collecting and storing of comment data from Reddit.

    In practice, `collection` requires the name of a Reddit web app to
    connect to the Reddit api. By default, no web app is passed to the class,
    in which case the class is useless.

    The database for storing comments has a table `subm_table` for storing
    submission data with columns

    * *submission_id* (varchar(45)),
    * *subreddit* (varchar(45)), and
    * *timestamp* (int(11))

    and a table `comm_table` for storing comment data with columns

    * *submission_id* (varchar(45)),
    * *subm_title* (varchar(45)),
    * *subm_content* (blob),
    * *subm_created* (int(11)),
    * *subm_created_local* (int(11)),
    * *subm_score* (int(11)),
    * *subm_author* (varchar(45)),
    * *subm_num_comments* (int(11)),
    * *comment_id* (varchar(45)),
    * *user_id* (varchar(45)),
    * *prev_comment_id* (varchar(45)),
    * *created* (int(11)),
    * *created_local* (int(11)),
    * *timestamp* (int(11)),
    * *content* (blob),
    * *subreddit* (varchar(45)),
    * *score* (int(11)),
    * *ups* (int(11)),
    * *downs* (int(11)), and
    * *controversiality* (int(11)).

    :param string user_agent:
        the name of the web app to connect to the api through
    :param list subreddits:
        the list of subreddits to collect from
    :param sqlalchemy.sql.schema.Table subm_table:
        the submissions sql table
    :param sqlalchemy.sql.schema.Table comm_table:
        the comments sql table
    :param sqlalchemy.engine.base.Connection conn:
        the connection to the sql database

    For example,

    >>> col = Collect()
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
                 comm_table=None,
                 conn=None,
                 debug=False):

        self.user_agent = user_agent
        self.subreddits = subreddits
        self.subm_table = subm_table
        self.comm_table = comm_table
        self.conn = conn
        self.debug = debug

        if user_agent:
            self.red = pr.Reddit(user_agent=self.user_agent)
        else:
            self.red = None

    def get_random_subm(self):
        """
        Get a random submission from Reddit.

        :returns:
            a random submission from a random subreddit in `self.subreddits`
            if `self.red` exists, else None
        :rtype:
            praw.objects.Submission
        """

        if not self.red:
            return None

        random_index = rn.randint(0, len(self.subreddits)-1)
        subr = self.red.get_subreddit(self.subreddits[random_index])
        return subr.get_random_submission()

    def fetch_subm_from_table(self, subm):
        """
        Fetch the rows corresponding to `subm` in `self.subm_table`.

        :returns: the rows in `self.subm_table` that come from `subm`
        :rtype: list
        """

        if not self.conn:
            return []

        query = select([self.subm_table])\
            .where(self.subm_table.c.submission_id == subm.name)
        return self.conn.execute(query).fetchall()

    def subm_to_db(self, subm):
        """Add `subm` to the database with all of its comments."""

        if not self.conn:
            return

        subm_table_values = {
            'submission_id': subm.name,
            'subreddit': subm.subreddit.title,
            'timestamp': int(tm.time())
            }

        self.conn.execute(self.subm_table.insert(), [subm_table_values])

        subm_values = {
            'submission_id': subm.name,
            'subm_title': subm.title,
            'subm_content': subm.selftext,
            'subm_created': int(subm.created_utc),
            'subm_created_local': int(subm.created),
            'subm_score': subm.score,
            'subm_author': str(subm.author),
            'subm_num_comments': subm.num_comments
            }

        subm.replace_more_comments(limit=None, threshold=0)

        comments = pr.helpers.flatten_tree(subm.comments)

        for comment in comments:
            self.comment_to_db(comment, subm_values)

    def comment_to_db(self, comment=None, subm_values=None):
        """Add `comment` to the database."""

        if not self.conn:
            return

        values = {
            'comment_id': comment.name,
            'user_id': str(comment.author),
            'prev_comment_id': comment.parent_id,
            'created': int(comment.created_utc),
            'created_local': int(comment.created),
            'timestamp': int(tm.time()),
            'content': comment.body,
            'subreddit': comment.subreddit.title,
            'score': comment.score,
            'ups': comment.ups,
            'downs': comment.downs,
            'controversiality': comment.controversiality
            }
        values.update(subm_values)

        # In case the comment cant be added because of encoding,
        # print a note and pass
        try:
            self.conn.execute(self.comm_table.insert(), [values])
        except (UnicodeEncodeError, StatementError):
            if self.debug:
                print("unicode error at comment level")
            pass

    def rand_subm_to_db(self):
        """Add a random submission to the database."""

        if not self.conn:
            return

        sub_already_there = True
        while sub_already_there:
            subm = self.get_random_subm()
            subm_rows = self.fetch_subm_from_table(subm)
            sub_already_there = len(subm_rows) > 0
            if sub_already_there:
                # get the time since the last collection if it exists,
                # else 0
                time_since = tm.time()\
                    - max([row['timestamp'] for row in subm_rows])\
                    if not subm_rows[0]['timestamp'] is None\
                    else 0

                if tm.time() - subm.created_utc < 120000\
                        and time_since > 600\
                        and subm.num_comments > 10:

                    sub_already_there = False

                    if self.debug:
                        debug_string = "sub already here add {0} in {1}"
                        print(debug_string.format(subm.name,
                                                  subm.subreddit.title
                                                  ))
                else:
                    if self.debug:
                        debug_string = "sub already here ignore {0} in {1}"
                        print(debug_string.format(subm.name,
                                                  subm.subreddit.title
                                                  ))

        self.subm_to_db(subm)

        if self.debug:
            debug_string = "added {0} {1}, {2}"
            print(debug_string.format(subm.subreddit.title,
                                      subm.name,
                                      subm.num_comments
                                      ))
