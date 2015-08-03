#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_collection
----------------------------------

Tests for `collection` module.
"""

import unittest

from cRedditscore import collection


class TestCollection(unittest.TestCase):

    def setUp(self):
        pass

    def test_something(self):

        col = collection.Collect()
        assert col.subreddits[0] == 'dataisbeautiful'

        subm = col.get_random_subm()
        assert not subm

        rows = col.fetch_subm_from_table(subm)
        assert rows == []

        a = col.subm_to_db(subm)
        assert not a

        b = col.comment_to_db()
        assert not b

        c = col.rand_subm_to_db()
        assert not c

    def tearDown(self):
        pass

if __name__ == '__main__':
    unittest.main()
