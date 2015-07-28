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
        col = collection.collection()
        assert col.subreddits[0] == 'dataisbeautiful'

        subm = col.get_random_subm()
        assert not subm

    def tearDown(self):
        pass

if __name__ == '__main__':
    unittest.main()
