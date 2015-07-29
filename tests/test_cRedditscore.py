#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_cRedditscore
----------------------------------

Tests for `cRedditscore` module.
"""

import unittest
import pandas as pd

from cRedditscore import cRedditscore


class TestCredditscore(unittest.TestCase):

    def setUp(self):
        self.test_df = pd.read_csv('test_data_1000.csv', index_col='id')

    def test_something(self):
        tfm = cRedditscore.TermFreqModel(self.test_df)
        assert len(tfm.comments_df) == 1000
        assert len(tfm.comments_most_recent) == 990

    def tearDown(self):
        pass

if __name__ == '__main__':
    unittest.main()
