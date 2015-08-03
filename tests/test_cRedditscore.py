#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
test_cRedditscore
----------------------------------

Tests for `cRedditscore` module.
'''

import unittest
import pandas as pd
import numpy as np

from cRedditscore import cRedditscore


class TestCredditscore(unittest.TestCase):

    def setUp(self):
        # read in test csv
        self.test_df = pd.read_csv('tests/test_data_1000.csv', index_col='id')

    def test_something(self):
        # make a TermFreqModel object
        tfm = cRedditscore.TermFreqModel(self.test_df)

        assert len(tfm.comments_df) == 1000

        # testing get_data
        assert len(tfm.get_data()) == 990

        # test get_quality
        assert cRedditscore.get_quality(
            score=15,
            low_thresh=0,
            high_thresh=15
            ) == 'neutral'

        assert np.sum(tfm.get_data().qual == 'good') == 71

        # test train_test
        tfm.train_test(test_size=0.1)
        assert len(tfm.X_train) == 177

        # test make_model
        tfm.make_model()
        one_prediction = tfm.model.predict(['Hello'])
        assert one_prediction in ['good', 'bad']

    def tearDown(self):
        pass

if __name__ == '__main__':
    unittest.main()
