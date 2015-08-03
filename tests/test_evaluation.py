#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_evaluation
----------------------------------

Tests for `evaluation` module.
"""

import unittest

from cRedditscore import evaluation


class TestEvaluation(unittest.TestCase):

    def setUp(self):
        pass

    def test_something(self):

        col = evaluation.Evaluate()
        assert not type(col) is None

    def tearDown(self):
        pass

if __name__ == '__main__':
    unittest.main()
