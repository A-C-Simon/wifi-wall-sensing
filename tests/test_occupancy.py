import unittest

import numpy as np

from validate.occupancy_test import auc, paired_summary


class OccupancyStatsTests(unittest.TestCase):
    def test_auc_matches_pairwise_definition_with_ties(self):
        pos, neg = np.array([1, 2, 2, 4]), np.array([0, 2, 3])
        differences = pos[:, None] - neg
        expected = np.mean((differences > 0) + .5 * (differences == 0))
        self.assertAlmostEqual(auc(pos, neg), expected)

    def test_more_correlated_buckets_cannot_increase_replication(self):
        empty = [np.array([0, 1]), np.array([0, 1]), np.array([1, 2])]
        occupied = [np.array([2, 3]), np.array([3, 4]), np.array([-1, 0])]
        first = paired_summary(empty, occupied)
        duplicated = paired_summary([np.repeat(v, 100) for v in empty],
                                    [np.repeat(v, 100) for v in occupied])
        self.assertEqual(first, duplicated)
        self.assertGreaterEqual(first["paired_p"], .25)

    def test_incomplete_capture_has_no_inferential_claim(self):
        self.assertEqual(paired_summary([np.array([0]), np.array([])],
                                        [np.array([1]), np.array([1])]), {})
