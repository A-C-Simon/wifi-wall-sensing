import unittest

import numpy as np

from sensing import features as F


class SweepTimingTests(unittest.TestCase):
    def test_hold_starts_at_completed_sweep_and_expires(self):
        t = np.array([0, 11.9, 12, 20, 24, 35, 36])
        actual = F.hold_sweeps(t, np.array([12, 24]), np.array([4, 8]), 12)
        np.testing.assert_array_equal(actual, [0, 0, 4, 4, 8, 8, 0])

    def test_no_sweeps(self):
        np.testing.assert_array_equal(
            F.hold_sweeps(np.arange(3), np.array([]), np.array([]), 12), [0, 0, 0])
