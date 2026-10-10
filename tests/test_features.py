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


class FineWindowTests(unittest.TestCase):
    def test_future_motion_does_not_change_past_features(self):
        t = np.arange(1.0, 11.0)
        rssi = np.r_[np.full(5, -60), np.full(5, -40)]
        full = F.fine_features(t, rssi, t * 0, t * 0, 4, 1)
        prefix = F.fine_features(t[:5], rssi[:5], t[:5] * 0, t[:5] * 0, 4, 1)
        np.testing.assert_array_equal(full[:5], prefix)
        np.testing.assert_allclose(full[5, 0], -55)

    def test_windows_use_time_and_correct_startup_transition_rate(self):
        t = np.array([1., 2., 10.])
        actual = F.fine_features(t, np.array([-60., -50., -40.]),
                                 np.zeros(3), np.ones(3), 3, 1)
        np.testing.assert_allclose(actual[:, 0], [-60, -55, -40])
        np.testing.assert_allclose(actual[:, 2], [1, 1, 1 / 3])

    def test_detrend_accounts_for_time_gap(self):
        actual = F.detrend(np.array([0., 2., 2.]), np.array([0., 1., 3.]), 1)
        np.testing.assert_allclose(actual, [0, 1, .25])

    def test_bad_window_rejected(self):
        with self.assertRaises(ValueError):
            F.fine_features(np.arange(3), np.zeros(3), np.zeros(3), np.zeros(3), 0, 1)
