import unittest

import numpy as np

from sensing import detect, io


def session(values):
    return io.Session(fine=[io.Fine(i * .05, 10, float(r), r, r, 0, 0, -90, 100)
                            for i, r in enumerate(values)])


class CalibrationTests(unittest.TestCase):
    def test_quiet_baseline_is_finite_and_detects_shift(self):
        baseline = detect.baseline_threshold(session(np.full(1000, -60)))
        self.assertGreater(baseline["fine_scale"], 0)
        quiet = detect.detect(session(np.full(1000, -60)), calibration=baseline)
        active = detect.detect(session(np.r_[np.full(500, -60), np.full(500, -50)]),
                               calibration=baseline)
        self.assertEqual(quiet.occupancy_fraction, 0)
        self.assertGreater(active.occupancy_fraction, .1)

    def test_calibrated_history_does_not_change_with_capture_length(self):
        baseline = detect.baseline_threshold(session(np.full(1000, -60)))
        values = np.r_[np.full(500, -60), np.full(500, -50)]
        full = detect.motion_score(session(values), calibration=baseline)[1]
        part = detect.motion_score(session(values[:750]), calibration=baseline)[1]
        np.testing.assert_allclose(full[:750], part)

    def test_old_or_mismatched_baseline_rejected(self):
        with self.assertRaises(ValueError):
            detect.detect(session([-60, -60]), calibration={"threshold": 3})
        baseline = detect.baseline_threshold(session([-60, -60]))
        with self.assertRaises(ValueError):
            detect.detect(session([-60, -60]), win_s=20, calibration=baseline)

    def test_malformed_baseline_values_raise_validation_errors(self):
        baseline = detect.baseline_threshold(session([-60, -60]))
        invalid = [None, [], dict(baseline, fine_center=None),
                   dict(baseline, threshold="3"), dict(baseline, fine_scale=True),
                   dict(baseline, sweep_keys=["a", "a"]),
                   dict(baseline, sweep_keys=[[]]), dict(baseline, min_sweep_l1=float("nan"))]
        for calibration in invalid:
            with self.assertRaises(ValueError):
                detect.validate_calibration(calibration, 10)

    def test_calibration_preserves_custom_sweep_gate(self):
        s = session(np.full(1000, -60))
        s.sweeps = [io.Sweep(t, 0, {"a": -60 + i % 2})
                    for i, t in enumerate(range(0, 50, 2))]
        calibration = detect.baseline_threshold(s, min_sweep_l1=.5)
        self.assertEqual(calibration["min_sweep_l1"], .5)
        target = session(np.full(1000, -60))
        target.sweeps = [io.Sweep(t, 0, {"a": -60 + 2 * (i % 2)})
                         for i, t in enumerate(range(0, 50, 2))]
        scored = detect.motion_score(target, calibration=calibration)
        self.assertGreater(scored[3].max(), 0)

    def test_live_and_recorded_scores_match(self):
        from wifisense_live import scores
        s = session(np.sin(np.arange(1000) / 30) - 60)
        s.sweeps = [io.Sweep(i * 12, 0, {"a": -60 + i}) for i in range(4)]
        snap = {"t": np.array([f.t for f in s.fine]),
                "rssi": np.array([f.rssi for f in s.fine]),
                "sd": np.array([f.sd for f in s.fine]),
                "chg": np.array([f.chg for f in s.fine]),
                "sweeps": [(sw.t, sw.aps) for sw in s.sweeps]}
        for calibration in (None, detect.baseline_threshold(s)):
            actual = scores(snap, 10, calibration)[1]
            expected = detect.motion_score(s, calibration=calibration)[1]
            np.testing.assert_allclose(actual, expected)

    def test_later_scan_cadence_cannot_rewrite_calibrated_history(self):
        baseline = session(np.full(2000, -60))
        baseline.sweeps = [io.Sweep(t, 0, {"a": -60 + i})
                           for i, t in enumerate((0, 5, 10))]
        calibration = detect.baseline_threshold(baseline)
        target = session(np.full(2000, -60))
        target.sweeps = [io.Sweep(t, 0, {"a": -60 + 10 * i})
                         for i, t in enumerate((0, 5, 10))]
        prefix = detect.motion_score(target, calibration=calibration)
        target.sweeps += [io.Sweep(t, 0, {"a": -30}) for t in (100, 200, 300)]
        full = detect.motion_score(target, calibration=calibration)
        for actual, expected in zip(full[:5], prefix[:5]):
            np.testing.assert_allclose(actual, expected)
