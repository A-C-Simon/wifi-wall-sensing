import unittest

import numpy as np

from sensing import detect, io
from sensing.stream import FineEnergy


class StreamingTests(unittest.TestCase):
    def test_irregular_buckets_match_recorded_filter(self):
        rng = np.random.default_rng(1)
        t = np.cumsum(rng.uniform(.04, .08, 3000))
        rssi = -60 + 4 * np.sin(t / 11)
        expected, _ = detect._fine_energy(t, rssi, t * 0, t * 0, 7.3)
        stream = FineEnergy(7.3)
        actual = [stream.update(float(ti), float(ri)) for ti, ri in zip(t, rssi)]
        np.testing.assert_allclose(actual, expected, atol=1e-10)
        self.assertLess(len(stream.window), 200)

    def test_plot_buffer_rollover_does_not_reset_filters(self):
        from wifisense_live import State, scores
        values = -60 + 15 * np.sin(np.arange(45000) / 20000)
        recording = io.Session(fine=[io.Fine(i * .05, 1, r, r, r, 0, 0, -90, 100)
                                     for i, r in enumerate(values)])
        calibration = detect.baseline_threshold(recording, win_s=600)
        expected = detect.motion_score(recording, win_s=600, calibration=calibration)[1]
        state = State()
        state.win_s = 600
        for f in recording.fine:
            state.append_fine({"t": f.t, "rssi": f.rssi, "min": f.mn,
                               "max": f.mx, "sd": f.sd, "chg": f.chg})
        snap = state.snapshot()
        self.assertEqual(len(snap["t"]), 40000)
        self.assertEqual(snap["t"][0], recording.fine[5000].t)
        actual = scores(snap, 600, calibration)[1]
        np.testing.assert_allclose(actual, expected[-40000:], atol=1e-8)

    def test_bad_sample_does_not_change_filter_state(self):
        stream = FineEnergy(10)
        stream.update(1, -60)
        for t, rssi in ((1, -50), (0, -50), (2, float("nan"))):
            with self.assertRaises(ValueError):
                stream.update(t, rssi)
        self.assertEqual(stream.update(2, -60), 0)
