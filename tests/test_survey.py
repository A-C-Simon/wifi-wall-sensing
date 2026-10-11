import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from cli.survey import cmd_calibrate, cmd_coarsen, fingerprint_from_session
from sensing.io import Session, Sweep
from sensing.radiomap import Fingerprint, RadioMap


class CoarseningTests(unittest.TestCase):
    def coarsen(self, step):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, target = root / "in.json", root / "out.json"
            keys = list("abcde")
            RadioMap([Fingerprint(x, 0, str(x), 0, dict.fromkeys(keys, -60 + x), 3)
                      for x in (0, 2, 4)], keys, {"verdict": "usable"}).save(source)
            args = SimpleNamespace(radiomap=source, out=target, extent=step)
            with patch("cli.survey.OUT", root), patch("cli.survey.viz.plot_radio_map"), \
                 contextlib.redirect_stdout(io.StringIO()):
                status = cmd_coarsen(args)
            return status, RadioMap.load(target) if target.exists() else None

    def test_halfway_tie_uses_lower_center_and_invalidates_verdict(self):
        status, result = self.coarsen(4)
        self.assertEqual(status, 0)
        self.assertEqual([p.x for p in result.points], [0, 4])
        self.assertEqual([p.n_samples for p in result.points], [6, 3])
        self.assertNotIn("verdict", result.meta)

    def test_tiny_step_does_not_allocate_the_candidate_grid(self):
        status, result = self.coarsen(.000001)
        self.assertEqual(status, 0)
        self.assertEqual(len(result.points), 3)

    def test_invalid_step_is_rejected(self):
        for step in (0, -1, float("nan"), float("inf")):
            self.assertEqual(self.coarsen(step), (1, None))


class SurveyAcquisitionTests(unittest.TestCase):
    def test_each_cell_waits_for_operator_before_capture(self):
        events = []
        session = Session(sweeps=[Sweep(t, 0, dict.fromkeys("abcde", -60))
                                 for t in (0, 12, 24)])
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(cols=2, rows=1, origin_x=.001, origin_y=0,
                                   extent=.001, hold=1, sweeps=3, iface="en0",
                                   laptop_x=0, laptop_y=0, room_x=7, room_y=6)
            with patch("cli.survey.OUT", Path(tmp)), patch("cli.survey.build_core"), \
                 patch("builtins.input", side_effect=lambda _: events.append("ready")), \
                 patch("cli.survey.subprocess.run", side_effect=lambda *a, **k: events.append("capture")), \
                 patch("cli.survey.io.load", return_value=session), \
                 patch("cli.survey.viz.plot_radio_map"), \
                 patch("cli.survey.separability_report", return_value=""), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cmd_calibrate(args), 0)
            rm = RadioMap.load(Path(tmp) / "radiomap.json")
        self.assertEqual(events, ["ready", "capture", "ready", "capture"])
        self.assertEqual([p.label for p in rm.points], ["0.001,0", "0.002,0"])

    def test_requested_sweep_count_is_enforced(self):
        session = Session(sweeps=[Sweep(t, 0, dict.fromkeys("abcde", -60)) for t in (0, 12)])
        self.assertIsNone(fingerprint_from_session(session, 0, 0, min_samples=3))
