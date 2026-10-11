import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from cli.match import cmd_localize
from sensing.io import Session, Sweep
from sensing.radiomap import Fingerprint, RadioMap, error_metrics


class LocalizationScoringTests(unittest.TestCase):
    def test_unresolved_sweep_does_not_shift_ground_truth(self):
        keys = list("abcde")
        a, b = dict.fromkeys(keys, -60), dict.fromkeys(keys, -40)
        rm = RadioMap([Fingerprint(0, 0, "a", 0, a, 3),
                       Fingerprint(10, 0, "b", 0, b, 3)], keys, {"verdict": "usable"})
        session = Session(sweeps=[Sweep(0, 0, a), Sweep(12, 0, {}), Sweep(24, 0, b)])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mp, truth = root / "map.json", root / "truth.csv"
            rm.save(mp)
            truth.write_text("0,0\n5,0\n10,0\n")
            args = SimpleNamespace(radiomap=mp, capture="capture", truth=truth)
            output = io.StringIO()
            with patch("cli.match.io.load", return_value=session), \
                 patch("cli.match.viz.plot_trace"), \
                 contextlib.redirect_stdout(output):
                self.assertEqual(cmd_localize(args), 0)
            self.assertIn('"mean_err_m": 0.0', output.getvalue())
            self.assertIn("resolved 2/3 sweeps; 1 unresolved", output.getvalue())

    def test_error_metrics_rejects_silent_truncation(self):
        with self.assertRaises(ValueError):
            error_metrics(None, [(0, 0), (1, 0)], [(0, 0)])
        with self.assertRaises(ValueError):
            error_metrics(None, [(0, 0)], [(float("nan"), 0)])
