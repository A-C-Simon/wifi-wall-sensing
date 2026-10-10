import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from cli.survey import cmd_coarsen
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
