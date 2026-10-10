import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from cli.validate import cmd_repeat_test, cmd_validate_map
from sensing.radiomap import Fingerprint, RadioMap


class MapValidationTests(unittest.TestCase):
    def validate(self, values, noise=0):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "map.json"
            rm = RadioMap([Fingerprint(i, 0, str(i), 0, v, 3)
                           for i, v in enumerate(values)], list("abcde"), {})
            rm.save(path)
            args = SimpleNamespace(radiomap=str(path), raw_dir=tmp,
                                   default_noise=noise, sep_threshold=2, repeat_margin=1)
            with contextlib.redirect_stdout(io.StringIO()):
                status = cmd_validate_map(args)
            return status, RadioMap.load(path).meta

    def test_invalid_maps_cannot_pass(self):
        a = dict.fromkeys("abcde", -60)
        for values in ([], [a], [{}, {}], [a, a], [a, {"a": -40}]):
            status, meta = self.validate(values)
            self.assertEqual(status, 2)
            self.assertEqual(meta["verdict"], "unusable")

    def test_separated_cells_pass(self):
        status, meta = self.validate([dict.fromkeys("abcde", -60),
                                      dict.fromkeys("abcde", -40)], noise=10)
        self.assertEqual(status, 0)
        self.assertEqual(meta["verdict"], "usable")


class RepeatabilityTests(unittest.TestCase):
    def compare(self, a, b):
        with tempfile.TemporaryDirectory() as tmp:
            pa, pb = Path(tmp) / "a.json", Path(tmp) / "b.json"
            a.save(pa)
            b.save(pb)
            with contextlib.redirect_stdout(io.StringIO()):
                result = cmd_repeat_test(SimpleNamespace(a=pa, b=pb, tol=.05))
            return result, RadioMap.load(pb)

    def test_unmatched_distant_cells_cannot_hide_session_drift(self):
        keys = list("abcde")
        def fp(x, r):
            return Fingerprint(x, 0, str(x), 0, dict.fromkeys(keys, r), 3)
        a = RadioMap([fp(0, -60), fp(1, -59), fp(100, -10)], keys, {})
        b = RadioMap([fp(0, -50), fp(1, -49)], keys, {"verdict": "usable"})
        status, saved = self.compare(a, b)
        self.assertEqual(status, 2)
        self.assertEqual(saved.meta["repeatability"]["mean_same_position_db"], 50)
        self.assertEqual(saved.meta["repeatability"]["mean_different_position_db"], 5)
        self.assertNotIn("verdict", saved.meta)

    def test_missing_readings_do_not_inflate_spatial_separation(self):
        keys = list("abcdefghij")
        common = dict.fromkeys("abcde", -60)
        a = RadioMap([Fingerprint(i, 0, str(i), 0,
                                 dict(common, **({k: -10 for k in "fghij"} if i else {})), 3)
                      for i in range(2)], keys, {})
        b = RadioMap([Fingerprint(i, 0, str(i), 0, dict.fromkeys("abcde", -50), 3)
                      for i in range(2)], keys, {})
        status, saved = self.compare(a, b)
        self.assertEqual(status, 2)
        rep = saved.meta["repeatability"]
        self.assertEqual(rep["anchor_keys"], list("abcde"))
        self.assertEqual(rep["mean_different_position_db"], 0)
