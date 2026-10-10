import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from cli.validate import cmd_validate_map
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
