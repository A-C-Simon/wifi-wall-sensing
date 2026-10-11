import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


class CaptureArgumentTests(unittest.TestCase):
    def test_invalid_values_rejected_before_acquisition(self):
        import wifisense
        import wifisense_live
        import wifisense_radar
        from validate import occupancy_test, walk_test
        cases = [(wifisense, ["baseline", "--duration", "0"]),
                 (wifisense, ["capture", "--fine-hz", "nan"]),
                 (wifisense, ["calibrate", "--cols", "0"]),
                 (wifisense_live, ["walktest", "--empty", "0"]),
                 (wifisense_radar, ["--pathloss", "0"]),
                 (walk_test, ["--walk", "0"]),
                 (occupancy_test, ["--block", "nan"])]
        for module, args in cases:
            with self.subTest(module=module.__name__, args=args), \
                 patch("sys.argv", [module.__name__, *args]), \
                 contextlib.redirect_stderr(io.StringIO()), \
                 self.assertRaises(SystemExit) as raised:
                module.main()
            self.assertEqual(raised.exception.code, 2)

    def test_fractional_occupancy_capture_never_becomes_unlimited(self):
        from validate import occupancy_test
        from sensing.io import Session
        with tempfile.TemporaryDirectory() as tmp:
            proc = Mock()
            proc.wait.return_value = 0
            args = ["occupancy", "--block", ".1", "--reps", "1", "--lead", "0",
                    "--out", str(Path(tmp) / "capture")]
            with patch("sys.argv", args), patch("common.build_core"), \
                 patch("validate.occupancy_test.subprocess.Popen", return_value=proc) as popen, \
                 patch("validate.occupancy_test.io.load", return_value=Session()), \
                 contextlib.redirect_stdout(io.StringIO()):
                occupancy_test.main()
            cmd = popen.call_args.args[0]
            self.assertAlmostEqual(float(cmd[cmd.index("--duration") + 1]), .2)

    def test_distance_calibration_rejects_bad_points_before_recording(self):
        from cli.validate import cmd_calibrate_distance
        from types import SimpleNamespace
        for points in ("1,1", "0,2", "nan,2", "inf,2", "bad,2"):
            with patch("cli.validate.build_core") as build, \
                 contextlib.redirect_stdout(io.StringIO()):
                status = cmd_calibrate_distance(SimpleNamespace(points=points, hold=.5))
            self.assertEqual(status, 1)
            build.assert_not_called()
