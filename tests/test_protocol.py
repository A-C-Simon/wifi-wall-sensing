import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sensing.detect import Detection, DetectResult
from sensing.io import Session
from validate.score import score_file


class ProtocolTests(unittest.TestCase):
    def test_lead_and_post_protocol_samples_are_excluded(self):
        result = DetectResult(threshold=3, detections=[
            Detection(t, 10 if t in (0, 65) else 0, False, 0, 0, 0)
            for t in (0, 5, 15, 25, 35, 45, 55, 65)])
        with tempfile.TemporaryDirectory() as tmp:
            with patch("validate.score.OUT", Path(tmp)), \
                 patch("validate.score.io.load", return_value=Session()), \
                 patch("validate.score.detect.detect", return_value=result), \
                 patch("validate.score.viz.plot_channels"), \
                 contextlib.redirect_stdout(io.StringIO()):
                scored = score_file("capture", 3, 10, 10, 20, 10, 20, lead=5)
        self.assertEqual(scored["marks"], [5, 15, 35, 45, 65])
        self.assertEqual([p["n"] for p in scored["phases"].values()], [1, 2, 1, 2])
        self.assertTrue(all(p["max"] == 0 for p in scored["phases"].values()))
