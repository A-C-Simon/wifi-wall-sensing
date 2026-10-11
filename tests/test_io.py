import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from sensing import io


class CaptureIOTests(unittest.TestCase):
    def test_cell_paths_preserve_precision_and_read_legacy_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = io.cell_capture_path(tmp, .001, 0, for_write=True)
            b = io.cell_capture_path(tmp, .002, 0, for_write=True)
            self.assertNotEqual(a, b)
            legacy = Path(tmp) / "cell_2.9_0.0.jsonl"
            legacy.touch()
            self.assertEqual(io.cell_capture_path(tmp, 2.9000000000000004, 0), legacy)
            self.assertNotEqual(io.cell_capture_path(tmp, 2.9000000000000004, 0,
                                                     for_write=True), legacy)

    def test_corrupt_complete_records_have_file_and_line_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "capture"
            for bad in ('not json', 'null', '[]', '{"type":"fine","t":1}',
                        '{"type":"sweep","t":NaN,"aps":[]}',
                        '{"type":"sweep","t":1,"aps":null}'):
                path.write_text('{"type":"meta"}\n' + bad + '\n')
                with self.assertRaisesRegex(ValueError, 'capture:2:'):
                    io.load(path)

    def test_incomplete_final_json_keeps_prior_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "capture"
            path.write_text('{"type":"meta","iface":"en0"}\n{"type":')
            self.assertEqual(io.load(path).meta["iface"], "en0")

    def test_invalid_fine_values_are_rejected_before_filtering(self):
        record = dict(type="fine", t=1, n=10, rssi=-60, min=-61, max=-59,
                      sd=1, chg=2, noise=-90, rate=100)
        for field, value in (("n", 0), ("n", True), ("rssi", float("nan")),
                             ("rssi", -70), ("sd", -1), ("chg", 11), ("t", -1)):
            with self.assertRaises(ValueError):
                io.validate_record(dict(record, **{field: value}))

    def test_live_corruption_is_reported_without_mutating_series(self):
        from wifisense_live import State, reader
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "capture"
            path.write_text('{"type":"fine","t":1,"rssi":NaN}\n')
            st = State()
            reader(path, st, {})
            self.assertIn("invalid capture record", st.error)
            self.assertEqual(len(st.fine_t), 0)

    def test_anchor_aggregation_does_not_depend_on_row_order(self):
        rows = [{"ch": 1, "w": 2, "rssi": r} for r in (-70, -40, -50)]
        self.assertEqual(io.scan_aps(rows), {"1/2": -40})
        self.assertEqual(io.scan_aps(rows[::-1]), {"1/2": -40})

    def test_sweeps_sorted_before_deduplication(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "capture"
            path.write_text("".join(json.dumps({"type": "sweep", "t": t, "aps": []}) + "\n"
                                    for t in (12, 0, 24)))
            self.assertEqual([sw.t for sw in io.load(path).sweeps], [0, 12, 24])

    def test_live_reader_waits_for_complete_record(self):
        from wifisense_live import State, reader
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "capture"
            record = json.dumps({"type": "meta", "rssi_dbm": -60}) + "\n"
            path.write_text(record[:10])
            st = State()
            worker = threading.Thread(target=reader, args=(path, st, {}))
            worker.start()
            try:
                time.sleep(.08)
                with path.open("a") as f:
                    f.write(record[10:])
                self.assertTrue(st.ready.wait(2))
                self.assertEqual(st.snapshot()["meta"]["rssi_dbm"], -60)
            finally:
                st.stop = True
                worker.join(2)
            self.assertFalse(worker.is_alive())

    def test_reader_reports_process_exit_without_hanging(self):
        from wifisense_live import State, reader
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "capture"
            path.write_text('{"type": "meta"}\n')
            st = State()
            process = Mock(returncode=2)
            process.poll.return_value = 2
            reader(path, st, {"proc": process})
            self.assertEqual(st.error, "capture ended (exit 2)")
            self.assertFalse(st.snapshot()["fresh"])

    def test_stale_samples_are_not_current_evidence(self):
        from wifisense_live import State
        st = State()
        st.start = 100
        st.fine_t.append(5)
        with patch("wifisense_live.time.monotonic", return_value=105.5):
            self.assertTrue(st.snapshot()["fresh"])
        with patch("wifisense_live.time.monotonic", return_value=107):
            self.assertFalse(st.snapshot()["fresh"])
