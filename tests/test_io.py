import json
import tempfile
import threading
import time
import unittest
from pathlib import Path

from sensing import io


class CaptureIOTests(unittest.TestCase):
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
