import unittest

from sensing.radiomap import Fingerprint, Matcher, RadioMap, localize


def point(label, values):
    return Fingerprint(0, 0, label, 0, values, 3)


class LocalizationTests(unittest.TestCase):
    def setUp(self):
        self.keys = list("abcde")
        self.a = dict.fromkeys(self.keys, -60.)
        self.b = dict.fromkeys(self.keys, -40.)
        self.rm = RadioMap([point("a", self.a), point("b", self.b)], self.keys, {})

    def test_no_shared_evidence_cannot_produce_a_position(self):
        for live in ({}, {"unknown": -60}, {"a": -60}, dict.fromkeys(self.keys, float("nan"))):
            self.assertEqual(localize(self.rm, live), [])

    def test_measured_match_and_distance(self):
        ranked = Matcher(self.rm).localize(self.b, 2)
        self.assertEqual([p.label for p, _ in ranked], ["b", "a"])
        self.assertEqual([d for _, d in ranked], [0, 20])

    def test_missing_map_anchor_is_not_copied_from_another_cell(self):
        del self.a["a"]
        self.assertEqual(localize(self.rm, self.b), [])

    def test_cell_order_does_not_change_distances(self):
        reverse = RadioMap(self.rm.points[::-1], self.keys, {})
        for rm in (self.rm, reverse):
            self.assertEqual([(p.label, d) for p, d in localize(rm, self.a)],
                             [("a", 0), ("b", 20)])
