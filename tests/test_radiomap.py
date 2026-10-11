import unittest
import numpy as np

from sensing.radiomap import Fingerprint, Matcher, RadioMap, localize, pairwise_summary


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


class PairwiseTests(unittest.TestCase):
    def test_summary_matches_exhaustive_pairs(self):
        mat = np.random.default_rng(42).normal(size=(270, 8))
        expected = np.abs(mat[:, None, :] - mat[None, :, :]).sum(axis=2)
        distances = expected[np.triu_indices(len(mat), 1)]
        closest, mean = pairwise_summary(mat)
        self.assertAlmostEqual(closest, distances.min())
        self.assertAlmostEqual(mean, distances.mean())

    def test_degenerate_matrix_is_rejected(self):
        for mat in (np.zeros((1, 5)), np.zeros((2, 0)), np.full((2, 5), np.nan)):
            with self.assertRaises(ValueError):
                pairwise_summary(mat)


class MapInputTests(unittest.TestCase):
    def test_duplicate_keys_cannot_multiply_anchor_evidence(self):
        with self.assertRaises(ValueError):
            RadioMap([point("a", {"a": -60})], ["a"] * 5, {})

    def test_invalid_coordinates_and_values_rejected(self):
        for x in (None, "1", True, float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                Fingerprint(x, 0, "bad", 0, {"a": -60}, 3)
        for values in ({"a": float("nan")}, {"a": True}, {"": -60}):
            with self.assertRaises(ValueError):
                point("bad", values)
