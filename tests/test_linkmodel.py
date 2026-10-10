import math
import unittest

from sensing.linkmodel import LinkModel, estimate_uncertainty


class LinkModelTests(unittest.TestCase):
    def test_recovers_known_model(self):
        truth = LinkModel(-35, 3.1)
        d = [1, 2, 4, 8]
        fitted = LinkModel().fit(d, [truth.rssi_at(x) for x in d])
        self.assertAlmostEqual(fitted.A, -35)
        self.assertAlmostEqual(fitted.n, 3.1)
        self.assertAlmostEqual(fitted.range_of(truth.rssi_at(4)), 4)

    def test_invalid_models_rejected(self):
        for n in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                LinkModel(n=n)

    def test_invalid_fit_preserves_existing_model(self):
        m = LinkModel()
        for distances, rssis in (([1, 2], [-60, -60]), ([1, 2], [-60, -50]),
                                 ([1, 1], [-50, -60]), ([1, 2], [-60]),
                                 ([1, float("inf")], [-50, -60])):
            with self.assertRaises(ValueError):
                m.fit(distances, rssis)
            self.assertEqual((m.A, m.n), (-30, 2.7))

    def test_extreme_extrapolation_does_not_overflow(self):
        self.assertTrue(math.isinf(LinkModel(n=.0001).range_of(-100)))
        with self.assertRaises(ValueError):
            estimate_uncertainty(LinkModel(), -60, (0, 4))
