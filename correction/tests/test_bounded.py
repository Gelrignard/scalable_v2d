"""Mechanics, sparse scaling, temporal and bounded-search regression checks."""
import unittest
import numpy as np

from correction.src.dynamics.sparse_forces import allocate_forces
from correction.src.dynamics.window_audit import box_contacts
from correction.src.optimization.contact_hypotheses import hypotheses


class SparseForceTests(unittest.TestCase):
    def static_case(self, normal, active=True, samples=5):
        return allocate_forces(np.zeros((samples, 1, 3)), np.tile(normal, (samples, 1, 1)),
                               np.full((samples, 1), active), np.tile([0, 0, 9.81, 0, 0, 0], (samples, 1)),
                               np.arange(samples)*.05, np.zeros(3), cap=15.)

    def test_underside_supports_but_top_cannot_lift(self):
        below = self.static_case([0, 0, 1])
        above = self.static_case([0, 0, -1])
        self.assertTrue(below.feasible.all())
        self.assertFalse(above.feasible.any())
        self.assertGreater(np.abs(above.residual[:, 2]).min(), 9.)

    def test_missing_contact_cannot_apply_force(self):
        missing = self.static_case([0, 0, 1], False)
        self.assertFalse(missing.feasible.any())
        np.testing.assert_allclose(missing.forces, 0, atol=1e-5)

    def test_temporal_storage_scales_linearly(self):
        small = self.static_case([0, 0, 1], samples=10)
        large = self.static_case([0, 0, 1], samples=100)
        self.assertTrue(large.feasible.all())
        self.assertLess(large.matrix_nnz, 12*small.matrix_nnz)

    def test_friction_and_torque_cannot_be_ignored(self):
        p = np.tile([[-.05, .03, 0], [.05, .03, 0]], (5, 1, 1))
        n = np.tile([[1, 0, 0], [-1, 0, 0]], (5, 1, 1))
        demand = np.tile([0, 0, 9.81, 0, 0, 0], (5, 1))
        result = allocate_forces(p, n, np.ones((5, 2), bool), demand, np.arange(5)*.1,
                                 np.zeros(3), mu=.6, cap=15.)
        self.assertFalse(result.feasible.any())
        self.assertGreater(np.linalg.norm(result.residual[:, 3:], axis=1).max(), .002)


class ContactHypothesisTests(unittest.TestCase):
    def setUp(self):
        self.bounds = np.array([[-.1, -.01, -.15], [.1, .01, .15]])

    def test_embedded_points_are_signed_and_faces_differ(self):
        p, n, d = box_contacts(np.array([[[0., 0., 0.], [0., .02, 0.]]]), self.bounds)
        np.testing.assert_allclose(d, [[-.01, .01]])
        self.assertEqual(n[0, 1, 1], -1.)

    def test_contact_search_is_bounded_and_demonstration_local(self):
        source = np.tile([[-.02, 0, 0], [0, 0, 0], [.02, 0, 0]]*2, (10, 1, 1))
        candidates = hypotheses(source, source, self.bounds, np.ones((10, 6), bool), max_candidates=4)
        self.assertLessEqual(len(candidates), 4)
        self.assertGreater(len(candidates), 1)
        for c in candidates:
            self.assertTrue(np.all(np.linalg.norm(c.points-source[0], axis=-1)[c.active.any(axis=0)] <= .08))


if __name__ == '__main__':
    unittest.main()
