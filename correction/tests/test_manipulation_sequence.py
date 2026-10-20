"""Phase assembly invariants; these do not certify physical manipulation."""
import unittest
from types import SimpleNamespace
import numpy as np
from correction.src.optimization.surface_reference import PhaseTiming
from correction.src.optimization.manipulation_sequence import union_channels


class SequenceTests(unittest.TestCase):
    def test_phase_timing_rejects_overlap_and_missing_release(self):
        PhaseTiming(0, 1, 2, 3, 4, 5, 6, 7).validate(8)
        with self.assertRaises(ValueError):
            PhaseTiming(0, 1, 2, 3, 4, 5, 6, 8).validate(8)
        with self.assertRaises(ValueError):
            PhaseTiming(0, 1, 2, 3, 4, 3, 6, 7).validate(8)

    def test_regrasp_changes_patch_only_while_inactive_without_combining_forces(self):
        count = 40
        p = lambda x: SimpleNamespace(body=17, local=np.array([x, 0., 0.]),
                                      object_local=np.array([.1, x, 0.]),
                                      object_outward=np.array([1., 0., 0.]))
        phases = [SimpleNamespace(points=[p(.01)], timing=PhaseTiming(0, 1, 2, 3, 7, 9, 10, 11)),
                  SimpleNamespace(points=[p(.02)], timing=PhaseTiming(18, 20, 22, 25, 30, 32, 35, 39))]
        commands = []
        for a, b, value in [(3, 9, 2.), (23, 32, 3.)]:
            active = np.zeros((count, 1), bool); active[a:b] = True
            commands.append(dict(contact_active=active,
                                 force_ff=np.repeat(active[..., None]*value, 3, axis=2)))
        points, result = union_channels(phases, commands, count)
        self.assertEqual(len(points), 1)
        np.testing.assert_allclose(result['force_ff'][4], 2.)
        np.testing.assert_allclose(result['force_ff'][25], 3.)
        np.testing.assert_array_equal(result['force_ff'][10:22], 0.)
        moving = np.linalg.norm(np.diff(result['robot_local_points'], axis=0), axis=2) > 1e-9
        active = result['contact_active']
        self.assertFalse(np.any(moving & (active[:-1] | active[1:])))
        np.testing.assert_allclose(result['robot_local_points'][4, 0], [.01, 0, 0])
        np.testing.assert_allclose(result['robot_local_points'][25, 0], [.02, 0, 0])


if __name__ == '__main__':
    unittest.main()
