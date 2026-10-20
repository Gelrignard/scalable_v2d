"""SE3 command update semantics, independent of any task-success claim."""
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from correction.src.optimization.phase_prediction import updated_planning_poses, transport_palms


class PhasePredictionTests(unittest.TestCase):
    def test_update_keeps_observation_and_recovers_target_after_decay(self):
        p = np.tile([3., 4., 5.], (80, 1)); original = p.copy()
        r = Rotation.from_euler('x', .4)
        q = np.tile(r.as_quat(scalar_first=True), (80, 1)); original_q = q.copy()
        delta = Rotation.from_rotvec([0., 0., .15])
        old, new, quat, audit = updated_planning_poses(
            p, q, p[30]+[.03, -.02, .01], (delta*r).as_quat(scalar_first=True),
            np.array([.01, .02, 0.]), boundary=30, acquired=40, fade_begin=50, fade_end=60)
        np.testing.assert_array_equal(p, original)
        np.testing.assert_array_equal(q, original_q)
        np.testing.assert_array_equal(new[:31], old[:31])
        np.testing.assert_allclose(new[40], p[30]+[.03, -.02, .01])
        np.testing.assert_allclose(Rotation.from_quat(quat[40], scalar_first=True).as_matrix(),
                                   (delta*r).as_matrix(), atol=1e-12)
        np.testing.assert_array_equal(new[60:], p[60:])
        np.testing.assert_allclose(quat[60:], q[60:], atol=1e-12)
        _, stationary, stationary_r, _ = updated_planning_poses(
            p, q, p[30]+[.03, -.02, .01], (delta*r).as_quat(scalar_first=True),
            np.array([.01, .02, 0.]), boundary=30, acquired=40, fade_begin=50, fade_end=60,
            stationary_approach=True)
        np.testing.assert_allclose(stationary[30:41], np.tile(stationary[30], (11, 1)))
        np.testing.assert_allclose(stationary_r[30:41], np.tile(stationary_r[30], (11, 1)))
        np.testing.assert_allclose(stationary[60:], p[60:])

    def test_palm_rotates_about_object_not_world_origin(self):
        center = np.array([3., 4., 5.])
        rotation = Rotation.from_euler('z', 90, degrees=True)
        palms, orientations = transport_palms(np.array([center+[1., 0., 0.]]), Rotation.identity(1),
            center, np.array([1., 0., 0., 0.]), center+[.1, 0., 0.],
            rotation.as_quat(scalar_first=True))
        np.testing.assert_allclose(palms, [center+[.1, 1., 0.]], atol=1e-12)
        np.testing.assert_allclose(orientations.as_matrix(), rotation.as_matrix()[None], atol=1e-12)


if __name__ == '__main__':
    unittest.main()
