import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from data_tools.observation_completion import complete_object_poses


class ObservationCompletionTests(unittest.TestCase):
    def test_gaps_keep_timeline_mask_and_observed_rotation(self):
        t = np.arange(5, dtype=float)
        p = np.zeros((5, 7))
        p[1] = [1, 0, 0, 1, 0, 0, 0]
        p[3, :3] = [3, 0, 0]
        p[3, 3:] = Rotation.from_euler('z', 90, degrees=True).as_quat(scalar_first=True)
        out, valid = complete_object_poses(t, p, [False, True, False, True, False])
        np.testing.assert_array_equal(valid, [False, True, False, True, False])
        np.testing.assert_allclose(out[:, 0], [1, 1, 2, 3, 3])
        np.testing.assert_allclose(Rotation.from_quat(out[2, 3:], scalar_first=True).as_euler('xyz', degrees=True), [0, 0, 45])
        np.testing.assert_allclose(out[[1, 3]], p[[1, 3]])

    def test_invalid_visible_quaternion_is_not_an_observation(self):
        out, mask = complete_object_poses([0., 1.], [[0,0,0,1,0,0,0], [0]*7], [True, True])
        np.testing.assert_array_equal(mask, [True, False])
        np.testing.assert_array_equal(out[1], out[0])

    def test_no_usable_pose_rejected(self):
        with self.assertRaises(ValueError):
            complete_object_poses([0., 1.], np.zeros((2,7)), [False, False])
