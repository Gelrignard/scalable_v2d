import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from correction.dynamics import object_wrench, world_points_to_object
from correction.models import RigidBody


class DynamicsTests(unittest.TestCase):
    def setUp(self):
        self.times = np.linspace(0, 2, 121)
        self.poses = np.zeros((len(self.times), 7))
        self.poses[:, 3] = 1
        self.visible = np.ones(len(self.times), dtype=bool)
        self.body = RigidBody(2., np.zeros(3), np.diag([.02, .03, .04]), np.array([0, 0, -9.81]), "analytic test")

    def test_static_gravity(self):
        result = object_wrench(self.times, self.poses, self.visible, self.body)
        np.testing.assert_allclose(result["required_wrench_object"][result["valid_wrench"]],
                                   np.tile([0, 0, 19.62, 0, 0, 0], (111, 1)), atol=1e-9)

    def test_constant_acceleration(self):
        self.poses[:, 2] = .5 * 3 * self.times**2
        result = object_wrench(self.times, self.poses, self.visible, self.body)
        actual = result["required_wrench_object"][result["valid_wrench"]]
        np.testing.assert_allclose(actual[:, 2], 2 * (3 + 9.81), atol=1e-8)
        np.testing.assert_allclose(actual[:, 3:], 0, atol=1e-10)

    def test_free_fall_requires_no_contact_force(self):
        self.poses[:, 2] = -.5 * 9.81 * self.times**2
        result = object_wrench(self.times, self.poses, self.visible, self.body)
        np.testing.assert_allclose(result["required_wrench_object"][result["valid_wrench"]], 0, atol=1e-8)

    def test_body_rate_and_gyroscopic_torque_with_com_offset(self):
        omega = np.array([.4, .3, .2])
        offset = np.array([.12, -.04, .03])
        start = Rotation.from_euler("xyz", [.2, -.4, .7])
        rotation = start * Rotation.from_rotvec(self.times[:, None] * omega)
        self.poses[:, 3:] = rotation.as_quat()[:, [3, 0, 1, 2]]
        body = RigidBody(2., offset, self.body.inertia_com_object_kg_m2, np.zeros(3), "analytic spin")
        result = object_wrench(self.times, self.poses, self.visible, body, window=1)
        valid = result["valid_wrench"]
        np.testing.assert_allclose(result["angular_velocity_object_rad_s"][valid], np.tile(omega, (valid.sum(), 1)), atol=1e-9)
        expected_force = body.mass_kg * np.cross(omega, np.cross(omega, offset))
        expected_torque = np.cross(omega, body.inertia_com_object_kg_m2 @ omega)
        np.testing.assert_allclose(result["required_wrench_object"][valid],
                                   np.tile(np.r_[expected_force, expected_torque], (valid.sum(), 1)), atol=1e-8)

    def test_world_frame_rotation_does_not_change_object_wrench(self):
        self.poses[:, 0] = .3 * self.times**2
        original = object_wrench(self.times, self.poses, self.visible, self.body)
        rotation = Rotation.from_rotvec([.3, -.2, .5])
        transformed = self.poses.copy()
        transformed[:, :3] = rotation.apply(self.poses[:, :3]) + [4, 2, -3]
        transformed[:, 3:] = rotation.as_quat()[[3, 0, 1, 2]]
        body = RigidBody(2., np.zeros(3), self.body.inertia_com_object_kg_m2,
                         rotation.apply(self.body.gravity_world_m_s2), "rotated test")
        result = object_wrench(self.times, transformed, self.visible, body)
        valid = result["valid_wrench"]
        np.testing.assert_allclose(result["required_wrench_object"][valid], original["required_wrench_object"][valid], atol=1e-8)

    def test_missing_poses_are_not_interpolated(self):
        self.visible[55:65] = False
        self.poses[55:65] = 0
        result = object_wrench(self.times, self.poses, self.visible, self.body)
        self.assertFalse(np.any(result["valid_wrench"][50:70]))
        self.assertTrue(np.isnan(result["required_wrench_object"][50:70]).all())
        np.testing.assert_array_equal(result["raw_pose_xyz_wxyz"], self.poses)
        np.testing.assert_array_equal(result["timestamp_s"], self.times)

    def test_quaternion_sign_flips_do_not_create_motion(self):
        self.poses[::2, 3:] *= -1
        result = object_wrench(self.times, self.poses, self.visible, self.body)
        np.testing.assert_allclose(result["required_wrench_object"][result["valid_wrench"], 3:], 0, atol=1e-10)

    def test_visible_zero_quaternion_rejected(self):
        self.poses[50, 3:] = 0
        with self.assertRaisesRegex(ValueError, "zero quaternion"):
            object_wrench(self.times, self.poses, self.visible, self.body)

    def test_nonuniform_time_rejected(self):
        self.times[40] += .001
        with self.assertRaisesRegex(ValueError, "nonuniform"):
            object_wrench(self.times, self.poses, self.visible, self.body)

    def test_object_relative_conversion(self):
        rotation = Rotation.from_euler("xyz", [.2, .3, -.4])
        self.poses[:, :3] = [1, 2, 3]
        self.poses[:, 3:] = rotation.as_quat()[[3, 0, 1, 2]]
        local = np.array([[.1, .2, .3], [-.2, .3, .1]])
        world = np.tile(rotation.apply(local) + [1, 2, 3], (len(self.times), 1, 1))
        np.testing.assert_allclose(world_points_to_object(self.poses, world), np.tile(local, (len(self.times), 1, 1)), atol=1e-10)

    def test_nonphysical_inertia_rejected(self):
        with self.assertRaisesRegex(ValueError, "triangle"):
            RigidBody(1, np.zeros(3), np.diag([1, 1, 3]), np.zeros(3), "test")


if __name__ == "__main__":
    unittest.main()
