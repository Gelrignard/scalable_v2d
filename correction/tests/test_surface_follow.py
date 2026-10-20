"""Bounds and geometry/sign checks for the optional live-surface adapter."""
import unittest
from types import SimpleNamespace
import mujoco
import numpy as np
from correction.src.dynamics.surface_follow import SurfaceFollowConfig, BoundedOffset, SurfaceFollowController, safe_follow_columns
from correction.src.dynamics.contact_controller import ContactControllerConfig, G1ContactController


class SurfaceFollowTests(unittest.TestCase):
    def test_nominal_margin_catches_between_sample_overshoot(self):
        q = np.zeros((2, 3)); v = np.array([[6., 0., 0.], [-6., 0., 0.]])
        allowed, margin = safe_follow_columns(q, v, [0., 1.], -np.ones(3), np.ones(3),
                                               [False, False, True], .08, 0)
        np.testing.assert_array_equal(allowed, [False, True, False])
        self.assertLess(margin[0], 0.)

    def test_offset_step_bounds_and_limit_conflict_are_observable(self):
        cfg = SurfaceFollowConfig()
        offset = BoundedOffset(1, cfg)
        for _ in range(1000):
            offset.step(np.array([.04]), np.zeros(1), np.array([-1.]), np.array([1.]), .001)
        self.assertGreater(offset.position[0], .039)
        self.assertLessEqual(offset.maximum_speed, cfg.joint_speed_limit_rad_s+1e-9)
        self.assertLessEqual(offset.maximum_acceleration, cfg.joint_acceleration_limit_rad_s2+1e-9)
        self.assertEqual(offset.conflicts, 0)
        offset.step(np.array([.04]), np.array([1.]), np.array([-1.]), np.array([1.]), .001)
        self.assertEqual(offset.position[0], 0.)
        self.assertGreater(offset.conflicts, 0)

    def test_current_patch_feedback_preserves_prefix_and_never_writes_object(self):
        model = mujoco.MjModel.from_xml_string('''<mujoco><option gravity="0 0 0" timestep=".001"/>
          <worldbody><body name="object"><freejoint/><geom name="box" type="box" size=".05 .05 .05"/></body>
          <body name="finger" pos=".07 0 0"><joint type="slide" axis="1 0 0"/>
            <geom type="sphere" size=".01" mass=".1"/></body></worldbody></mujoco>''')
        data = mujoco.MjData(model); data.qpos[0] = .005; mujoco.mj_forward(model, data)
        scene = SimpleNamespace(model=model, data=data, book_q=0, book_body=model.body('object').id,
            active_v=np.array([6]), movable=np.array([0]), hand=np.array([False]),
            object_geoms={model.geom('box').id}, bounds=np.array([[-.05]*3, [.05]*3]),
            kin=SimpleNamespace(robot=SimpleNamespace(model=SimpleNamespace(
                lowerPositionLimit=np.full(8, -1.), upperPositionLimit=np.full(8, 1.)))))
        point = SimpleNamespace(body=model.body('finger').id, local=np.array([-.005, 0, 0]),
                                object_local=np.array([.04, 0, 0]))
        times = np.linspace(0, .8, 9); active = np.ones((9, 1), bool); active[:3] = False
        local = np.tile([-.01, 0., 0.], (9, 1, 1)); local[:2] = [-.005, 0, 0]
        anchors = np.tile([.05, 0., 0.], (9, 1, 1)); anchors[:2] = [.04, 0, 0]
        kwargs = dict(scene=scene, points=[point], normals_local=np.tile([-1., 0, 0], (9, 1, 1)),
            active=active, times=times, config=ContactControllerConfig(patch_distance_gate=True), kp=np.array([20.]),
            point_local_trajectory=local, object_local_trajectory=anchors)
        for cfg in (SurfaceFollowConfig(frames=(2, 3, 6, 7)),
                    SurfaceFollowConfig(frames=None), SurfaceFollowConfig(frames=None, gain=0)):
            with self.subTest(config=cfg):
                controller = SurfaceFollowController(**kwargs, follow_config=cfg,
                    nominal_positions=np.zeros((9, 1)), nominal_velocities=np.zeros((9, 1)),
                    planning_poses=dict(object_positions=np.zeros((9, 3)), object_wxyz=np.tile([1., 0, 0, 0], (9, 1))))
                baseline = G1ContactController(**kwargs); physical = data.qpos.copy()
                maximum = 0.
                for step in range(801):
                    time = step*.001
                    actual = controller.step(time, np.zeros(1), np.zeros(1), np.zeros((1, 3)), np.zeros(1))
                    if time <= .2 or cfg.gain == 0:
                        original = baseline.step(time, np.zeros(1), np.zeros(1), np.zeros((1, 3)), np.zeros(1))
                        np.testing.assert_array_equal(actual, original)
                    maximum = max(maximum, actual[0][0])
                # Positive object displacement moves the target positively.
                if cfg.gain:
                    self.assertGreater(maximum, .001)
                    self.assertLess(maximum, .003)
                else:
                    self.assertEqual(maximum, 0.)
                np.testing.assert_array_equal(data.qpos, physical)
                np.testing.assert_array_equal(data.qfrc_applied, np.zeros(model.nv))
                self.assertEqual(controller.follow.conflicts, 0)

    def test_inherited_planning_shift_is_not_confused_with_task_target(self):
        from correction.scripts.replay_nominal_residual import inherited_planning_poses
        reference = dict(q=np.zeros((10, 1)), object_positions=np.zeros((10, 3)),
            object_wxyz=np.tile([1., 0, 0, 0], (10, 1)), contact_active=np.zeros((10, 1), bool))
        reference['contact_active'][5:9] = True
        sequence = dict(command=dict(second_planning_translation_m=[.05, 0, 0]),
            phase_contacts=[{}, dict(timing=dict(approach_start=3, force_full=5, contact_end=8))])
        poses = inherited_planning_poses(reference, sequence)
        np.testing.assert_array_equal(poses['object_positions'][:3], 0.)
        np.testing.assert_allclose(poses['object_positions'][3:6, 0], .05)
        np.testing.assert_allclose(poses['object_positions'][8:], 0.)
        np.testing.assert_array_equal(reference['object_positions'], 0.)
        reference['contact_active'][3] = True
        with self.assertRaises(ValueError):
            inherited_planning_poses(reference, sequence)


if __name__ == '__main__':
    unittest.main()
