"""Controller equivalence, clipping, contact direction and fallback behavior."""
from types import SimpleNamespace
import unittest
import mujoco
import numpy as np
from correction.src.dynamics.contact_controller import (
    ContactControllerConfig, G1ContactController, NormalForceFeedback, ContactEngagement, position_target)


class ContactControllerTests(unittest.TestCase):
    def test_engagement_decay_is_explicit_and_inactive_modes_clear_it(self):
        gate = ContactEngagement(2, .03)
        gain = gate.update([True, False], [True, True], .001)
        np.testing.assert_allclose(gain, [1/31, 0.])
        gain = gate.update([False, True], [True, True], .001)
        np.testing.assert_allclose(gain, [30/31**2, 1/31])
        gain = gate.update([False, True], [False, True], .001)
        self.assertEqual(gain[0], 0.)
        direct = ContactEngagement(2, 0.)
        np.testing.assert_array_equal(direct.update([True, False], [True, True], .001), [1., 0.])
        np.testing.assert_array_equal(direct.update([False, True], [True, True], .001), [0., 1.])

    def test_position_conversion_preserves_torque_under_tracking_error(self):
        q, v, actual_q, actual_v = np.array([[.2, -.4], [.1, .2], [.21, -.38], [.12, .18]])
        kp, kd, contact, compensation = np.array([[300., 20.], [8., .4], [2., .3], [1., .2]])
        target, requested, lost = position_target(q, contact, kp, -2., 2., .1)
        torque_mode = kp*(q-actual_q)+kd*(v-actual_v)+compensation+contact
        position_mode = kp*(target-actual_q)+kd*(v-actual_v)+compensation
        np.testing.assert_allclose(position_mode, torque_mode, atol=1e-13)
        np.testing.assert_allclose(lost, 0., atol=1e-13)

    def test_conversion_reports_force_lost_to_joint_and_offset_limits(self):
        target, requested, lost = position_target([.99, 0.], [10., 10.], [20., 20.], -1., 1., .1)
        np.testing.assert_allclose(target, [1., .1])
        np.testing.assert_allclose(lost, [9.8, 8.])
        self.assertTrue(np.all(requested > target))
        with self.assertRaises(ValueError): position_target(0, 1, 0, -1, 1, .1)

    def test_feedback_is_bounded_rate_limited_and_cleared_on_contact_loss(self):
        control = NormalForceFeedback(2, ContactControllerConfig(feedback_gain=.25))
        for _ in range(100):
            normal, correction = control.update([10., 10.], [0., 30.], [True, True], [True, True], .001)
            self.assertLessEqual(np.max(abs(correction)), 2.)
        self.assertGreater(correction[0], 0)
        self.assertLess(correction[1], 0)
        normal, correction = control.update([10., 10.], [0., 0.], [False, True], [True, False], .001)
        np.testing.assert_array_equal(correction, [0., 0.])
        np.testing.assert_array_equal(control.filtered, [0., 0.])
        normal, correction = control.update([10., 10.], [0., 0.], [True, True], [True, True], .001)
        self.assertLessEqual(np.max(abs(correction)), .1+1e-12)

    def test_force_sensor_sign_and_wrong_face_gate_in_actual_mujoco_contact(self):
        # Independent fixture: a finite-mass finger on the +X object surface.
        xml = '''<mujoco><option timestep=".001" gravity="0 0 0"/><worldbody>
          <body name="object"><freejoint name="object_root"/><geom name="object_geom" type="box" size=".1 .1 .1" mass="1"/></body>
          <body name="finger" pos=".1099 0 0"><joint name="joint" type="slide" axis="1 0 0" range="-.2 .2"/>
          <geom name="finger_geom" type="sphere" size=".01" mass=".1"/></body>
          </worldbody></mujoco>'''
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        scene = SimpleNamespace(model=model, data=data, book_q=0, book_body=model.body('object').id, active_v=np.array([6]), movable=np.array([0]),
            hand=np.array([True]), object_geoms={model.geom('object_geom').id}, bounds=np.array([[-.1]*3, [.1]*3]),
            kin=SimpleNamespace(robot=SimpleNamespace(model=SimpleNamespace(
                lowerPositionLimit=np.full(8, -.2), upperPositionLimit=np.full(8, .2)))))
        point = SimpleNamespace(body=model.body('finger').id, local=np.array([-.01, 0, 0]))
        kwargs = dict(scene=scene, points=[point], active=np.ones((2, 1), bool), times=np.array([0., .1]),
                      config=ContactControllerConfig(feedback_gain=.25), kp=np.array([20.]))
        controller = G1ContactController(normals_local=np.tile([-1., 0, 0], (2, 1, 1)), **kwargs)
        with self.assertRaises(ValueError):
            G1ContactController(normals_local=np.tile([-1., 0, 0], (2, 1, 1)),
                point_local_trajectory=np.array([[[-.01, 0., 0.]], [[-.01, .05, 0.]]]), **kwargs)
        controller.step(0, np.zeros(1), np.zeros(1), np.array([[-6., 0, 0]]), np.zeros(1))
        self.assertTrue(controller.last['touching'][0])
        self.assertLess(controller.last['measured_contact_force'][0, 0], 0)
        self.assertLess(controller.last['contact_torque'][0], 0)
        # Feedback/no-feedback use identical friction limiting at zero error.
        sensed = -controller.last['measured_contact_force'][0, 0]
        limited = []
        for gain in (0., .25):
            options = dict(kwargs, config=ContactControllerConfig(feedback_gain=gain))
            test = G1ContactController(normals_local=np.tile([-1., 0, 0], (2, 1, 1)), **options)
            test.feedback.filtered[:] = sensed
            test.step(0, np.zeros(1), np.zeros(1), np.array([[-sensed, 2*sensed, 0]]), np.zeros(1))
            limited.append(test.last['commanded_contact_force'])
        np.testing.assert_allclose(limited[0], limited[1], atol=1e-12)
        self.assertLessEqual(abs(limited[0][0, 1]), .45*sensed+1e-12)
        reverse = G1ContactController(normals_local=np.tile([1., 0, 0], (2, 1, 1)), **kwargs)
        reverse.step(0, np.zeros(1), np.zeros(1), np.array([[6., 0, 0]]), np.zeros(1))
        self.assertFalse(reverse.last['touching'][0])
        self.assertEqual(reverse.last['feedback_correction_n'][0], 0.)
        # Across a large gap, neither feedback nor feedforward can act.
        data.qpos[0] = 1.
        mujoco.mj_forward(model, data)
        controller.step(.001, np.zeros(1), np.zeros(1), np.array([[-6., 0, 0]]), np.zeros(1))
        np.testing.assert_array_equal(controller.last['contact_torque'], [0.])
        np.testing.assert_array_equal(controller.last['feedback_correction_n'], [0.])
        np.testing.assert_array_equal(data.qfrc_applied[:6], np.zeros(6))
        # An infeasible physical reference cannot silently pass conflicting
        # servo joint-limit and offset-rate constraints.
        options = dict(kwargs, config=ContactControllerConfig(interface='position'))
        invalid = G1ContactController(normals_local=np.tile([-1., 0, 0], (2, 1, 1)), **options)
        target, _ = invalid.step(0, np.array([.3]), np.zeros(1), np.zeros((1, 3)), np.zeros(1))
        self.assertLessEqual(target[0], .2)
        self.assertEqual(invalid.incompatible_limit_steps, 1)
        self.assertFalse(invalid.export()['controller_rate_contract_satisfied'])

    def test_failed_or_shakier_candidate_never_replaces_fallback(self):
        from correction.scripts.compare_g1_controllers import choose_controller
        base = dict(accepted_physics=False, rms_position_error_mm=10., rms_joint_acceleration_rad_s2=1.)
        cases = {name: dict(base) for name in ('legacy_feedforward', 'gated_feedforward',
                  'position_equivalent', 'force_feedback', 'position_feedback')}
        cases['force_feedback'].update(accepted_physics=True, rms_position_error_mm=5., rms_joint_acceleration_rad_s2=2.)
        choice = choose_controller(cases)
        self.assertEqual(choice['selected'], 'legacy_feedforward')
        self.assertFalse(choice['fallback_is_feasible'])
        cases['force_feedback']['rms_joint_acceleration_rad_s2'] = .9
        self.assertEqual(choose_controller(cases)['selected'], 'force_feedback')

    def test_empty_object_interior_is_not_a_near_surface_contact(self):
        xml = '''<mujoco><option gravity="0 0 0"/><worldbody>
          <body name="object"><freejoint/>
            <geom name="wall_left" type="box" pos="-.11 0 0" size=".01 .1 .1"/>
            <geom name="wall_right" type="box" pos=".11 0 0" size=".01 .1 .1"/>
          </body>
          <body name="finger"><joint type="slide" axis="1 0 0"/>
            <geom type="sphere" size=".01" mass=".1"/>
          </body></worldbody></mujoco>'''
        model = mujoco.MjModel.from_xml_string(xml); data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        scene = SimpleNamespace(model=model, data=data, book_q=0, book_body=model.body('object').id, active_v=np.array([6]), movable=np.array([0]),
            hand=np.array([True]), object_geoms={model.geom('wall_left').id, model.geom('wall_right').id},
            bounds=np.array([[-.12, -.1, -.1], [.12, .1, .1]]),
            kin=SimpleNamespace(robot=SimpleNamespace(model=SimpleNamespace(
                lowerPositionLimit=np.full(8, -1.), upperPositionLimit=np.full(8, 1.)))))
        point = SimpleNamespace(body=model.body('finger').id, local=np.array([.01, 0, 0]),
                                object_local=np.array([.1, 0, 0]))
        controller = G1ContactController(scene, [point], np.tile([1.,0,0], (2,1,1)),
            np.ones((2,1), bool), np.array([0., .1]), ContactControllerConfig(patch_distance_gate=True), np.array([20.]))
        controller.step(0., np.zeros(1), np.zeros(1), np.array([[5.,0,0]]), np.zeros(1))
        self.assertFalse(controller.last['touching'][0])
        self.assertFalse(controller.last['force_enabled'][0])
        np.testing.assert_allclose(controller.last['contact_gate_gap_m'], [.09])
        np.testing.assert_array_equal(controller.last['contact_torque'], [0.])


if __name__ == '__main__': unittest.main()
