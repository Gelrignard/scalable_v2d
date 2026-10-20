"""Independent boundary and physical-force checks for command proposals."""
from types import SimpleNamespace
import unittest
import numpy as np
import mujoco
from correction.src.optimization.servo_preload import phase_envelope, fit_load_proposal, apply_offset


class ServoPreloadTests(unittest.TestCase):
    def test_envelope_is_zero_at_start_end_and_c2_at_transitions(self):
        times = np.linspace(0, 5, 5001)
        value, velocity, acceleration = phase_envelope(times, [1., 2., 3., 4.])
        np.testing.assert_array_equal(value[times <= 1.], 0.)
        np.testing.assert_array_equal(value[times >= 4.], 0.)
        np.testing.assert_array_equal(value[(times >= 2.) & (times <= 3.)], 1.)
        np.testing.assert_allclose(velocity[[1000, 2000, 3000, 4000]], 0., atol=1e-12)
        np.testing.assert_allclose(acceleration[[1000, 2000, 3000, 4000]], 0., atol=1e-12)
        np.testing.assert_allclose(np.gradient(value, times)[20:-20], velocity[20:-20], atol=1e-5)

    def test_servo_load_error_direction_unloads_an_actual_contact(self):
        model = mujoco.MjModel.from_xml_string('''<mujoco><option timestep=".001" gravity="0 0 0"/>
          <worldbody><geom type="box" size=".1 .1 .1"/>
            <body pos=".11 0 0"><joint type="slide" axis="1 0 0" damping="1"/>
              <geom type="sphere" size=".01" mass=".1"/></body></worldbody></mujoco>''')
        data = mujoco.MjData(model)
        def force(target):
            mujoco.mj_resetData(model, data)
            for _ in range(1500):
                data.qfrc_applied[0] = 200*(target-data.qpos[0])-2*data.qvel[0]-6
                mujoco.mj_step(model, data)
            wrench = np.zeros(6); normal = 0.
            for i in range(data.ncon):
                mujoco.mj_contactForce(model, data, i, wrench)
                normal += wrench[0]
            return normal
        before = force(0.)
        # Desired object force is -2X, measured -6X. A positive slider
        # command moves the finger away from the box and should unload it.
        alpha, _ = fit_load_proposal([[4.]], [before-2.], [10.])
        after = force(.02*alpha[0])
        # MuJoCo's compliant contact shifts the equilibrium away from q=0,
        # so the position servo partly unloads the constant 6N feedforward.
        self.assertGreater(before, 5.)
        self.assertLess(before, 6.)
        self.assertAlmostEqual(after, 2., delta=.2)

    def test_composite_command_preserves_initial_state_and_inactive_path(self):
        times = np.linspace(0, 5, 151)
        q = np.zeros((151, 8)); q[:, 6] = 1.
        scene = SimpleNamespace(movable=np.array([0]), kin=SimpleNamespace(robot=SimpleNamespace(
            model=SimpleNamespace(lowerPositionLimit=np.full(8, -1.),
                upperPositionLimit=np.full(8, 1.), velocityLimit=np.ones(7)))))
        ref = dict(q=q, timestamps=times, command_velocity=np.zeros((151, 1)),
                   initial_q=q[0].copy(), initial_velocity=np.zeros(1),
                   spline_knots=np.array([1.]), spline_coefficients=np.array([1.]))
        command, audit = apply_offset(scene, ref, np.array([[.08]]), [1.], frame_bounds=(30, 60, 90, 120))
        self.assertTrue(audit['accepted'])
        np.testing.assert_array_equal(command['q'][120:], ref['q'][120:])
        np.testing.assert_array_equal(command['initial_q'], ref['initial_q'])
        np.testing.assert_array_equal(command['command_velocity'][[0, -1]], 0.)
        self.assertNotIn('spline_knots', command)
        self.assertEqual(float(ref['q'][:, 7].max()), 0.)


if __name__ == '__main__':
    unittest.main()
