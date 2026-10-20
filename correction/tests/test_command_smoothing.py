"""Check spline derivatives against SciPy's independent continuous evaluator."""
import unittest
from types import SimpleNamespace
import mujoco
import numpy as np
from scipy.interpolate import BSpline, CubicHermiteSpline
from correction.src.optimization.command_smoothing import spline_operators, refine_commands, frozen_prefix_coefficients


class CommandSplineTests(unittest.TestCase):
    def test_fixed_prefix_preserves_continuous_curve_and_two_derivatives(self):
        t = np.linspace(0., 2., 61)
        knots, b, _, _ = spline_operators(t)
        original = np.random.default_rng(815).normal(size=(b.shape[1], 2))
        constant, free = frozen_prefix_coefficients(knots, original, t[30])
        updated = constant.copy()
        updated[free] = np.random.default_rng(817).normal(size=len(free))
        first = BSpline(knots, original, 3)
        second = BSpline(knots, updated.reshape(original.shape), 3)
        for derivative in range(3):
            np.testing.assert_array_equal(first(np.linspace(0., 1., 1003), derivative),
                                          second(np.linspace(0., 1., 1003), derivative))
        self.assertGreater(np.linalg.norm(first(1.5)-second(1.5)), .01)
        with self.assertRaises(ValueError):
            frozen_prefix_coefficients(knots, original, t[31])

    def test_derivatives_and_physics_interpolation_match_between_samples(self):
        times = np.linspace(0., 2., 61)
        knots, b, v, a = spline_operators(times, 3)
        coefficients = np.random.default_rng(821).normal(size=(b.shape[1], 2))
        spline = BSpline(knots, coefficients, 3)
        np.testing.assert_allclose(b@coefficients, spline(times), atol=1e-12)
        np.testing.assert_allclose(v@coefficients, spline(times, 1), atol=1e-12)
        np.testing.assert_allclose(a@coefficients, spline(times, 2), atol=1e-10)
        applied = CubicHermiteSpline(times, b@coefficients, v@coefficients)
        dense = np.linspace(0., 2., 1001)
        np.testing.assert_allclose(applied(dense), spline(dense), atol=1e-12)
        np.testing.assert_allclose(applied(dense, 2), spline(dense, 2), atol=1e-9)

    def test_one_kg_vertical_axis_matches_independent_force_balance(self):
        model = mujoco.MjModel.from_xml_string('''<mujoco><option gravity="0 0 -9.81"/>
          <worldbody><body><freejoint/><geom type="sphere" size=".1" mass="1"/>
            <body pos="0 0 1"><joint type="slide" axis="0 0 1" damping="0"/>
              <geom type="sphere" size=".05" mass="1"/>
            </body></body></worldbody></mujoco>''')
        data = mujoco.MjData(model)
        def state(q, position, rotation):
            data.qpos[:7] = np.r_[q[:3], q[[6,3,4,5]]]
            data.qpos[7] = q[7]; data.qvel[:] = 0
            mujoco.mj_forward(model, data)
        scene = SimpleNamespace(model=model, data=data, set_state=state, movable=np.array([0]),
            active_v=np.array([6]), root_v=0, caps=np.array([30.]), hand=np.array([False]),
            kin=SimpleNamespace(robot=SimpleNamespace(model=SimpleNamespace(
                lowerPositionLimit=np.full(8,-1.), upperPositionLimit=np.full(8,1.),
                velocityLimit=np.full(7,2.)))))
        t = np.linspace(0.,1.,31)
        q = np.zeros((31,8)); q[:,6] = 1.; q[:,7] = .01*np.sin(2*np.pi*t)
        reference = dict(q=q, timestamps=t, command_velocity=(.02*np.pi*np.cos(2*np.pi*t))[:,None],
            initial_q=q[0], initial_velocity=np.zeros(1),
            contact_active=np.zeros((31,0),bool), robot_local_points=np.zeros((31,0,3)),
            force_ff=np.zeros((31,0,3)), object_positions=np.zeros((31,3)),
            object_wxyz=np.tile([1.,0.,0.,0.],(31,1)))
        result, audit = refine_commands(scene, reference, [], seconds=2.,continuous_bounds=True)
        self.assertTrue(audit['solved'], audit)
        # F=m(z_ddot+g), independently of the implementation's mass-matrix assembly.
        force = result['command_acceleration'][:,0]+9.81
        self.assertLessEqual(np.max(abs(force)), 27.+1e-4)
        self.assertLess(np.max(abs(result['q'][:,7]-q[:,7])), .06+1e-5)
        np.testing.assert_allclose(result['command_velocity'][0],0.,atol=1e-5)
        dense = np.linspace(t[0], t[-1], 1001)
        continuous = BSpline(result['spline_knots'], result['spline_coefficients'], 3)
        self.assertLessEqual(float(np.max(abs(continuous(dense)))), 1.+1e-5)
        self.assertLessEqual(float(np.max(abs(continuous(dense, 1)))), 2.+1e-5)
        tail = {k: x.copy() for k, x in result.items()}
        tail['q'][16:, 7] += .008*((t[16:]-.5)/.5)**3
        frozen, frozen_audit = refine_commands(scene, tail, [], seconds=3.,
                                               continuous_bounds=True, freeze_prefix_frame=15)
        self.assertTrue(frozen_audit['solved'], frozen_audit)
        for key in ['q', 'command_velocity', 'command_acceleration']:
            np.testing.assert_array_equal(frozen[key][:16], result[key][:16])
        self.assertGreater(abs(frozen['q'][-1, 7]-result['q'][-1, 7]), .001)
        reference.update(contact_active=np.ones((31,1),bool),robot_local_points=np.zeros((31,1,3)),
                         force_ff=np.zeros((31,1,3)))
        constrained,certificate=refine_commands(scene,reference,[SimpleNamespace(body=2)],seconds=2.)
        self.assertTrue(certificate['solved'],certificate)
        self.assertLessEqual(np.max(abs(constrained['q'][:,7]-q[:,7])),.0011)
        self.assertAlmostEqual(certificate['constraint_violation_by_group']['contact_geometry']*100,
                               certificate['solver_scaled_violation_by_group']['contact_geometry'])
        reference['initial_velocity']=np.array([3.])
        rejected,reason=refine_commands(scene,reference,[],seconds=2.)
        self.assertIsNone(rejected)
        self.assertEqual(reason['status'],'initial physical state exceeds speed bound')


if __name__ == '__main__':
    unittest.main()
