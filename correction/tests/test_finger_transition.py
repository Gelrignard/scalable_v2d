"""An independent two-axis physical clearance fixture for the free-path SQP."""
from types import SimpleNamespace
import unittest
import mujoco
import numpy as np
from scipy.interpolate import CubicHermiteSpline
from correction.src.optimization.finger_transition import refine_free_path
from correction.src.optimization.command_smoothing import spline_operators
from correction.src.optimization.spline_clearance import clearance_rows


class FreePathTests(unittest.TestCase):
    def test_collision_repair_preserves_endpoints_and_rejects_loaded_interval(self):
        model=mujoco.MjModel.from_xml_string('''<mujoco><option gravity="0 0 0"/>
          <worldbody><geom name="obstacle" type="sphere" size=".035"/>
            <body><joint type="slide" axis="1 0 0" range="-.3 .3"/>
              <joint type="slide" axis="0 1 0" range="-.3 .3"/>
              <geom name="robot_pad" type="sphere" size=".015" mass=".1" margin=".001"/>
            </body></worldbody></mujoco>''')
        data=mujoco.MjData(model)
        def set_state(q, position, rotation):
            data.qpos[:]=q[7:];data.qvel[:]=0.;mujoco.mj_forward(model,data)
        scene=SimpleNamespace(model=model,data=data,movable=np.arange(2),active_v=np.arange(2),
            hand=np.zeros(2,bool),set_state=set_state,
            kin=SimpleNamespace(robot=SimpleNamespace(model=SimpleNamespace(
                lowerPositionLimit=np.full(9,-.3),upperPositionLimit=np.full(9,.3),
                velocityLimit=np.ones(8)))))
        t=np.linspace(0,2,41);q=np.zeros((41,9));q[:,6]=1.
        q[:,7]=np.linspace(-.12,.12,41);q[:,8]=.025*np.sin(np.pi*t/2)
        ref=dict(q=q,timestamps=t,object_positions=np.zeros((41,3)),
            object_wxyz=np.tile([1.,0.,0.,0.],(41,1)),contact_active=np.zeros((41,1),bool))
        static = {k: x.copy() for k, x in ref.items()}
        static['q'][:, 7:] = [.04, .01]
        static['command_velocity'] = np.zeros((41, 2))
        knots, basis, _, _ = spline_operators(t)
        obstacle = dict(frames=[3, 30], object_positions=ref['object_positions'],
                        object_wxyz=ref['object_wxyz'])
        matrix, low, certificate = clearance_rows(scene, static, knots, obstacle, frequency=30)
        coefficients = np.tile([.04, .01], (basis.shape[1], 1))
        self.assertGreater(certificate['rows'], 0)
        # Analytic sphere clearance: center distance minus summed radii.
        np.testing.assert_allclose(low-matrix@coefficients.ravel(),
                                   .002+.05-np.hypot(.04, .01), atol=1e-9)
        outward = np.tile(np.array([.04, .01])/np.hypot(.04, .01), (basis.shape[1], 1))
        np.testing.assert_allclose(matrix@outward.ravel(), 1., atol=1e-9)
        result,audit=refine_free_path(scene,ref,start=0,end=40,seconds=10.,dense_pairs=True)
        self.assertTrue(audit['iterations'])
        np.testing.assert_allclose(result['q'][[0,-1],7:],q[[0,-1],7:],atol=1e-6)
        spline=CubicHermiteSpline(t,result['q'][:,7:],result['command_velocity'])
        # Analytic sphere distance, independent of the optimization Jacobian.
        radius=np.linalg.norm(spline(np.linspace(0,2,1001)),axis=1)
        self.assertGreaterEqual(radius.min(),.05-1e-5)
        self.assertGreater(np.max(result['q'][:,8]),.045)
        ref['contact_active'][20]=True
        with self.assertRaises(ValueError):
            refine_free_path(scene,ref,start=0,end=40)


if __name__=='__main__':
    unittest.main()
