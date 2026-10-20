"""Physical branch isolation, continuous rejection and acceptance guards."""
import unittest
from types import SimpleNamespace
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from correction.src.dynamics.contact_controller import ContactControllerConfig
from correction.src.dynamics.surface_follow import SurfaceFollowConfig, SurfaceFollowController
from correction.src.optimization.guarded_residual import (
    GuardSettings, GuardedResidual, clone_follow, controller_state, adoption_reasons)


def fixture(inertial=False):
    inertia_tag = ('<inertial pos=".01 .02 .03" quat=".980066578 0 .198669331 0" mass="1" diaginertia=".01 .02 .03"/>'
                   if inertial else '')
    model = mujoco.MjModel.from_xml_string('''<mujoco><option timestep=".001"/>
      <worldbody><geom name="floor" type="plane" size="1 1 .1"/>
      <body name="fixture" mocap="true" pos="0 0 2"/>
      <body name="object" pos=".005 0 .05"><freejoint/>'''+inertia_tag+'''
        <geom name="box" type="box" size=".05 .05 .05" mass="1"/></body>
      <body name="finger" pos=".065 0 .05"><joint type="slide" axis="1 0 0"/>
        <geom type="sphere" size=".01" mass=".1"/></body></worldbody></mujoco>''')
    data = mujoco.MjData(model); mujoco.mj_forward(model, data)
    scene = SimpleNamespace(model=model, data=data, book_q=0, book_v=0, book_body=model.body('object').id,
        active_q=np.array([7]), active_v=np.array([6]), caps=np.array([20.]),
        movable=np.array([0]), hand=np.array([False]), object_geoms={model.geom('box').id},
        bounds=np.array([[-.05]*3,[.05]*3]),
        kin=SimpleNamespace(robot=SimpleNamespace(model=SimpleNamespace(
            lowerPositionLimit=np.full(8,-1.),upperPositionLimit=np.full(8,1.)))))
    point = SimpleNamespace(body=model.body('finger').id, local=np.array([-.01,0.,0.]), object_local=np.array([.05,0.,0.]))
    times = np.arange(101)*.001
    ctl = SurfaceFollowController(scene=scene, points=[point],
        normals_local=np.tile([-1.,0.,0.],(101,1,1)), active=np.ones((101,1),bool), times=times,
        config=ContactControllerConfig(patch_distance_gate=True), kp=np.array([20.]),
        follow_config=SurfaceFollowConfig(frames=None, gain=.25),
        nominal_positions=np.zeros((101,1)), nominal_velocities=np.zeros((101,1)),
        planning_poses=dict(object_positions=np.tile([0.,0.,.05],(101,1)), object_wxyz=np.tile([1.,0.,0.,0.],(101,1))))
    plan = dict(times=times, commands=np.zeros((101,1)), velocities=np.zeros((101,1)),
        acceleration=np.zeros((101,1)), force=np.zeros((101,1,3)), kp=np.array([20.]), kd=np.array([2.]),
        desired_p=np.tile([.005,0.,.05],(101,1)), desired_r=np.tile([1.,0.,0.,0.],(101,1)),
        fixture_p=np.tile([0.,0.,2.],(101,1)), fixture_r=np.tile([1.,0.,0.,0.],(101,1)),
        fixture_a=np.zeros((101,3)), fixture_alpha=np.zeros((101,3)), inertia_ff=False, inertia_scale=1.)
    guard = GuardedResidual(scene, ctl, GuardSettings(horizon_s=.08,period_s=.02,impulse_window_s=.02),plan)
    return scene, ctl, guard


class GuardedResidualTests(unittest.TestCase):
    def test_momentum_uses_world_frame_and_actual_com_offset(self):
        scene,ctl,guard=fixture(inertial=True)
        body=scene.book_body
        scene.data.qpos[3:7]=Rotation.from_euler('xyz',[.2,.5,.7]).as_quat(scalar_first=True)
        scene.data.qvel[:6]=[.2,.3,.4,.7,.8,.9]
        mujoco.mj_forward(scene.model,scene.data)
        twist=np.zeros(6)
        mujoco.mj_objectVelocity(scene.model,scene.data,mujoco.mjtObj.mjOBJ_BODY,body,twist,0)
        ri=scene.data.ximat[body].reshape(3,3)
        expected=np.r_[guard.mass*twist[3:],ri@np.diag(scene.model.body_inertia[body])@ri.T@twist[:3]]
        np.testing.assert_allclose(guard.momentum(scene),expected,atol=1e-12)

    def test_prediction_isolated_and_prefix_reproduces_full_controller_memory(self):
        for enabled in (False,True):
            with self.subTest(enabled=enabled):
                scene, ctl, guard = fixture()
                ctl.follow.position[:] = .002; ctl.follow.velocity[:] = .003
                initial_q=scene.data.qpos.copy(); initial_v=scene.data.qvel.copy()
                initial_force=scene.data.qfrc_applied.copy(); memory=controller_state(ctl)
                audit, prefix=guard.predict(0,enabled)
                self.assertTrue(audit['valid'])
                self.assertTrue(np.isfinite(audit['impulse_cost']))
                np.testing.assert_array_equal(scene.data.qpos,initial_q)
                np.testing.assert_array_equal(scene.data.qvel,initial_v)
                np.testing.assert_array_equal(scene.data.qfrc_applied,initial_force)
                for key,value in memory.items():
                    np.testing.assert_array_equal(controller_state(ctl)[key],value)
                ctl.feedback_enabled=enabled; ctl.next_update=0.
                for step in range(guard.period):
                    guard.advance(scene,ctl,step)
                guard.pending=prefix;guard.verify(guard.period)
                self.assertEqual(guard.max_prediction_error,0.)
                np.testing.assert_array_equal(scene.data.qfrc_applied[:6],0.)

    def test_clone_and_disabled_feedback_do_not_reset_accumulated_residual(self):
        scene,ctl,guard=fixture()
        ctl.follow.position[:]=.01;ctl.follow.velocity[:]=.02
        cloned=clone_follow(ctl,scene);cloned.follow.position[:]=.5;cloned.target_offset[:]=.4
        self.assertEqual(ctl.follow.position[0],.01)
        self.assertEqual(ctl.target_offset[0],0.)
        ctl.feedback_enabled=False
        guard.advance(scene,ctl,0)
        self.assertGreater(ctl.follow.position[0],.01)
        self.assertLess(ctl.follow.position[0],.011)
        self.assertLessEqual(abs(ctl.follow.velocity[0]-.02),.001*ctl.follow_config.joint_acceleration_limit_rad_s2+1e-12)

    def test_lower_task_error_alone_cannot_accept_bad_support_or_limits(self):
        nominal=dict(valid=True,task_cost=2.,terminal_cost=2.,impulse_cost=1.,
            peak_torque_fraction=1.2,penetration_m=.001,joint_violation_rad=.002,bound_conflicts=0)
        improved=dict(nominal,task_cost=1.,terminal_cost=1.,impulse_cost=.5)
        self.assertEqual(adoption_reasons(nominal,improved,.01),[])
        for key,value in [('peak_torque_fraction',1.3),('penetration_m',.002),
                          ('joint_violation_rad',.003),('impulse_cost',1.2),('terminal_cost',2.2),('bound_conflicts',1)]:
            with self.subTest(key=key):
                self.assertTrue(adoption_reasons(nominal,dict(improved,**{key:value}),.01))
        self.assertTrue(adoption_reasons(nominal,nominal,.01))
        self.assertTrue(adoption_reasons(dict(valid=False),improved,.01))
        self.assertTrue(adoption_reasons(nominal,dict(valid=False),.01))

    def test_supervision_does_not_change_state_during_candidate_selection(self):
        scene,ctl,guard=fixture()
        before=scene.data.qpos.copy()
        guard.replan(0)
        np.testing.assert_array_equal(scene.data.qpos,before)
        self.assertEqual(guard.evaluations,2)
        for step in range(guard.period):
            guard.advance(scene,ctl,step)
        guard.verify(guard.period)
        self.assertEqual(guard.max_prediction_error,0.)


if __name__=='__main__':
    unittest.main()
