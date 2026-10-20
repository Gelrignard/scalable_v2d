"""Invariants for physical limits, object-frame evidence and wrench screening."""
import unittest
from types import SimpleNamespace
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from correction.src.dynamics.execution_contract import ContactExecutionContract
from correction.src.preprocessing.interaction_evidence import object_coordinates, regions
from correction.src.optimization.wrench_screen import screen_wrench
from correction.src.optimization.patch_geometry import triangle_point, triangle_coordinates, normal_cone_gap
from correction.src.optimization.patch_geometry import solve_patch_geometry, matches_selected_patch
from correction.src.optimization.surface_hypotheses import SurfacePoint
from correction.src.dynamics.robot_scene import RobotScene


class ContactStrategyTests(unittest.TestCase):
    def test_patch_neighbours_reject_remote_and_opposite_faces(self):
        scene=SimpleNamespace(object_geoms={1,2},
            model=SimpleNamespace(geom_bodyid=np.array([3,4,4])),
            data=SimpleNamespace(xpos=np.zeros((5,3)),xmat=np.tile(np.eye(3).ravel(),(5,1))))
        point=SimpleNamespace(body=3,local=np.array([.1,0,0]),object_local=np.array([.1,0,0]),
                              object_outward=np.array([-1.,0,0]))
        row=dict(closest_points=np.array([[.1,.005,0],[.101,.005,0]]),
                 normal_1_to_2=np.array([1.,0,0]))
        self.assertTrue(matches_selected_patch(scene,[point],0,2,row,np.zeros(3),np.eye(3)))
        reverse=dict(closest_points=row['closest_points'][::-1],normal_1_to_2=-row['normal_1_to_2'])
        self.assertTrue(matches_selected_patch(scene,[point],2,0,reverse,np.zeros(3),np.eye(3)))
        far=dict(row,closest_points=row['closest_points']+[0,.04,0])
        self.assertFalse(matches_selected_patch(scene,[point],0,2,far,np.zeros(3),np.eye(3)))
        wrong=dict(row,normal_1_to_2=-row['normal_1_to_2'])
        self.assertFalse(matches_selected_patch(scene,[point],0,2,wrong,np.zeros(3),np.eye(3)))

    def test_clearance_optimizer_retains_world_translated_contact_target(self):
        vertices=' '.join(str(x) for p in [(x,y,z) for x in (-1,1) for y in (-1,1) for z in (-1,1)] for x in p)
        xml=f'''<mujoco><option gravity="0 0 0"/><asset>
            <mesh name="pad" vertex="{vertices}" scale=".01 .01 .01"/>
            <mesh name="box" vertex="{vertices}" scale=".1 .1 .1"/></asset>
            <worldbody><body name="robot">
              <joint type="slide" axis="1 0 0"/><joint type="slide" axis="0 1 0"/><joint type="slide" axis="0 0 1"/>
              <geom name="pad_geom" type="mesh" mesh="pad" mass=".1" margin=".001"/>
            </body><body name="object"><freejoint/>
              <geom name="object_geom" type="mesh" mesh="box" mass="1" margin=".001"/>
            </body></worldbody></mujoco>'''
        model=mujoco.MjModel.from_xml_string(xml); data=mujoco.MjData(model)
        def set_state(q,p,r):
            data.qpos[:3]=q[7:];data.qpos[3:]=np.r_[p,r];data.qvel[:]=0
            mujoco.mj_forward(model,data)
        scene=SimpleNamespace(model=model,data=data,movable=np.arange(3),active_v=np.arange(3),
            vadr=np.arange(3),hand=np.zeros(3,bool),floor_geom=-1,environment=set(),
            object_geoms={model.geom('object_geom').id},set_state=set_state,
            kin=SimpleNamespace(robot=SimpleNamespace(model=SimpleNamespace(
                lowerPositionLimit=np.full(10,-2.),upperPositionLimit=np.full(10,2.)))))
        scene.contacts=lambda movable=None:RobotScene.contacts(scene,movable)
        point=SurfacePoint(model.body('robot').id,model.geom('pad_geom').id,
            np.array([-.01,0,0]),np.array([-1.,0,0]),np.array([.1,0,0]),np.array([1.,0,0]),
            model.geom('object_geom').id,
            np.array([[-.01,-.01,-.01],[-.01,.01,-.01],[-.01,.01,.01]]),
            np.array([[.1,-.1,-.1],[.1,.1,-.1],[.1,.1,.1]]))
        q=np.zeros(10);q[6]=1.;q[7:]=[.5101,.2,.3]
        result,_,audit=solve_patch_geometry(scene,[point],q,np.array([.4,.2,.3]),[1.,0,0,0],
            non_contact_clearance_m=.002,seconds=3.)
        self.assertLess(audit['max_gap_m'],.0002)
        np.testing.assert_allclose(result[7:],q[7:],atol=.001)

    def test_surface_chart_stays_on_triangle_and_derivative_matches(self):
        triangle = np.array([[.1, .2, .3], [.3, .2, .4], [.1, .5, .2]])
        uv = np.array([.6, .3])
        point, jac = triangle_point(triangle, uv)
        np.testing.assert_allclose(triangle_coordinates(triangle, point), uv, atol=1e-12)
        for i in range(2):
            delta = np.eye(2)[i]*1e-6
            numerical = (triangle_point(triangle, uv+delta)[0]-triangle_point(triangle, uv-delta)[0])/2e-6
            np.testing.assert_allclose(jac[:, i], numerical, atol=1e-10)
        for uv in ([0,0], [1,0], [1,1], [.8,.4]):
            point, _ = triangle_point(triangle, uv)
            weights = np.linalg.lstsq(np.vstack((triangle.T, np.ones(3))), np.r_[point, 1.], rcond=None)[0]
            self.assertGreaterEqual(weights.min(), -1e-10)

    def test_corner_contact_accepts_normal_cone_not_only_one_face_normal(self):
        vertices = np.array([[x,y,z] for x in [-1,1] for y in [-1,1] for z in [-1,1]], float)
        direction = np.array([1., 1., 1.])/np.sqrt(3)
        gap, _ = normal_cone_gap(vertices, np.ones(3), direction)
        self.assertAlmostEqual(gap, 0.)
        gap, _ = normal_cone_gap(vertices, np.array([1., 0., 0.]), direction)
        self.assertGreater(gap, 1.)

    def test_force_contract_rejects_wider_plan_cone_and_is_frame_invariant(self):
        contract = ContactExecutionContract()
        force, inward = np.array([[10., 5., 0.], [5., 1., 0.]]), np.array([[1., 0, 0], [1., 0, 0]])
        a = contract.force_audit(force, inward)
        self.assertEqual(a['nontrivial_cone_violations'], 1)
        rotation = Rotation.from_rotvec([.4, -.2, .7])
        b = contract.force_audit(rotation.apply(force), rotation.apply(inward))
        self.assertEqual(a['violation_samples'], b['violation_samples'])
        self.assertAlmostEqual(a['max_cone_excess_n'], b['max_cone_excess_n'])
        with self.assertRaises(ValueError): ContactExecutionContract(friction=.7)

    def test_inactive_force_and_missing_normals_do_not_pass(self):
        contract = ContactExecutionContract()
        a = contract.force_audit(np.array([[1., 0, 0], [0., 0, 0]]),
                                np.zeros((2, 3)), np.array([False, True]))
        self.assertFalse(a['accepted'])
        self.assertEqual(a['inactive_force_samples'], 1)
        self.assertEqual(a['invalid_normal_samples'], 1)

    def test_object_evidence_invariant_to_world_rigid_transform(self):
        rng = np.random.default_rng(42)
        p = rng.normal(size=(5, 3))
        r = Rotation.from_rotvec(rng.normal(size=(5, 3)))
        points = rng.normal(size=(5, 6, 3))
        a = object_coordinates(points, p, r.as_quat(scalar_first=True))
        transform = Rotation.from_rotvec([.3, .2, -.4]); shift = np.array([10., -2., 4.])
        b = object_coordinates(transform.apply(points.reshape(-1, 3)).reshape(points.shape)+shift,
                               transform.apply(p)+shift, (transform*r).as_quat(scalar_first=True))
        np.testing.assert_allclose(a, b, atol=1e-12)

    def test_contact_regions_do_not_average_opposing_normals_to_zero(self):
        normal = np.tile([1., 0, 0], (10, 6, 1)); normal[7:] *= -1
        raw = dict(timestamps=np.arange(10), contact_active=np.ones((10, 6), bool),
                   contact_anchors_local=np.zeros((10, 6, 3)), contact_normals_local=normal)
        result = regions(raw, 0, 9)
        self.assertEqual(len(result), 2)
        np.testing.assert_allclose(result[0].outward_object, [1., 0, 0])
        self.assertAlmostEqual(result[0].normal_coherence, .7)

    def test_two_friction_contacts_cannot_hold_weight_but_four_can(self):
        points = np.array([[-.1, -.03, 0], [-.1, .03, 0], [.1, -.03, 0], [.1, .03, 0]])
        inward = np.array([[1., 0, 0], [1., 0, 0], [-1., 0, 0], [-1., 0, 0]])
        wanted = np.array([0., 0, 19.62, 0, 0, 0])
        full = screen_wrench(points, inward, np.zeros(3), wanted, np.zeros((4, 3, 1)), [0.], [10.])
        self.assertTrue(full['feasible'], full)
        fewer = screen_wrench(points[[0, 2]], inward[[0, 2]], np.zeros(3), wanted,
                              np.zeros((2, 3, 1)), [0.], [10.])
        self.assertFalse(fewer['feasible'])
        self.assertGreater(fewer['force_residual_n'], 5.)

    def test_actuator_limits_prevent_an_algebraically_feasible_hold(self):
        points = np.array([[-.1, 0, 0], [.1, 0, 0]])
        inward = np.array([[1., 0, 0], [-1., 0, 0]])
        jac = np.zeros((2, 3, 2)); jac[0, 2, 0] = 1.; jac[1, 2, 1] = 1.
        wanted = np.array([0., 0, 4., 0, 0, 0])
        unconstrained = screen_wrench(points, inward, np.zeros(3), wanted, jac, [0., 0.], [3., 3.])
        constrained = screen_wrench(points, inward, np.zeros(3), wanted, jac, [0., 0.], [.2, .2])
        self.assertTrue(unconstrained['feasible'], unconstrained)
        self.assertFalse(constrained['feasible'])
        self.assertLessEqual(constrained['max_torque_fraction'], 1.+1e-6)


if __name__ == '__main__': unittest.main()
