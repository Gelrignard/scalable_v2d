import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from v2d_retarget.paths import MODULE, WORKSPACE
from v2d_retarget.dataprocessing import load_demo, load_npz, save_npz
from v2d_retarget.core.contacts import latch_contacts
from v2d_retarget.core.targets import build_body_targets
from v2d_retarget.core.robot import G1Kinematics
from v2d_retarget.core.constraints import OutputVelocityLimit, HullAvoidanceTask
from v2d_retarget.core.solver import solve_baseline
from v2d_retarget.core.surface import ObjectSurface
import pink
import pinocchio as pin
from pink.tasks import PostureTask
import trimesh


class ContactTests(unittest.TestCase):
    def test_hysteresis_anchor_and_release(self):
        cfg = json.loads((MODULE / "configs/contact_aware.json").read_text())
        cfg.update(contact_enter_duration_s=0.02, contact_release_duration_s=0.02)
        ts = np.arange(15) * 0.01
        local = np.zeros((15, 1, 3))
        local[:, 0, 0] = np.arange(15) * 0.001
        nearest = local.copy()
        normals = np.zeros_like(local)
        normals[:, :, 2] = 1
        d = np.full((15, 1), 0.02)
        d[6] = 0.1
        d[10:] = 0.1
        active, blend, anchors, _, _ = latch_contacts(
            local, d, nearest, normals, ts, cfg
        )
        self.assertFalse(active[0, 0])
        self.assertTrue(active[4, 0])
        self.assertTrue(active[6, 0], "One noisy sample should not break contact")
        np.testing.assert_array_equal(
            anchors[4:10, 0], np.broadcast_to(anchors[4, 0], (6, 3))
        )
        self.assertFalse(active[-1, 0])
        self.assertLess(blend[-1, 0], blend[9, 0])

    def test_no_proximity_means_no_contact(self):
        cfg = json.loads((MODULE / "configs/contact_aware.json").read_text())
        p = np.zeros((20, 2, 3))
        ts = np.arange(20) * 0.01
        active, *_ = latch_contacts(p, np.ones((20, 2)), p, p, ts, cfg)
        self.assertFalse(active.any())


class KinematicTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.motion = load_demo().subset(stop=3)

    def test_target_frame_conversion_matches_nvidia(self):
        k = G1Kinematics()
        body = build_body_targets(self.motion, k.config)
        w = np.asarray(k.config.r_world)
        for t in range(3):
            source_q = Rotation.from_matrix(
                w.T
                @ Rotation.from_quat(
                    self.motion.rotations_wxyz[t], scalar_first=True
                ).as_matrix()
            ).as_quat(scalar_first=True)
            k.set_frame_tasks_target(self.motion.positions[t] @ w, source_q)
            k.align_foot_task_targets_to_ground(0.0)
            for j, task in enumerate(k.frame_tasks.values()):
                np.testing.assert_allclose(
                    body.positions[t, j],
                    task.transform_target_to_world.translation,
                    atol=1e-10,
                )
                np.testing.assert_allclose(
                    body.rotations[t, j],
                    task.transform_target_to_world.rotation,
                    atol=1e-10,
                )

    def test_baseline_regression(self):
        path = WORKSPACE / "outputs/local_ik.npz"
        if not path.exists():
            self.skipTest("Original audit artifact is not installed")
        with np.load(path, allow_pickle=False) as data:
            expected = data["q"][:3]
        result = solve_baseline(G1Kinematics(), self.motion, progress=lambda _: None)
        np.testing.assert_allclose(result.q, expected, atol=1e-7)

    def test_velocity_bound_survives_many_internal_iterations(self):
        k = G1Kinematics()
        previous = k.robot.q0.copy()
        target = previous.copy()
        target[7] = 0.7
        task = PostureTask(cost=1.0)
        task.set_target(target)
        frame_dt = 0.005
        limit = OutputVelocityLimit(k.robot.model, previous, frame_dt, 0.9)
        for _ in range(30):
            v = pink.solve_ik(
                k.configuration,
                [task],
                k.dt,
                solver="daqp",
                limits=k.configuration_limits + [limit],
            )
            k.configuration.integrate_inplace(v, k.dt)
        bound = k.robot.model.velocityLimit[6:] * frame_dt * 0.9
        self.assertTrue(
            np.all(np.abs(k.configuration.q[7:] - previous[7:]) <= bound + 1e-8)
        )
        self.assertGreater(
            k.configuration.q[7],
            0.01,
            "Solver must actually move toward the distant target",
        )

    def test_canonical_roundtrip_and_bad_timing(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "motion.npz"
            save_npz(self.motion, path)
            read = load_npz(path)
            np.testing.assert_array_equal(read.positions, self.motion.positions)
            np.testing.assert_array_equal(read.object_wxyz, self.motion.object_wxyz)
            from dataclasses import replace

            with self.assertRaisesRegex(ValueError, "increase"):
                replace(read, timestamps=np.zeros(3))

    def test_collision_jacobian_by_finite_difference(self):
        k = G1Kinematics()
        q = k.robot.q0.copy()
        with tempfile.TemporaryDirectory() as td:
            mesh = trimesh.creation.box(extents=[1.7, 1.3, 2.1])
            mesh.apply_translation([0.13, 0.07, 0.19])
            path = Path(td) / "box.ply"
            mesh.export(path)
            task = HullAvoidanceTask(k, ObjectSurface(path), cost=1.0)
            k.configuration.update(q)
            analytic = task.compute_jacobian(k.configuration).copy()
            direction = np.random.default_rng(4).normal(size=k.robot.model.nv)
            eps = 1e-7
            k.configuration.update(pin.integrate(k.robot.model, q, eps * direction))
            plus = task.compute_error(k.configuration).copy()
            k.configuration.update(pin.integrate(k.robot.model, q, -eps * direction))
            minus = task.compute_error(k.configuration).copy()
            np.testing.assert_allclose(
                (plus - minus) / (2 * eps), analytic @ direction, atol=2e-6
            )

    def test_contact_broadphase_never_drops_near_points(self):
        with tempfile.TemporaryDirectory() as td:
            mesh = trimesh.creation.icosphere(subdivisions=1, radius=0.2)
            path = Path(td) / "sphere.ply"
            mesh.export(path)
            surface = ObjectSurface(path)
            points = np.random.default_rng(0).normal(size=(100, 3)) * 0.3
            _, exact, _ = surface.nearest(points)
            _, bounded, _ = surface.contact_candidates(points, 0.055)
            np.testing.assert_array_equal(exact <= 0.055, bounded <= 0.055)
            np.testing.assert_allclose(exact[exact <= 0.055], bounded[exact <= 0.055])

    def test_training_export_preserves_selected_frames_and_named_fk(self):
        from v2d_retarget.core.contacts import ContactTargets
        from v2d_retarget.core.solver import SolveResult
        from v2d_retarget.output.motion_v1 import export_motion_v1
        from robotic_grounding.motion_schema import load_motion_data_parquet

        motion = (
            load_demo().subset(start=10, stop=30).subset(start=2, stop=12, stride=3)
        )
        k = G1Kinematics()
        n = len(motion.timestamps)
        q = np.repeat(k.robot.q0[None], n, axis=0)
        solved = SolveResult(q, np.zeros(n, int), ["test"] * n, 0.0)
        xyz = np.zeros((n, 6, 3))
        scalar = np.zeros((n, 6))
        active = np.zeros((n, 6), bool)
        contacts = ContactTargets(
            xyz, xyz, active, scalar, xyz, xyz, scalar, scalar, 0.002
        )
        with tempfile.TemporaryDirectory() as td:
            export_motion_v1(
                td,
                motion,
                k,
                solved,
                contacts,
                {},
                dict(body_error_m=np.zeros((n, 10))),
            )
            path = next(Path(td).rglob("data.parquet"))
            md = load_motion_data_parquet(path, device=None)
            np.testing.assert_allclose(md.robot_joint_positions, q[:, 7:])
            np.testing.assert_allclose(
                md.object_root_position, motion.object_positions, atol=1e-7
            )
            self.assertAlmostEqual(
                md.fps, 1 / float(np.diff(motion.timestamps)[0]), places=4
            )
            self.assertEqual(len(md.robot_root_position), n)
            if len(md.object_articulation):
                self.assertEqual(len(md.object_articulation), n)
            for side_idx, side in enumerate(("left", "right")):
                for name in [f"{side}_hand_index_1_link", f"{side}_index_tip"]:
                    j = md.hand_frame_names[side_idx].index(name)
                    np.testing.assert_allclose(
                        md.hand_frames_w[side_idx][0, j, :3],
                        k.poses(q[0], [name])[0].translation,
                        atol=1e-7,
                    )
                self.assertEqual(len(md.hand_finger_joint_names[side_idx]), 7)


if __name__ == "__main__":
    unittest.main()
