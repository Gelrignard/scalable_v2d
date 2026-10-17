from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from correction.demo import pinch_case
from correction.optimizer import CorrectionSettings, PhaseCorrector, validate_dense
from correction.prepare_phase import prepare


class IntegrationTests(unittest.TestCase):
    def test_dense_check_detects_load_missed_by_knots(self):
        times, patches, reference, required = pinch_case(samples=2)
        settings = CorrectionSettings()
        result = PhaseCorrector(times, patches, reference, required, np.zeros(3), settings).solve()
        self.assertTrue(result["after"]["metrics"]["object_wrench_within_tolerance"])
        dense_required = np.tile(required[0], (3, 1))
        dense_required[1, 2] += 10
        dense = validate_dense(result, [0, .5, 1], dense_required, np.zeros(3), settings)
        self.assertFalse(dense["report"]["object_wrench_within_tolerance"])
        self.assertGreater(dense["report"]["max_force_residual_n"], 9.9)

    def test_reference_to_world_target_pipeline(self):
        with tempfile.TemporaryDirectory(prefix="v2d-correction-test-") as folder:
            root = Path(folder)
            times = np.linspace(0, 4, 121)
            poses = np.zeros((len(times), 7))
            poses[:, 2] = .5 * times**2
            poses[:, 3] = 1
            _, patches, _, _ = pinch_case(samples=2)
            points = np.tile([[-.06, .04, 0], [.06, .04, 0]], (len(times), 1, 1)) + poses[:, None, :3]
            np.savez(root / "reference.npz", timestamp_s=times, frame_index=np.arange(len(times)),
                     object_pose_xyz_wxyz=poses, object_visible=np.ones(len(times), dtype=bool),
                     reference_points_world_m=points, point_names=np.array([p.name for p in patches]),
                     world_frame_id=np.array("analytic_world"), object_frame_id=np.array("analytic_box"))
            config = {"contact_mode": "sticking", "world_frame_id": "analytic_world", "object_frame_id": "analytic_box",
                      "provenance": "analytic constant-acceleration integration test",
                      "body_model": {"mass_kg": 1., "com_object_m": [0,0,0],
                                     "inertia_com_object_kg_m2": np.diag([.02,.03,.04]).tolist(),
                                     "gravity_world_m_s2": [0,0,-9.81], "provenance": "analytic test"},
                      "patches": [asdict(p) for p in patches]}
            (root / "config.json").write_text(json.dumps(config, default=lambda value: value.tolist()))
            report = prepare(root / "reference.npz", root / "config.json", root / "phase.npz", 6, 115)
            self.assertEqual(report["dense_frames"], 109)
            process = subprocess.run([sys.executable, "-m", "correction.run_phase", "--phase", str(root/"phase.npz"),
                                      "--config", str(root/"config.json"), "--output", str(root/"out")],
                                     capture_output=True, text=True, timeout=30)
            self.assertEqual(process.returncode, 0, process.stderr)
            saved = json.loads((root / "out/report.json").read_text())
            self.assertTrue(saved["dense_validation"]["object_wrench_within_tolerance"])
            with np.load(root / "out/correction.npz") as output, np.load(root/"phase.npz") as phase:
                self.assertEqual(output["corrected_contact_targets_world_m"].shape, (109, 2, 3))
                np.testing.assert_array_equal(output["dense_object_pose_xyz_wxyz"], phase["dense_object_pose_xyz_wxyz"])
                relative = output["corrected_contact_targets_world_m"] - output["dense_object_pose_xyz_wxyz"][:, None, :3]
                np.testing.assert_allclose(relative, np.tile(output["after_contact_points_object_m"], (109,1,1)), atol=1e-10)
            with self.assertRaisesRegex(ValueError, "boundaries"):
                prepare(root / "reference.npz", root / "config.json", root / "invalid.npz", 0, 115)
            config["world_frame_id"] = "another_world"
            (root / "config.json").write_text(json.dumps(config, default=lambda value: value.tolist()))
            with self.assertRaisesRegex(ValueError, "world_frame_id"):
                prepare(root / "reference.npz", root / "config.json", root / "invalid.npz", 6, 115)


if __name__ == "__main__":
    unittest.main()
