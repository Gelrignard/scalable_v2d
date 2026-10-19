"""Acceptance and export checks on the saved bounded experiment, when present."""
import json
from pathlib import Path
import unittest
import numpy as np

DATA = Path(__file__).resolve().parents[1]/'outputs/bounded_book_20261019_final'


@unittest.skipUnless((DATA/'bounded_report.json').exists(), 'bounded trial not generated')
class BoundedArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = json.loads((DATA/'bounded_report.json').read_text())
        cls.before = np.load(cls.report['input_reference'])
        cls.after = np.load(DATA/'bounded_reference.npz')

    def test_coverage_and_input_invariants(self):
        self.assertTrue(self.report['all_requested_frames_processed'])
        self.assertFalse(self.report['budget_exhausted'])
        for key in ('timestamps', 'object_positions', 'object_wxyz'):
            np.testing.assert_array_equal(self.before[key], self.after[key])
        np.testing.assert_array_equal(self.before['q'][:, :7], self.after['q'][:, :7])
        changed = np.flatnonzero(np.max(np.abs(self.before['q']-self.after['q']), axis=1) > 1e-10)
        np.testing.assert_array_equal(changed, self.report['changed_frames'])

    def test_temporal_bounds_including_seams(self):
        lo, hi = self.report['correction_frames_inclusive']
        sl = slice(lo-2, hi+3)
        times, q = self.after['timestamps'][sl], self.after['q'][sl, 7:]
        velocity = np.diff(q, axis=0)/np.diff(times)[:, None]
        acceleration = np.diff(velocity, axis=0)/((np.diff(times)[:-1]+np.diff(times)[1:])/2)[:, None]
        names = json.loads(str(self.after['metadata_json']))['joint_names']
        for j, name in enumerate(names):
            if any(part in name for part in ('shoulder', 'elbow', 'wrist', 'hand')):
                cap = 200. if 'hand' in name else 80.
                self.assertLessEqual(float(np.abs(acceleration[:, j]).max()), cap+.01, name)
        self.assertLessEqual(self.report['after']['max_joint_speed_ratio'], 1.0001)

    def test_failures_are_not_promoted_to_training(self):
        meta = json.loads(str(self.after['metadata_json']))
        self.assertFalse(meta['ready_for_training'])
        self.assertFalse(meta['accepted_as_feasible_manipulation'])
        self.assertNotIn('probe_penetration_m', self.after.files)
        self.assertNotIn('fk_body_positions', self.after.files)
        for metrics in self.report['rollouts'].values():
            self.assertGreater(metrics['steps'], 0)
            self.assertTrue(metrics['no_external_object_or_root_assistance'])
            if not metrics['initial_state_valid']:
                self.assertFalse(metrics['tracking_pass'])


if __name__ == '__main__':
    unittest.main()
