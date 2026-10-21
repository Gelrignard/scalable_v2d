"""Planning/scoring reference separation across saved episode formats."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from correction.src.preprocessing.staged_planning import resolve_planning


class StagedPlanningTests(unittest.TestCase):
    def test_unknown_source_cannot_silently_use_scoring_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'report.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'Unknown planning contract'):
                resolve_planning(root, {})

    def test_transfer_reference_must_match_saved_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); profile = root/'profile'; profile.mkdir()
            raw = root/'input'; (raw/'retargeted').mkdir(parents=True)
            ref = dict(timestamps=np.arange(5)/30, object_positions=np.zeros((5,3)),
                       object_wxyz=np.tile([1.,0.,0.,0.],(5,1)))
            np.savez(raw/'retargeted/contact_aware.npz', **ref)
            (profile/'report.json').write_text(json.dumps(dict(input_directory=str(raw))))
            (root/'report.json').write_text(json.dumps(dict(input_directory=str(profile),
                scope='Single-interval transfer of staged bin prototype',
                provenance=dict(source_snapshot=[dict(path='correction/scripts/transfer_episode.py')]))))
            poses, files, contract = resolve_planning(root, ref)
            np.testing.assert_array_equal(poses['object_positions'], ref['object_positions'])
            poses['object_positions'][0,0] = 1.
            self.assertEqual(ref['object_positions'][0,0], 0.)
            ref['object_positions'][1,0] = .1
            with self.assertRaises(AssertionError):
                resolve_planning(root, ref)

    def test_sequence_shift_comes_from_verified_ancestor(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root/'source'; source.mkdir()
            ancestor = root/'sequence'; ancestor.mkdir()
            ref = dict(q=np.zeros((10,8)), timestamps=np.arange(10)/30,
                object_positions=np.zeros((10,3)), object_wxyz=np.tile([1.,0.,0.,0.],(10,1)),
                object_local_points=np.zeros((10,1,3)), contact_active=np.zeros((10,1),bool))
            np.savez(ancestor/'reference.npz', **ref)
            (source/'report.json').write_text(json.dumps(dict(source=str(ancestor))))
            (ancestor/'report.json').write_text(json.dumps(dict(
                command=dict(second_planning_translation_m=[.1,0.,0.]),
                phase_contacts=[{},dict(timing=dict(approach_start=3,force_full=5,contact_end=8))])))
            poses, _, _ = resolve_planning(source, ref, ancestor)
            self.assertEqual(poses['object_positions'][3,0], .1)
            self.assertEqual(poses['object_positions'][0,0], 0.)
            self.assertEqual(ref['object_positions'][3,0], 0.)


if __name__ == '__main__':
    unittest.main()
