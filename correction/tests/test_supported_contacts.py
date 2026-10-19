"""Physical checks for the deliberately reduced supported-EEF experiment."""
import importlib.util
import json
from pathlib import Path
import unittest

import numpy as np

AVAILABLE = importlib.util.find_spec('mujoco') is not None
OUTPUT = Path(__file__).resolve().parents[1]/'outputs/supported_book_local_20261019'


@unittest.skipUnless(AVAILABLE, 'requires optional CPU MuJoCo runtime')
class SupportedContactsTests(unittest.TestCase):
    def test_object_is_free_and_falls_when_contacts_are_removed(self):
        from correction.src.dynamics.supported_contacts import SupportedContacts
        scene = SupportedContacts(np.array([[-.1, -.03, -.13], [.1, .03, .13]]))
        times = np.linspace(0, .1, 11)
        targets = np.tile([[[10.+i, 10., 10.] for i in range(6)]], (len(times), 1, 1))
        positions = np.tile([0., 0., 2.], (len(times), 1))
        rotations = np.tile([1., 0., 0., 0.], (len(times), 1))
        result = scene.rollout(times, targets, positions, rotations, initial_pads=targets[0])
        self.assertEqual(scene.model.neq, 0, 'No equality constraint may attach the object')
        self.assertEqual(result['max_object_applied_force'], 0.)
        self.assertAlmostEqual(result['object_positions'][-1, 2], 2-.5*9.81*.1**2, delta=.0006)
        repeated = scene.rollout(times, targets, positions, rotations, initial_pads=targets[0])
        np.testing.assert_array_equal(repeated['object_positions'], result['object_positions'])

    @unittest.skipUnless((OUTPUT/'report.json').exists(), 'requires generated Tier 1 demo')
    def test_saved_commands_reproduce_success_through_contact_dynamics(self):
        from correction.src.dynamics.supported_contacts import SupportedContacts
        from correction.src.optimization.supported_manipulation import accepted
        with np.load(OUTPUT/'reference.npz', allow_pickle=False) as data:
            ref = {k: data[k].copy() for k in data.files}
        with np.load(OUTPUT/'corrected.npz', allow_pickle=False) as data:
            commands = data['targets'].copy()
            saved_position = data['object_positions'].copy()
        scene = SupportedContacts(ref['bounds'], table=(ref['table_center'], ref['table_radius'], ref['table_height']))
        result = scene.rollout(ref['times'], commands, ref['object_positions'], ref['object_wxyz'],
                               initial_pads=ref['initial_pads'])
        self.assertTrue(accepted(result), {k: v for k, v in result.items() if not isinstance(v, np.ndarray)})
        self.assertLessEqual(result['max_motor_force_n'], scene.force_limit+1e-10)
        np.testing.assert_allclose(result['object_positions'], saved_position, atol=1e-12)

    @unittest.skipUnless((OUTPUT/'report.json').exists(), 'requires generated Tier 1 demo')
    def test_ablation_shares_initial_state_and_does_not_hide_failure(self):
        report = json.loads((OUTPUT/'report.json').read_text())
        initial = []
        for name in ('baseline', 'geometry_only', 'wrench_seed', 'corrected'):
            with np.load(OUTPUT/f'{name}.npz', allow_pickle=False) as data:
                initial.append(np.r_[data['object_positions'][0], data['object_wxyz'][0], data['pad_positions'][0].ravel()])
        for state in initial[1:]:
            np.testing.assert_array_equal(state, initial[0])
        self.assertTrue(report['accepted_supported_contact_demo'])
        self.assertTrue(report['accepted_demonstration_local_contact_reference'])
        self.assertLess(report['max_contact_command_shift_from_human_mm'], 80.)
        self.assertGreater(report['baseline']['max_position_error_mm'], 50.)
        self.assertFalse(report['robot_hand_or_arm_feasibility_validated'])
        self.assertFalse(report['ready_for_training'])


if __name__ == '__main__':
    unittest.main()
