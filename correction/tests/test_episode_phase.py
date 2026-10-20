import unittest
import numpy as np
from correction.src.preprocessing.episode_phase import single_phase


class EpisodePhaseTests(unittest.TestCase):
    def test_single_interval_keeps_unassisted_prefix_and_full_clock(self):
        active=np.zeros((946,6),bool); active[96:896]=True
        phase,frames=single_phase(active,np.arange(946)/30)
        self.assertEqual((phase.contact_start,phase.contact_end),(96,895))
        self.assertLess(phase.approach_start,96)
        self.assertLessEqual(phase.hold_end,945)
        self.assertTrue(all(96<f<895 for f in frames))
        phase.validate(946)

    def test_multiple_phases_are_not_silently_discarded(self):
        active=np.zeros((400,6),bool);active[70:120]=True;active[200:300]=True
        with self.assertRaisesRegex(ValueError,'exactly one'):
            single_phase(active,np.arange(400)/30)

    def test_initial_contact_is_not_replaced_by_an_artificial_approach(self):
        active=np.ones((400,6),bool);active[300:]=False
        with self.assertRaisesRegex(ValueError,'recorded time'):
            single_phase(active,np.arange(400)/30)
