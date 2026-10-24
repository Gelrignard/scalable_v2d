import unittest
import numpy as np
from correction.src.preprocessing.pilot_phases import infer_phases
from correction.src.optimization.surface_reference import PhaseTiming


class PilotPhaseTests(unittest.TestCase):
    def test_repeated_and_single_hand_without_overlap(self):
        active=np.zeros((300,6),bool)
        active[40:100,:3]=True; active[135:190,3:]=True; active[210:270,:]=True
        phases,audit=infer_phases(active,np.arange(300)/30)
        self.assertEqual([p['sides'] for p in phases],[['left'],['right'],['left','right']])
        for p in phases:PhaseTiming(**p['timing']).validate(300)
        for a,b in zip(phases[:-1],phases[1:]):
            self.assertLess(a['timing']['hold_end'],b['timing']['approach_start'])

    def test_handover_compression_is_explicit(self):
        active=np.zeros((250,6),bool);active[30:110,:3]=True;active[110:220,3:]=True
        phases,_=infer_phases(active,np.arange(250)/30)
        self.assertEqual(len(phases),2)
        self.assertTrue(any(p['compressed_transition'] for p in phases))

    def test_no_contact_does_not_invent_phase(self):
        phases,_=infer_phases(np.zeros((60,6),bool),np.arange(60)/30)
        self.assertEqual(phases,[])
