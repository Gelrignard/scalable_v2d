"""Resolve saved planning poses without mistaking scoring poses for a plan."""
import json
from pathlib import Path

import numpy as np

from .strategy_bundle import read_npz
from ..optimization.surface_reference import smoothstep


def inherited_planning_poses(reference, sequence):
    """The saved two-grasp planner shifts only its second planning path."""
    poses = {key: reference[key].copy() for key in ('object_positions', 'object_wxyz')}
    shift = sequence['command']['second_planning_translation_m']
    if shift is not None:
        timing = sequence['phase_contacts'][1]['timing']
        frames = np.arange(len(reference['q']))
        weight = 1-smoothstep((frames-timing['force_full'])/(timing['contact_end']-timing['force_full']))
        weight[frames < timing['approach_start']] = 0.
        if np.any(reference['contact_active'][timing['approach_start']-1:timing['approach_start']+1]):
            raise ValueError('Planning chart switch must occur during an inactive interval')
        poses['object_positions'] += weight[:, None]*np.asarray(shift)
    return poses


def resolve_planning(source, reference, sequence=None):
    """Accept either verified bin ancestry or an explicitly recognized transfer.

    Unknown artifacts must supply their actual planning poses, not silently
    fall back to the intended target or a previous simulated outcome.
    """
    source = Path(source).resolve()
    old = json.loads((source/'report.json').read_text())
    if sequence is None:
        paths = [p['path'] for p in old.get('provenance', {}).get('source_snapshot', [])]
        if (old.get('scope') != 'Single-interval transfer of staged bin prototype' or
                'correction/scripts/transfer_episode.py' not in paths):
            raise ValueError('Unknown planning contract; provide --planning-sequence for a saved two-grasp sequence')
        # transfer_episode passes raw object poses directly to its geometry and
        # force stages. Verify those source observations; never use mechanics.npz.
        profile = Path(old['input_directory'])/'report.json'
        input_directory = Path(json.loads(profile.read_text())['input_directory'])
        raw_path = input_directory/'retargeted/contact_aware.npz'
        raw = read_npz(raw_path)
        for key in ('timestamps', 'object_positions', 'object_wxyz'):
            np.testing.assert_array_equal(reference[key], raw[key])
        return ({key: reference[key].copy() for key in ('object_positions', 'object_wxyz')},
                [source/'report.json', profile, raw_path],
                'Verified single-interval transfer: planning and intended object poses coincide')
    sequence = Path(sequence).resolve()
    ancestor, chain = source, []
    for _ in range(12):
        path = ancestor/'report.json'; chain.append(path)
        saved = json.loads(path.read_text())
        if ancestor == sequence:
            break
        ancestor = Path(saved['source']).resolve()
    else:
        raise ValueError('Planning sequence is not in source ancestry')
    sequence_ref = read_npz(sequence/'reference.npz')
    for key in ('timestamps', 'object_positions', 'object_wxyz', 'object_local_points', 'contact_active'):
        np.testing.assert_array_equal(reference[key], sequence_ref[key])
    return (inherited_planning_poses(reference, saved), [*chain, sequence/'reference.npz'],
            'Verified two-grasp sequence ancestry, including saved second planning translation')
