"""Inspect commanded/actual self-collisions throughout the inactive regrasp."""
import argparse
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from correction.src.preprocessing.strategy_bundle import read_npz, load_strategy_scene
from correction.src.pass_traj.experiment_record import snapshot_sources, process_cost


def run(directory):
    start = perf_counter(); cpu = process_cost()['cpu_seconds']
    report = json.loads((directory/'report.json').read_text())
    ref, actual = read_npz(directory/'reference.npz'), read_npz(directory/'mechanics.npz')
    _, _, _, scene = load_strategy_scene(report['input_directory'])
    rows = []
    for frame in range(525, 701):
        for label, q, pos, quat in [('command', ref['q'], ref['object_positions'], ref['object_wxyz']),
                                    ('actual', actual['q'], actual['object_positions'], actual['object_wxyz'])]:
            scene.set_state(q[frame], pos[frame], quat[frame])
            contacts = [dict(pair=r['pair'], distance_m=r['distance']) for r in scene.contacts()
                        if r['kind']=='self']
            rows.append(dict(frame=frame, kind=label, contacts=contacts))
    audit = dict(scope='Geometry reconstructed from saved poses; no force inferred from reconstructed state',
        rows=rows, provenance=snapshot_sources(directory/'regrasp_audit_sources', [Path(__file__)]),
        wall_seconds=perf_counter()-start, cost=process_cost())
    audit['cost']['cpu_seconds'] -= cpu
    (directory/'regrasp_path_audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    worst = sorted([(c['distance_m'],r['frame'],r['kind'],c['pair']) for r in rows for c in r['contacts']])
    print(json.dumps(dict(worst=worst[:10], wall_seconds=audit['wall_seconds']), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    run(parser.parse_args().directory)
