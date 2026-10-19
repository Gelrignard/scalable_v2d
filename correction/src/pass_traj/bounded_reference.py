"""Export bounded correction candidates without claiming training readiness."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation


def save_bounded(directory, before, metadata, q, indices, targets, anchors, normals,
                 active, diagnostics, report):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    arrays = {k: v.copy() for k, v in before.items()}
    arrays['q'] = q
    arrays['fk_tip_positions'][indices] = diagnostics['tips']
    arrays['tip_targets'][indices] = targets
    arrays['contact_anchors_local'][indices] = anchors
    arrays['contact_normals_local'][indices] = -normals
    arrays['contact_active'][indices] = active
    arrays['contact_blend'][indices] = active.astype(float)
    arrays['tip_error_m'][indices] = np.linalg.norm(diagnostics['tips']-targets, axis=-1)
    rotations = Rotation.from_quat(before['object_wxyz'][indices], scalar_first=True).as_matrix()
    local = np.einsum('tji,tkj->tki', rotations, diagnostics['tips']-before['object_positions'][indices, None])
    arrays['contact_error_m'][indices] = np.linalg.norm(local-(anchors-.002*normals), axis=-1)
    # Old body/probe diagnostics no longer describe the new q. Remove them
    # explicitly; bounded_report.json contains freshly evaluated geometry.
    for key in ('fk_body_positions', 'body_error_m', 'probe_penetration_m'):
        arrays.pop(key, None)
    meta = dict(metadata, method='bounded_contact_correction',
                elapsed_seconds=report['correction_seconds'],
                dynamics_tested=bool(report.get('rollouts')), ready_for_training=False,
                diagnostics_scope='see bounded_report.json; removed stale body/probe arrays',
                accepted_as_feasible_manipulation=False)
    meta['correction_settings'] = report['settings']
    arrays['metadata_json'] = np.array(json.dumps(meta))
    arrays['stop_reason'] = arrays['stop_reason'].astype('<U80')
    arrays['iterations'] = np.zeros(len(q), dtype=int)
    arrays['stop_reason'][indices] = 'bounded correction candidate; inspect audit'
    np.savez_compressed(directory/'bounded_reference.npz', **arrays)
    np.savez_compressed(directory/'bounded_contacts.npz', ids=indices, **diagnostics,
                        planned_targets=targets, planned_anchors=anchors, planned_normals=normals)
    (directory/'bounded_report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    return directory/'bounded_reference.npz'
