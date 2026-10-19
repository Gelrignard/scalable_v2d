"""Demonstration-local contact repair followed by bounded shooting correction.

The shooting state is a smooth SE(3) bias shared by six contact commands. The
search direction is an approximate inverse tracking response, not an analytic
gradient of MuJoCo. Every accepted update must reduce a fresh physical rollout
objective; a backtracking search rejects harmful updates.
"""
from time import perf_counter

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation

from correction.src.dynamics.models import RigidBody
from correction.src.dynamics.object_wrench import object_wrench
from correction.src.dynamics.sparse_forces import allocate_forces
from correction.src.dynamics.supported_contacts import SupportedContacts


def metrics(result, times):
    return dict(max_position_error_mm=1000*float(result['max_position_error_m']),
                rms_position_error_mm=1000*float(np.sqrt(np.mean(result['position_error']**2))),
                max_rotation_error_deg=float(np.rad2deg(result['max_rotation_error_rad'])),
                rms_rotation_error_deg=float(np.rad2deg(np.sqrt(np.mean(result['rotation_error']**2)))),
                max_penetration_mm=1000*float(result['max_penetration_m']),
                initial_penetration_mm=1000*float(result['initial_max_penetration_m']),
                max_motor_force_n=float(result['max_motor_force_n']),
                max_object_applied_force=float(result['max_object_applied_force']),
                saturation_steps=int(result['saturation_steps']),
                max_command_speed_m_s=float(result['max_command_speed_m_s']),
                max_command_acceleration_m_s2=float(result['max_command_acceleration_m_s2']))


def accepted(result):
    return bool(result['max_position_error_m'] < .01 and
                result['max_rotation_error_rad'] < np.deg2rad(5) and
                result['max_penetration_m'] < .001 and
                result['max_command_speed_m_s'] < 1. and
                result['max_command_acceleration_m_s2'] < 10. and
                result['max_object_applied_force'] == 0. and
                result['initial_max_penetration_m'] < 1e-6)


def objective(result):
    return float(np.mean((result['position_error']/.01)**2 +
                         (result['rotation_error']/np.deg2rad(5))**2))


def limit_norm(x, limit):
    return x*np.minimum(1., limit/np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12))


def run(reference, *, iterations=8, budget=45., mass=1., friction=.6):
    started = perf_counter()
    ref = reference
    count = len(ref.times)
    rotations = Rotation.from_quat(ref.quaternions[:, [1, 2, 3, 0]])
    matrices = rotations.as_matrix()
    scene = SupportedContacts(ref.bounds, mass=mass, friction=friction, table=ref.table)
    size = np.ptp(ref.bounds, axis=0)
    body = RigidBody(mass, ref.bounds.mean(axis=0), np.diag(mass*(sum(size**2)-size**2)/12),
                     np.array([0., 0., -9.81]), 'Assumed uniform rigid box; not identified challenge parameters')
    dynamics = object_wrench(ref.all_times, ref.all_poses, np.ones(len(ref.all_times), bool), body, window=11)
    required = dynamics['required_wrench_object'][ref.ids]
    local_human = np.einsum('tji,tkj->tki', matrices, ref.human_tips-ref.positions[:, None])
    median = np.median(local_human, axis=0)
    axis = int(np.argmin(size))
    candidates = []
    for left in range(2):
        for right in range(2):
            points = np.clip(median, ref.bounds[0]+.012, ref.bounds[1]-.012)
            normals = np.zeros((6, 3))
            for hand, sign in enumerate((left, right)):
                for finger in range(3):
                    j = hand*3+finger
                    side = sign if finger == 0 else 1-sign
                    points[j, axis] = ref.bounds[side, axis]
                    normals[j, axis] = 1-2*side
            plan = allocate_forces(np.tile(points, (count, 1, 1)), np.tile(normals, (count, 1, 1)),
                                   np.ones((count, 6), bool), required, ref.times, body.com_object_m,
                                   cap=20., mu=friction, time_limit=3.)
            evidence = float(np.mean(np.sum((points-median)**2, axis=-1))/.02**2)
            wrench_error = float(np.mean(np.sum((plan.residual/np.array([10, 10, 10, 1, 1, 1]))**2, axis=-1)))
            candidates.append((evidence+10*wrench_error, points, normals, plan,
                               dict(thumb_faces=[left, right], human_cost=evidence,
                                    wrench_cost=wrench_error, solver_status=plan.status)))
    _, anchors, inward, plan, choice = min(candidates, key=lambda c: c[0])
    if np.max(np.linalg.norm(anchors-median, axis=1)) > .08:
        raise ValueError('No contact explanation within 8 cm of the demonstrated fingertips')
    if plan.status not in ('solved', 'solved inaccurate'):
        raise RuntimeError(f'Contact-force solve failed: {plan.status}')
    nominal_local = anchors-scene.radius*inward
    nominal = ref.positions[:, None]+np.einsum('tij,kj->tki', matrices, nominal_local)
    raw_positions = ref.positions.copy()
    initial_environment_penetration = scene.environment_penetration(ref.positions, ref.quaternions, nominal)
    registration = np.zeros(3)
    # A constant vertical correction preserves all object velocities,
    # accelerations, and rotations. Keep the recorded table in the experiment.
    # The same corrected object reference/initial state is used by every arm.
    for _ in range(4):
        penetration = scene.environment_penetration(raw_positions+registration, ref.quaternions, nominal+registration)
        if penetration < 1e-9:
            break
        registration[2] += penetration+.001
        if registration[2] > .03:
            raise ValueError('Environment clearance needs more than the allowed 3 cm object correction')
    else:
        raise ValueError('Could not establish an environment-clear initial reference')
    ref.positions = raw_positions+registration
    nominal += registration
    world_inward = np.einsum('tij,kj->tki', matrices, inward)
    world_forces = np.einsum('tij,tkj->tki', matrices, plan.forces)
    common = nominal+world_forces/scene.kp+.004*world_inward
    # Both methods start from the same zero-penetration supported grasp. A
    # shared 250 ms command handover avoids giving the raw reference an
    # artificial discontinuity at the beginning of this mid-episode crop.
    blend = np.clip((ref.times-ref.times[0])/.25, 0., 1.)
    blend = (blend**2*(3-2*blend))[:, None, None]
    evaluation_count = 0

    def evaluate(targets):
        nonlocal evaluation_count
        targets = (1-blend)*common+blend*targets
        result = scene.rollout(ref.times, targets, ref.positions, ref.quaternions,
                               initial_pads=nominal[0])
        evaluation_count += 1
        return objective(result), result

    baseline_targets = ref.robot_tips+registration-scene.radius*world_inward
    _, baseline = evaluate(baseline_targets)
    _, geometry = evaluate(nominal)
    shift, bias = np.zeros_like(ref.positions), np.zeros_like(ref.positions)
    best, best_score, preload = None, np.inf, None
    history = []
    for amount in (0., .002, .004, .006):
        score, result = evaluate(nominal+world_forces/scene.kp+amount*world_inward)
        history.append(dict(stage='preload_search', preload_m=amount, objective=score))
        if score < best_score:
            best_score, best, preload = score, result, amount
    initial_guided = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in best.items()}
    local = nominal_local+plan.forces/scene.kp+preload*inward
    for iteration in range(iterations):
        if perf_counter()-started > budget:
            break
        position_direction = gaussian_filter1d(ref.positions-best['object_positions'], 3, axis=0)
        rotation_direction = (rotations*Rotation.from_quat(best['object_wxyz'][:, [1, 2, 3, 0]]).inv()).as_rotvec()
        rotation_direction = gaussian_filter1d(rotation_direction, 3, axis=0)
        position_direction[0], rotation_direction[0] = 0., 0.
        improved = False
        for gain in (.7, .35, .15):
            if perf_counter()-started > budget:
                break
            proposal_shift = limit_norm(shift+gain*position_direction, .03)
            proposal_bias = limit_norm(bias+gain*rotation_direction, .3)
            command_rotation = Rotation.from_rotvec(proposal_bias)*rotations
            targets = ref.positions[:, None]+proposal_shift[:, None]+np.einsum('tij,tkj->tki', command_rotation.as_matrix(), local)
            score, result = evaluate(targets)
            collision_improved = (result['max_penetration_m'] < .001 or
                                  result['max_penetration_m'] < best['max_penetration_m'])
            keep = score < best_score and collision_improved
            history.append(dict(stage='shooting', iteration=iteration, gain=gain,
                                objective=score, max_penetration_mm=1000*result['max_penetration_m'], accepted=keep))
            if keep:
                best_score, best, shift, bias = score, result, proposal_shift, proposal_bias
                improved = True
                break
        if not improved:
            break
    runtime = perf_counter()-started
    surface_commands = best['targets']+scene.radius*world_inward
    human_deviation = np.linalg.norm(surface_commands-ref.human_tips-registration, axis=-1)
    robot_deviation = np.linalg.norm(surface_commands-ref.robot_tips-registration, axis=-1)
    demonstration_local = bool(np.max(human_deviation) < .08)
    report = dict(input_reference=str(ref.input_path), frames_inclusive=[int(ref.ids[0]), int(ref.ids[-1])],
                  duration_s=float(ref.times[-1]-ref.times[0]), method_seconds=runtime,
                  common_object_registration_xyz_m=registration.tolist(),
                  original_reference_environment_penetration_mm=1000*initial_environment_penetration,
                  rollout_evaluations=evaluation_count, accepted_supported_contact_demo=accepted(best),
                  contacts_within_demonstration_neighborhood=demonstration_local,
                  accepted_demonstration_local_contact_reference=accepted(best) and demonstration_local,
                  max_contact_command_shift_from_human_mm=1000*float(np.max(human_deviation)),
                  max_contact_command_shift_from_retargeted_mm=1000*float(np.max(robot_deviation)),
                  max_position_error_from_original_recording_mm=1000*float(np.max(np.linalg.norm(best['object_positions']-raw_positions, axis=1))),
                  robot_hand_or_arm_feasibility_validated=False, ready_for_training=False,
                  physical_parameters=dict(mass_kg=mass, friction=friction, pad_radius_m=scene.radius,
                      pad_mass_kg=scene.pad_mass, stiffness_n_m=scene.kp, damping_n_s_m=scene.kd,
                      pad_force_limit_n=scene.force_limit, timestep_s=scene.model.opt.timestep,
                      provenance=body.provenance),
                  contact_candidates=[c[-1] for c in candidates], selected_contact=choice,
                  preload_m=preload, history=history,
                  baseline=metrics(baseline, ref.times), geometry_only=metrics(geometry, ref.times),
                  wrench_seed=metrics(initial_guided, ref.times), corrected=metrics(best, ref.times),
                  max_bias_translation_mm=1000*float(np.max(np.linalg.norm(shift, axis=1))),
                  max_bias_rotation_deg=float(np.rad2deg(np.max(np.linalg.norm(bias, axis=1)))),
                  max_anchor_adjustment_from_human_mm=1000*float(np.max(np.linalg.norm(anchors-median, axis=1))),
                  force_plan_max_force_residual_n=float(np.max(np.linalg.norm(plan.residual[:, :3], axis=1))),
                  force_plan_max_torque_residual_nm=float(np.max(np.linalg.norm(plan.residual[:, 3:], axis=1))),
                  acceptance=dict(max_position_error_mm=10., max_rotation_error_deg=5., max_penetration_mm=1.,
                                  max_command_speed_m_s=1., max_command_acceleration_m_s2=10.,
                                  no_external_object_force=True, initially_nonpenetrating=True),
                  scope='Supported spherical fingertips with free book; no robot hand, arm, or walking feasibility claim')
    records = dict(baseline=baseline, geometry_only=geometry, wrench_seed=initial_guided, corrected=best)
    plan_data = dict(anchors=anchors, inward=inward, planned_forces_object=plan.forces, required_wrench=required,
                     position_bias=shift, rotation_bias_world=bias, initial_pads=nominal[0],
                     original_object_positions=raw_positions)
    return report, records, plan_data, scene.xml
