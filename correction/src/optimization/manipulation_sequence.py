"""Compose contact phases and an explicit free-hand transition.

Only commands are composed. Physical state is never stitched or reset.
Timing and clearance are declared prototype choices, not inferred optima.
"""
from dataclasses import dataclass
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.spatial.transform import Rotation, Slerp
from .surface_reference import PhaseTiming, build_reference, smoothstep
from .rigid_grasp import RigidGrasp
from .actuator_force_profile import allocate_profile


@dataclass
class ContactPhase:
    points: list
    grasp_q: np.ndarray
    frame: int
    forces: np.ndarray
    timing: PhaseTiming


def union_channels(phases, commands, count):
    """One sensor per physical body; change its patch only between contacts."""
    points_by_body = {}
    for phase in phases:
        for p in phase.points:
            points_by_body.setdefault(p.body, p)
    points = list(points_by_body.values())
    index = {p.body: i for i, p in enumerate(points)}
    shape = (count, len(points))
    forces = np.zeros((*shape, 3))
    active = np.zeros(shape, bool)
    local = np.tile([p.local for p in points], (count, 1, 1)).astype(float)
    object_local = np.tile([p.object_local for p in points], (count, 1, 1)).astype(float)
    normals = np.tile([-p.object_outward for p in points], (count, 1, 1)).astype(float)
    for phase_id, (phase, command) in enumerate(zip(phases, commands)):
        for j, p in enumerate(phase.points):
            channel = index[p.body]
            if phase_id:
                previous_end = phases[phase_id-1].timing.force_end+1
                next_start = phase.timing.contact_start-1
                if previous_end >= next_start:
                    raise ValueError('Regrasp needs an inactive surface-transition interval')
                a = smoothstep((np.arange(count)-previous_end)/(next_start-previous_end))[:, None]
                local[:, channel] = (1-a)*local[:, channel]+a*p.local
                object_local[:, channel] = (1-a)*object_local[:, channel]+a*p.object_local
                normals[:, channel] = (1-a)*normals[:, channel]+a*(-p.object_outward)
            if np.any(active[:, channel] & command['contact_active'][:, j]):
                raise ValueError('Overlapping physical contact strategies on one body')
            forces[:, channel] += command['force_ff'][:, j]
            active[:, channel] |= command['contact_active'][:, j]
    return points, dict(force_ff=forces, contact_active=active,
                        inward_normals_local=normals, robot_local_points=local,
                        object_local_points=object_local)


def build_sequence(scene, raw, phases, *, friction, transition_lift=.15, preload=2.,
                   second_shift=None, arm_neighborhood=.9):
    if len(phases) != 2:
        raise ValueError('This bounded prototype handles two nonoverlapping interactions')
    commands, reports, planning_inputs = [], [], []
    for phase_id, phase in enumerate(phases):
        planning = dict(raw)
        if phase_id == 1 and second_shift is not None:
            coefficient = 1-smoothstep((np.arange(len(raw['q']))-phase.timing.force_full)/
                                       (phase.timing.contact_end-phase.timing.force_full))
            planning['object_positions'] = raw['object_positions']+coefficient[:, None]*second_shift
        planning_inputs.append(planning)
        # Construct the palm grasp from the unchanged source pose. Translation
        # correction then moves the planning path, not the grasp definition.
        command, report = build_reference(scene, planning, phase.points, phase.grasp_q, phase.forces,
            phase.frame, approach=True, rigid_grasp=True, release=True, phase=phase.timing,
            arm_neighborhood=arm_neighborhood,
            grasp_object_pose=(raw['object_positions'][phase.frame], raw['object_wxyz'][phase.frame]))
        commands.append(command); reports.append(report)
    q = commands[0]['q'].copy()
    left, right = phases[0].timing.hold_end, phases[1].timing.approach_start
    if left >= right:
        raise ValueError('Free transition must have a positive duration')
    q[right:] = commands[1]['q'][right:]
    solver = RigidGrasp(scene, phases[0].points, phases[0].grasp_q,
                        raw['object_positions'][phases[0].frame], raw['object_wxyz'][phases[0].frame])
    endpoints = []
    for frame in (left, right):
        scene.set_state(q[frame], raw['object_positions'][frame], raw['object_wxyz'][frame])
        endpoints.append((scene.data.xpos[solver.bodies].copy(),
                          Rotation.from_matrix(scene.data.xmat[solver.bodies].reshape(2, 3, 3))))
    rotations = [Slerp([0., 1.], Rotation.concatenate([endpoints[0][1][i], endpoints[1][1][i]]))
                 for i in range(2)]
    transition = []
    for frame in range(left+1, right):
        fraction = (frame-left)/(right-left)
        u = smoothstep(fraction)
        target = (1-u)*endpoints[0][0]+u*endpoints[1][0]
        target[:, 2] += transition_lift*16*fraction**2*(1-fraction)**2
        fingers = (1-u)*q[left]+u*q[right]
        q[frame], audit = solver.solve(raw['q'][frame], raw['object_positions'][frame],
            raw['object_wxyz'][frame], q[frame-1],
            palm_targets=(target, [r(u) for r in rotations]), finger_q=fingers, arm_neighborhood=arm_neighborhood)
        transition.append(dict(frame=frame, **audit))
    times = commands[0]['timestamps']
    spline = PchipInterpolator(times, q[:, 7+scene.movable], axis=0)
    profiles = []
    for phase, command, planning in zip(phases, commands, planning_inputs):
        timing = phase.timing
        frames = np.arange(len(q))
        blend = smoothstep((frames-timing.contact_start)/(timing.force_full-timing.contact_start))
        blend *= smoothstep((timing.force_end-frames)/(timing.force_end-timing.contact_end))
        command['force_ff'], audit = allocate_profile(scene, phase.points, q, planning['object_positions'],
            raw['object_wxyz'], times, blend, friction=friction, preload=preload)
        profiles.append(audit)
    points, fields = union_channels(phases, commands, len(q))
    dense = np.minimum(np.arange(round(times[-1]/.001)+1)*.001, times[-1])
    speed = scene.kin.robot.model.velocityLimit[6+scene.movable]
    derivative = spline(dense, 1)
    acceleration = spline(dense, 2)
    return points, dict(q=q, command_velocity=spline(times, 1), timestamps=times, **fields), dict(
        phases=reports, force_profiles=profiles, transition_audits=transition,
        transition_lift_m=transition_lift, transition_frames=[left, right],
        second_planning_translation_m=None if second_shift is None else np.asarray(second_shift).tolist(),
        arm_neighborhood_rad=arm_neighborhood,
        max_speed_fraction=float(np.max(abs(derivative)/speed)),
        max_command_acceleration_rad_s2=float(np.max(abs(acceleration))),
        interpolation='PCHIP C1 over composed joint commands; original clock and world frame',
        full_episode_physics_required=True, ready_for_training=False)
