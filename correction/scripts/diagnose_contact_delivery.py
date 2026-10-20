"""Audit actual contact locations/wrenches and robot tracking, without resimulation."""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from scipy.spatial.transform import Rotation
from correction.src.preprocessing.strategy_bundle import read_npz


def run(directory):
    start = perf_counter()
    report = json.loads((directory/'report.json').read_text())
    ref = read_npz(directory/'reference.npz')
    data = read_npz(directory/'mechanics.npz')
    source = Path(report['input_directory'])
    movable = data['movable_joint_indices']
    kp_hand = report['kp_hand']
    # The frozen runner's gains are part of its explicit execution contract.
    from correction.src.preprocessing.strategy_bundle import load_strategy_scene
    _, _, _, scene = load_strategy_scene(source)
    kp = np.where(scene.hand, kp_hand, 300.)
    kd = np.where(scene.hand, .4, 8.)
    tracking_position = kp*(data['controller_servo_q']-data['q'][:, 7+movable])
    tracking_velocity = kd*(data['controller_servo_velocity']-data['joint_velocities'])
    compensation = data['controller_feedforward_torque']-data['controller_contact_torque']
    selected = data['controller_measured_contact_wrench_world'].sum(axis=1)
    names = [scene.kin.joint_names[i] for i in movable]
    rows = []
    for frame in [235,270,350,430,480,525,600,640,645,650,655,660,665,670,675,680,685,690,700,725]:
        rot = Rotation.from_quat(data['object_wxyz'][frame], scalar_first=True)
        anchors = rot.apply(ref['object_local_points'][frame])+data['object_positions'][frame]
        actual_points = data['controller_measured_contact_point_world'][frame]
        weights = data['controller_measured_contact_weight_n'][frame]
        displacement = np.linalg.norm(actual_points-anchors, axis=1)
        commanded_force = data['controller_commanded_contact_force'][frame]
        planned_wrench = np.r_[ref['force_ff'][frame].sum(axis=0),
            np.cross(anchors-data['object_com_world'][frame], ref['force_ff'][frame]).sum(axis=0)]
        # Planned wrench is re-expressed at the current object pose for this
        # audit; it remains a requested quantity, not a measured force.
        actuator = tracking_position[frame]+tracking_velocity[frame]+compensation[frame]+data['controller_contact_torque'][frame]
        index = int(np.argmax(abs(actuator)/data['torque_caps']))
        rows.append(dict(frame=frame, position_error_mm=1000*data['position_error'][frame],
            rotation_error_deg=float(np.rad2deg(data['rotation_error'][frame])),
            planned_wrench_current_pose=planned_wrench.tolist(),
            selected_actual_wrench=selected[frame].tolist(),
            all_robot_actual_wrench=data['robot_contact_wrench_world'][frame].tolist(),
            environment_actual_wrench=data['environment_contact_wrench_world'][frame].tolist(),
            selected_force_n=np.linalg.norm(data['controller_measured_contact_force'][frame], axis=1).tolist(),
            commanded_force_n=np.linalg.norm(commanded_force, axis=1).tolist(),
            selected_anchor_error_mm=[float(1000*d) if w>1e-6 else None for d,w in zip(displacement,weights)],
            gate_gap_mm=(1000*data['controller_contact_gate_gap_m'][frame]).tolist(),
            touching=data['controller_touching'][frame].tolist(),
            max_servo_error_rad=float(abs(data['controller_servo_q'][frame]-data['q'][frame,7+movable]).max()),
            largest_torque_joint=names[index], requested_fraction=float(abs(actuator[index])/data['torque_caps'][index]),
            torque_terms_nm=dict(position=float(tracking_position[frame,index]), velocity=float(tracking_velocity[frame,index]),
                compensation=float(compensation[frame,index]), contact=float(data['controller_contact_torque'][frame,index]))))
    audit = dict(source=str(directory.resolve()), source_sha256=hashlib.sha256((directory/'mechanics.npz').read_bytes()).hexdigest(),
        wrench_convention='world [Fx,Fy,Fz,Mx,My,Mz] at actual object COM; solver forces',
        sensor_note='Selected force matches robot patch within30mm and normal within60deg; object anchor may differ.',
        sampling_note='State/force samples at rounded input timestamps. Final controller sample is the held previous command. No force reconstructed from qvel=0.',
        ready_for_training=False, frames=rows, wall_seconds=perf_counter()-start)
    (directory/'contact_delivery_audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    print(json.dumps(dict(wall_seconds=audit['wall_seconds'], frames=[
        {k: r[k] for k in ('frame','position_error_mm','rotation_error_deg','all_robot_actual_wrench',
                           'environment_actual_wrench','selected_anchor_error_mm')} for r in rows]), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    run(parser.parse_args().directory)
