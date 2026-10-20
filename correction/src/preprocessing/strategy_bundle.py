"""Load the saved full-episode comparison into the shared geometry diagnostic."""
from pathlib import Path
import json
import numpy as np
from . import retarget_bridge
from v2d_retarget.core.robot import G1Kinematics
from .convex_object import decompose
from correction.src.dynamics.supported_robot import SupportedRobotScene


def read_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def load_strategy_scene(source, *, physics_friction=None):
    source = Path(source)
    ref = read_npz(source/'reference.npz')
    report = json.loads((source/'report.json').read_text())
    raw_path = Path(report['input_directory'])/'retargeted/contact_aware.npz'
    raw = read_npz(raw_path)
    meta = json.loads(str(raw['metadata_json']))
    physics = report['physics']
    parts = decompose(str(ref['object_mesh']), Path(report['input_directory'])/'collision')
    scene = SupportedRobotScene(G1Kinematics(), ref['bounds'], meta['source']['support_surface'], ref['initial_q'],
        mass=physics['mass_kg'], friction=physics['friction'] if physics_friction is None else physics_friction,
        dt=physics['timestep_s'], move_waist=True, moving_fixture=True,
        object_parts=parts, floor_z=float(ref['floor_z']),object_inertial=physics.get('object_inertial'))
    return ref, raw, report, scene
