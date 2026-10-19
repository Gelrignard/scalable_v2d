"""CPU MuJoCo scene shared by collision checks and physical diagnostic replay.

URDF collision geometry/inertia are imported, with a floating robot base, a
free rigid book box, the recorded cylindrical table, and a floor. This is a
local diagnostic model, not the official Isaac challenge environment.
"""
from pathlib import Path
import re
import tempfile
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


def numbers(values):
    return ' '.join(str(float(x)) for x in np.asarray(values).ravel())


class RobotScene:
    def __init__(self, kin, bounds, support_path, *, mass=1., friction=.6, object_parts=None, floor_z=0.):
        self.kin = kin
        self.friction = friction
        root = ET.parse(kin.config.urdf_path).getroot()
        compiler = ET.SubElement(ET.SubElement(root, 'mujoco'), 'compiler')
        compiler.set('fusestatic', 'false')
        compiler.set('discardvisual', 'true')
        for link in root.findall('link'):
            for visual in link.findall('visual'):
                link.remove(visual)
            for j, collision in enumerate(link.findall('collision')):
                collision.set('name', f'{link.get("name")}_collision_{j}')
                mesh = collision.find('geometry/mesh')
                if mesh is not None:
                    mesh.set('filename', str((kin.config.urdf_path.parent/mesh.get('filename')).resolve()))
        with tempfile.TemporaryDirectory(prefix='v2d-scene-') as temporary:
            urdf = Path(temporary)/'robot.urdf'
            urdf.write_text(ET.tostring(root, encoding='unicode'), encoding='utf-8')
            imported = mujoco.MjModel.from_xml_path(str(urdf))
            mjcf = Path(temporary)/'robot.xml'
            mujoco.mj_saveLastXML(str(mjcf), imported)
            xml = ET.parse(mjcf).getroot()
        compiler = xml.find('compiler')
        compiler.set('angle', 'radian')
        compiler.set('fusestatic', 'false')
        # The importer may abbreviate mesh paths in its XML serialization.
        for mesh in xml.findall('asset/mesh'):
            path = Path(mesh.get('file'))
            if not path.is_absolute():
                matches = list(kin.config.urdf_path.parents[2].rglob(path.name))
                if len(matches) != 1:
                    raise ValueError(f'Cannot resolve imported mesh {path}')
                mesh.set('file', str(matches[0].resolve()))
        option = xml.find('option')
        if option is None:
            option = ET.SubElement(xml, 'option')
        option.attrib.update(timestep='0.002', integrator='implicitfast',
                             gravity='0 0 -9.81', iterations='40')
        world = xml.find('worldbody')
        pelvis = world.find('body')
        ET.SubElement(pelvis, 'freejoint', name='robot_root')
        for geom in world.iter('geom'):
            geom.set('friction', f'{friction} .005 .0001')
            geom.set('margin', '.002')
        for joint in world.iter('joint'):
            name = joint.get('name', '')
            joint.set('armature', '.00149' if 'hand' in name else '.01')
            joint.set('damping', '.02')
        ET.SubElement(world, 'geom', name='floor', type='plane', size='5 5 .1',
                      friction=f'{friction} .005 .0001', pos=numbers([0.,0.,floor_z]))
        support = Path(support_path).read_text()
        self.table_center, self.table_radius, self.table_top = np.zeros(3), 0., 0.
        if 'Cylinder' in support:
            radius = float(re.search(r'double radius = ([\d.eE+-]+)', support).group(1))
            height = float(re.search(r'double height = ([\d.eE+-]+)', support).group(1))
            center = list(map(float, re.search(r'xformOp:translate = \(([^)]+)\)', support).group(1).split(',')))
            self.table_center, self.table_radius, self.table_top = np.array(center), radius, center[2]+height/2
            ET.SubElement(world, 'geom', name='table', type='cylinder', size=numbers([radius, height/2]),
                          pos=numbers(center), friction=f'{friction} .005 .0001')
        elif re.search(r'def (?:Mesh|Cube|Sphere|Capsule)', support):
            raise ValueError('Unsupported support surface geometry')
        self.bounds = np.asarray(bounds)
        book = ET.SubElement(world, 'body', name='book')
        ET.SubElement(book, 'freejoint', name='book_root')
        if object_parts is None:
            ET.SubElement(book, 'geom', name='book_geom', type='box',
                          size=numbers((bounds[1]-bounds[0])/2), pos=numbers(bounds.mean(axis=0)),
                          mass=str(mass), friction=f'{friction} .005 .0001', margin='.002')
        else:
            # Convex pieces preserve cavities much better than one bounding box.
            # Mass fractions are explicit and fixed before the comparison.
            for i, (path, fraction) in enumerate(object_parts):
                ET.SubElement(xml.find('asset'), 'mesh', name=f'object_part_{i}', file=str(Path(path).resolve()))
                ET.SubElement(book, 'geom', name='book_geom' if i == 0 else f'object_geom_{i}',
                              type='mesh', mesh=f'object_part_{i}', mass=str(mass*fraction),
                              friction=f'{friction} .005 .0001', margin='.002')
        self.xml = ET.tostring(xml, encoding='unicode')
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.model)
        self.joint_ids = np.array([self.model.joint(n).id for n in kin.joint_names])
        self.qadr = self.model.jnt_qposadr[self.joint_ids]
        self.vadr = self.model.jnt_dofadr[self.joint_ids]
        self.root_q = int(self.model.joint('robot_root').qposadr[0])
        self.book_q = int(self.model.joint('book_root').qposadr[0])
        self.book_v = int(self.model.joint('book_root').dofadr[0])
        self.book_body = self.model.body('book').id
        self.book_geom = self.model.geom('book_geom').id
        self.floor_geom = self.model.geom('floor').id
        self.object_geoms = set(np.flatnonzero(self.model.geom_bodyid == self.book_body))
        self.table_geom = self.model.geom('table').id if self.table_radius else -1
        self.environment = {g for g in (self.floor_geom, self.table_geom) if g >= 0}

    def set_state(self, q, object_position, object_wxyz):
        self.data.qpos[self.root_q:self.root_q+3] = q[:3]
        self.data.qpos[self.root_q+3:self.root_q+7] = q[[6, 3, 4, 5]]
        self.data.qpos[self.qadr] = q[7:]
        self.data.qpos[self.book_q:self.book_q+3] = object_position
        self.data.qpos[self.book_q+3:self.book_q+7] = object_wxyz
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)

    def robot_q(self):
        root = self.data.qpos[self.root_q:self.root_q+7]
        return np.r_[root[:3], root[[4, 5, 6, 3]], self.data.qpos[self.qadr]]

    def contacts(self, movable=None, margin=.001):
        """Signed collision rows at current geometry, with optional Jacobians.

        MuJoCo's parent/same-body exclusions apply. Allowed foot-floor support
        is checked for penetration but is not requested to clear the floor.
        Object-environment rows cannot be repaired by robot IK; retain in audit.
        """
        rows = []
        for contact in self.data.contact[:self.data.ncon]:
            a, b = int(contact.geom1), int(contact.geom2)
            names = [self.model.geom(g).name for g in (a, b)]
            foot_support = self.floor_geom in (a, b) and any('ankle' in n for n in names)
            target = -.002 if foot_support else margin
            kind = 'robot_object' if self.object_geoms.intersection((a, b)) else (
                'environment' if self.environment.intersection((a, b)) else 'self')
            jacobian = None
            if movable is not None:
                jacs = []
                for geom in (a, b):
                    jp, jr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
                    mujoco.mj_jac(self.model, self.data, jp, jr, contact.pos,
                                  int(self.model.geom_bodyid[geom]))
                    jacs.append(jp[:, self.vadr[movable]])
                jacobian = np.asarray(contact.frame).reshape(3, 3)[0] @ (jacs[1]-jacs[0])
            rows.append(dict(distance=float(contact.dist), target=target, kind=kind,
                             pair=' / '.join(names), jacobian=jacobian))
        return rows

    def rollout(self, q, positions, quaternions, times, *, max_steps=5000):
        """Torque-limited PD replay; no root/object pose assignment after reset.

        Gains below are diagnostic assumptions. Initial root/object velocities
        come from the reference, and all subsequent motion comes from dynamics.
        """
        mujoco.mj_resetData(self.model, self.data)
        self.set_state(q[0], positions[0], quaternions[0])
        dt = times[1]-times[0]
        first = self.data.qpos.copy()
        self.set_state(q[1], positions[1], quaternions[1])
        second = self.data.qpos.copy()
        self.set_state(q[0], positions[0], quaternions[0])
        mujoco.mj_differentiatePos(self.model, self.data.qvel, dt, first, second)
        initial_penetration = max([max(0., -c['distance']) for c in self.contacts()], default=0.)
        hand = np.array(['hand' in n for n in self.kin.joint_names])
        kp, kd = np.where(hand, 2., 40.), np.where(hand, .2, 3.)
        caps = np.minimum(self.kin.robot.model.effortLimit[6:], np.where(hand, .76, np.inf))
        actual_q, actual_p, actual_r, max_force = [], [], [], []
        steps, saturated, nonfinite = 0, 0, False
        relative = times-times[0]
        for target_t in relative:
            while self.data.time < target_t-1e-10 and steps < max_steps:
                command = np.array([np.interp(self.data.time, relative, q[:, 7+j]) for j in range(len(hand))])
                raw = kp*(command-self.data.qpos[self.qadr])-kd*self.data.qvel[self.vadr]
                saturated += int(np.any(np.abs(raw) > caps))
                self.data.qfrc_applied[:] = 0
                self.data.qfrc_applied[self.vadr] = np.clip(raw, -caps, caps)
                mujoco.mj_step(self.model, self.data)
                steps += 1
                if not np.isfinite(self.data.qpos).all() or np.max(np.abs(self.data.qvel)) > 1e4:
                    nonfinite = True
                    break
            actual_q.append(self.robot_q())
            actual_p.append(self.data.qpos[self.book_q:self.book_q+3].copy())
            actual_r.append(self.data.qpos[self.book_q+3:self.book_q+7].copy())
            forces = []
            for i, c in enumerate(self.data.contact[:self.data.ncon]):
                if self.book_geom in (c.geom1, c.geom2):
                    wrench = np.zeros(6)
                    mujoco.mj_contactForce(self.model, self.data, i, wrench)
                    forces.append(float(np.linalg.norm(wrench[:3])))
            max_force.append(max(forces, default=0.))
            if nonfinite or steps >= max_steps:
                break
        return dict(q=np.asarray(actual_q), object_positions=np.asarray(actual_p),
                    object_wxyz=np.asarray(actual_r), max_book_contact_force_n=np.asarray(max_force),
                    steps=steps, saturation_steps=saturated, nonfinite=nonfinite,
                    initial_max_penetration_m=initial_penetration,
                    completed=len(actual_q) == len(q) and not nonfinite and self.data.time >= relative[-1]-1e-6)
