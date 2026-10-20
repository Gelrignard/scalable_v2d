"""Recorded independent-pad intent and measured forces for the viewer.

These are visual annotations, never new simulated bodies or applied forces.
All arrow vectors are world-frame forces exerted by the pads on the object.
"""
import numpy as np
import trimesh


FORCE_SCALE = .01  # metres per newton, shared by planned and measured arrows
PLANNED_COLOR = (192, 82, 241)
MEASURED_COLOR = (31, 187, 105)
TARGET_COLOR = (35, 192, 231)


def arrow_segments(origins, forces, scale=FORCE_SCALE):
    """Shaft plus two arrowhead segments; vector lengths are not normalized."""
    origins = np.asarray(origins, dtype=float).reshape(-1, 3)
    vectors = scale*np.asarray(forces, dtype=float).reshape(-1, 3)
    lengths = np.linalg.norm(vectors, axis=1)
    mask = lengths > 1e-7
    origins, vectors, lengths = origins[mask], vectors[mask], lengths[mask]
    if not len(origins):
        return np.zeros((0, 2, 3), np.float32)
    directions = vectors/lengths[:, None]
    basis = np.eye(3)[np.argmin(np.abs(directions), axis=1)]
    sideways = np.cross(directions, basis)
    sideways /= np.linalg.norm(sideways, axis=1)[:, None]
    tips = origins+vectors
    head = np.minimum(.012, .25*lengths)[:, None]
    return np.concatenate((np.stack((origins, tips), axis=1),
        np.stack((tips, tips-head*directions+.45*head*sideways), axis=1),
        np.stack((tips, tips-head*directions-.45*head*sideways), axis=1))).astype(np.float32)


class ContactOverlay:
    def __init__(self, scene, prefix, intended_tips, active, *, planned_forces=None, radius=.006,
                 planned_origins=None):
        self.group = scene.add_frame(prefix, show_axes=False, visible=False)
        self.intended_tips = np.asarray(intended_tips)
        self.active = np.asarray(active, dtype=bool)
        self.planned_forces = planned_forces
        self.planned_origins = self.intended_tips if planned_origins is None else np.asarray(planned_origins)
        count=self.intended_tips.shape[1]
        marker = trimesh.creation.icosphere(subdivisions=2, radius=radius)
        self.actual_tips = [scene.add_mesh_simple(prefix+f'/tip{i}', marker.vertices, marker.faces,
            color=(72, 117, 192), cast_shadow=False, receive_shadow=False) for i in range(count)]
        self.targets = scene.add_point_cloud(prefix+'/intended_tips', self.intended_tips[0].astype(np.float32),
            colors=TARGET_COLOR, point_size=.012, point_shape='circle')
        self.contacts = scene.add_point_cloud(prefix+'/actual_contacts', np.zeros((2, 3), np.float32),
            colors=MEASURED_COLOR, point_size=.014, point_shape='circle')
        self.gaps = scene.add_line_segments(prefix+'/tracking_gaps', np.zeros((count, 2, 3), np.float32),
            colors=(120, 168, 190), line_width=1.)
        self.planned = scene.add_line_segments(prefix+'/planned_force', np.zeros((1, 2, 3), np.float32),
            colors=PLANNED_COLOR, line_width=3.)
        self.measured = scene.add_line_segments(prefix+'/measured_force', np.zeros((1, 2, 3), np.float32),
            colors=MEASURED_COLOR, line_width=3.)
        # Only the intended contact portions are traced, avoiding six full
        # approach/departure paths obscuring the object.
        p = np.stack((self.intended_tips[:-1], self.intended_tips[1:]), axis=2)
        valid = self.active[:-1] & self.active[1:]
        paths = p[valid].astype(np.float32)
        self.paths = scene.add_line_segments(prefix+'/intended_contact_paths',
            paths if len(paths) else np.zeros((1, 2, 3), np.float32),
            colors=TARGET_COLOR, line_width=1.)
        self.paths.visible = bool(len(paths))

    @staticmethod
    def set_arrows(handle, origins, forces, scale):
        segments = arrow_segments(origins, forces, scale)
        handle.visible = bool(len(segments))
        if len(segments):
            handle.points = segments

    def update(self, frame, record, visible, scale=FORCE_SCALE):
        self.group.visible = visible
        if not visible:
            return
        actual = record['pad_positions'][frame]
        intended = self.intended_tips[frame]
        for marker, position in zip(self.actual_tips, actual):
            marker.position = position
        self.targets.points = intended.astype(np.float32)
        self.gaps.points = np.stack((actual, intended), axis=1).astype(np.float32)
        contact = record['contact_counts'][frame] > 0
        self.contacts.visible = bool(np.any(contact))
        if np.any(contact):
            self.contacts.points = record['contact_points'][frame, contact].astype(np.float32)
        self.set_arrows(self.measured, record['contact_points'][frame, contact],
                        record['contact_forces'][frame, contact], scale)
        if self.planned_forces is None:
            self.planned.visible = False  # Baseline has no commanded force plan.
        else:
            active = self.active[frame]
            self.set_arrows(self.planned, self.planned_origins[frame,active], self.planned_forces[frame, active], scale)
