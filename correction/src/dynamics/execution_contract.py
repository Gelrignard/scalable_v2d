"""Shared physical limits and independent audits for contact candidates.

Audits are necessary checks, never an execution or policy-quality certificate.
Forces are robot-on-object and normals point into the object in the same frame.
"""
from dataclasses import dataclass, asdict
import numpy as np


@dataclass(frozen=True)
class ContactExecutionContract:
    friction: float = .45
    normal_limit_n: float = 15.
    physics_friction: float = .6
    contact_gap_tolerance_m: float = .003
    penetration_tolerance_m: float = .002
    normal_cosine_min: float = .7

    def __post_init__(self):
        values = np.array(list(asdict(self).values()))
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError('Contract values must be finite and nonnegative')
        if self.friction > self.physics_friction or self.normal_limit_n <= 0:
            raise ValueError('Planning friction must not exceed physical friction; force cap must be positive')
        if not 0 <= self.normal_cosine_min <= 1:
            raise ValueError('Invalid normal alignment threshold')

    def force_audit(self, forces, inward, active=None, tolerance_n=.01):
        f, n = np.asarray(forces, float), np.asarray(inward, float)
        if f.shape != n.shape or f.shape[-1] != 3 or not np.isfinite(f).all() or not np.isfinite(n).all():
            raise ValueError('Finite force/normal vectors of matching shape required')
        active = np.ones(f.shape[:-1], bool) if active is None else np.asarray(active, bool)
        if active.shape != f.shape[:-1]:
            raise ValueError('Active schedule shape mismatch')
        length = np.linalg.norm(n, axis=-1)
        n = n/np.maximum(length[..., None], 1e-12)
        normal = np.sum(f*n, axis=-1)
        tangent = np.linalg.norm(f-normal[..., None]*n, axis=-1)
        magnitude = np.linalg.norm(f, axis=-1)
        bad_normal = active & (length < 1e-8)
        cone = tangent > self.friction*np.maximum(normal, 0)+tolerance_n
        tension = normal < -tolerance_n
        cap = normal > self.normal_limit_n+tolerance_n
        inactive = ~active & (magnitude > tolerance_n)
        violation = bad_normal | (active & (cone | tension | cap)) | inactive
        return dict(accepted=bool(not violation.any()), samples=int(magnitude.size),
                    nontrivial_samples=int((magnitude > 1.).sum()),
                    nontrivial_cone_violations=int(((magnitude > 1.) & cone).sum()),
                    invalid_normal_samples=int(bad_normal.sum()),
                    tension_samples=int((active & tension).sum()),
                    normal_cap_samples=int((active & cap).sum()),
                    inactive_force_samples=int(inactive.sum()),
                    violation_samples=int(violation.sum()),
                    max_cone_excess_n=float(np.max(tangent-self.friction*np.maximum(normal, 0))),
                    max_normal_force_n=float(normal.max()), tolerance_n=float(tolerance_n))


def scene_contract(scene, contract):
    """Serialize actual MuJoCo settings, not a nominal XML/config assertion."""
    m = scene.model
    friction = m.geom_friction[:, 0]
    object_geoms = sorted(scene.object_geoms)
    return dict(contact=asdict(contract), object_mass_kg=float(m.body_mass[scene.book_body]),
                object_com_local_m=m.body_ipos[scene.book_body].tolist(),
                object_principal_inertia_kg_m2=m.body_inertia[scene.book_body].tolist(),
                object_inertia_frame_wxyz=m.body_iquat[scene.book_body].tolist(),
                timestep_s=float(m.opt.timestep), solver_iterations=int(m.opt.iterations),
                friction_range=[float(friction.min()), float(friction.max())],
                physics_friction_matches=bool(np.allclose(friction, contract.physics_friction)),
                object_geoms=[m.geom(i).name for i in object_geoms],
                contact_solref=m.geom_solref[object_geoms].tolist(),
                contact_solimp=m.geom_solimp[object_geoms].tolist(),
                movable_joints=[scene.kin.joint_names[i] for i in scene.movable],
                joint_effort_caps_nm=scene.caps.tolist(),
                supported_pelvis=bool(scene.moving_fixture), walking_simulated=False)
