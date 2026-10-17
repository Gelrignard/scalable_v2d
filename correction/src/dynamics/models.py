from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def finite_array(value, shape, name):
    result = np.asarray(value, dtype=float)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite with shape {shape}")
    return result


@dataclass(frozen=True)
class RigidBody:
    """Inertia is about the COM, expressed in the object's local frame."""

    mass_kg: float
    com_object_m: np.ndarray
    inertia_com_object_kg_m2: np.ndarray
    gravity_world_m_s2: np.ndarray
    provenance: str

    def __post_init__(self):
        if not np.isfinite(self.mass_kg) or self.mass_kg <= 0:
            raise ValueError("mass_kg must be positive")
        com = finite_array(self.com_object_m, (3,), "com_object_m")
        inertia = finite_array(self.inertia_com_object_kg_m2, (3, 3), "inertia")
        gravity = finite_array(self.gravity_world_m_s2, (3,), "gravity")
        eigen = np.linalg.eigvalsh(inertia)
        if not np.allclose(inertia, inertia.T) or eigen[0] <= 0:
            raise ValueError("inertia must be symmetric positive definite")
        if eigen[-1] > eigen[:2].sum() + 1e-10:
            raise ValueError("principal moments violate the inertia triangle inequality")
        if not self.provenance.strip():
            raise ValueError("physical parameters need explicit provenance/assumptions")
        object.__setattr__(self, "com_object_m", com)
        object.__setattr__(self, "inertia_com_object_kg_m2", inertia)
        object.__setattr__(self, "gravity_world_m_s2", gravity)


@dataclass(frozen=True)
class ContactPatch:
    """One bounded planar patch for a sticking point contact.

    center_object_m is a point on the object's surface; normal points INTO the
    object (the direction of a positive compressive force applied to it).
    The surface approximation and force cap are supplied inputs, not inferred
    robot reachability or actuator feasibility.
    """

    name: str
    center_object_m: np.ndarray
    inward_normal_object: np.ndarray
    tangent_u_object: np.ndarray
    tangent_v_object: np.ndarray
    half_extent_uv_m: np.ndarray
    friction: float
    max_normal_force_n: float

    def __post_init__(self):
        for key in ("center_object_m", "inward_normal_object", "tangent_u_object", "tangent_v_object"):
            object.__setattr__(self, key, finite_array(getattr(self, key), (3,), key))
        extent = finite_array(self.half_extent_uv_m, (2,), "half_extent_uv_m")
        if np.any(extent <= 0):
            raise ValueError("patch extents must be positive")
        object.__setattr__(self, "half_extent_uv_m", extent)
        if not np.allclose(self.basis.T @ self.basis, np.eye(3), atol=1e-7):
            raise ValueError("contact normal and tangents must be orthonormal")
        if not np.isfinite(self.friction) or self.friction < 0:
            raise ValueError("friction must be finite and nonnegative")
        if not np.isfinite(self.max_normal_force_n) or self.max_normal_force_n <= 0:
            raise ValueError("max_normal_force_n must be finite and positive")

    @property
    def basis(self):
        return np.column_stack((self.inward_normal_object, self.tangent_u_object, self.tangent_v_object))

    @property
    def tangents(self):
        return self.basis[:, 1:]

    def point(self, uv):
        return self.center_object_m + self.tangents @ uv
