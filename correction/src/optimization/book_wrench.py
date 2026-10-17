"""Demonstration-preserving object-wrench correction for Tier 1 episode 11.

The thin book is approximated by its two broad planar faces for planning.
All quantities are subsequently checked against the actual object mesh and FK.
"""
from __future__ import annotations
from dataclasses import asdict
from time import perf_counter
import numpy as np
from correction.src.preprocessing.book_reference import load_book_reference
from correction.src.dynamics.object_wrench import object_wrench, world_points_to_object
from correction.src.dynamics.models import RigidBody, ContactPatch
from correction.src.optimization.contact_optimizer import CorrectionSettings, PhaseCorrector
from v2d_retarget.core.contacts import TIP_NAMES


def plan(directory, start, stop):
    started = perf_counter()
    motion, robot, _, cfg, surface, contacts = load_book_reference(directory)
    ids = np.arange(start, stop)
    if not contacts.active[ids].all():
        raise ValueError('Select a phase in which all six inferred contacts are active')
    anchors = contacts.anchors_local[ids]
    if np.max(np.abs(anchors-anchors[0])) > 1e-8:
        raise ValueError('A contact relatches inside the requested sticking phase')
    bounds = surface.mesh.bounds
    size = bounds[1]-bounds[0]
    com = bounds.mean(axis=0)
    mass = 1.0
    inertia = np.diag(mass/12 * (np.sum(size**2)-size**2))
    body = RigidBody(mass,com,inertia,np.array([0.,0.,-9.81]),
        'Provisional 1 kg homogeneous mesh bounding box; not measured challenge simulator parameters')
    pose = np.column_stack((motion.object_positions,motion.object_wxyz))
    dynamics = object_wrench(motion.timestamps,pose,np.ones(len(pose),bool),body)
    if not dynamics['valid_wrench'][ids].all():
        raise ValueError('Phase includes unreliable boundary derivatives')
    patches=[]
    for j,name in enumerate(TIP_NAMES):
        sign = np.sign(anchors[0,j,1]-com[1])
        center = com.copy()
        center[1] = bounds[1 if sign>0 else 0,1]
        patches.append(ContactPatch(name,center,np.array([0.,-sign,0.]),np.array([1.,0.,0.]),np.array([0.,0.,1.]),
            size[[0,2]]/2-.010,friction=.6,max_normal_force_n=15.))
    settings=CorrectionSettings(max_tangent_shift_m=.03,max_normal_projection_m=.006,
        target_sigma_m=.02,effort_weight=.05,max_iterations=40)
    problem=PhaseCorrector(motion.timestamps[ids],patches,anchors,dynamics['required_wrench_object'][ids],com,settings)
    # A human hand and Dex3 cannot place all three tips in identical local
    # configurations. Use saved robot FK as a second, soft geometric reference;
    # the original human anchors still define every hard +/-3 cm trust region.
    local_fk=world_points_to_object(pose[ids],robot['fk_tip_positions'][ids])
    robot_uv=np.array([patch.tangents.T@(point-patch.center_object_m)
        for patch,point in zip(patches,np.median(local_fk,axis=0))])
    problem.uv_reference=.25*problem.uv_reference+.75*robot_uv
    result=problem.solve()
    before=result['before']['contact_points_object_m']
    after=result['after']['contact_points_object_m']
    actual,distances,normals=surface.nearest(after)
    report=result['report']
    report.update(episode=11,phase_indices_inclusive=[start,stop-1],phase_times_s=motion.timestamps[ids[[0,-1]]].tolist(),
        physical_parameters={'mass_kg':mass,'com_object_m':com.tolist(),'inertia_kg_m2':inertia.tolist(),
         'friction':.6,'normal_force_cap_n':15.,'provenance':body.provenance},
        settings=asdict(settings),object_motion_changed=False,
        wrench_estimation='11-frame Savitzky-Golay pose filter then spline derivatives; playback uses unchanged raw poses',
        contact_representation='Six demonstration-selected contacts on broad planar book-face proxies, 12 constant patch coordinates; no grasp discovery',
        planar_face_edge_margin_m=.010,
        before_definition='Initial planar target after projection into the 10 mm edge-margin patch; actual untouched robot baseline is evaluated separately in fk_force_audit.json',
        kinematic_reference='Tangential objective: 25% demonstrated anchor + 75% median retargeted FK; hard 3 cm bound still centered on demonstrated anchor',
        correction_displacement_m=np.linalg.norm(after-before,axis=1).tolist(),
        mesh_projection_distance_m=distances.tolist(),
        optimized_points_on_mesh_object_m=actual.tolist(),
        total_preparation_and_planning_seconds=perf_counter()-started)
    arrays = dict(ids=ids,times=motion.timestamps[ids],
        before_points=before,after_points=after,mesh_points=actual,mesh_normals=normals,
        before_forces=result['before']['contact_forces_object_n'],after_forces=result['after']['contact_forces_object_n'],
        before_residual=result['before']['wrench_residual_object'],after_residual=result['after']['wrench_residual_object'],
        required_wrench=dynamics['required_wrench_object'][ids],com=com,
        inertia=inertia,mass=mass)
    return arrays, report
