"""Time-coupled quasistatic wrench/preload allocation on a robot reference.

World gravity stays world-fixed. Contact effort and force-rate regularization
select among infinitely many allocations. Explicit joint torque constraints
include gravity but not inertial/tracking torque: replay remains mandatory.
No per-frame observed acceleration is used.
"""
from time import perf_counter
import numpy as np
import osqp
from scipy import sparse
from scipy.spatial.transform import Rotation
from .surface_hypotheses import contact_kinematics
from ..dynamics.sparse_forces import contact_basis,cross_matrix
from ..dynamics.environment_contact import environment_contacts


def allocate_profile(scene,points,q,positions,wxyz,times,blend,*,friction,preload=2.,normal_cap=15.,torque_fraction=.9,
                     include_environment=False):
    start=perf_counter();count=len(q);contacts=len(points);slots=8 if include_environment else 0
    width=3*(contacts+slots)
    operators=[];constraints=[];low=[];high=[];wanted=[];jt_all=[];bias_all=[]
    support_counts=[];support_positions=[];support_velocities=[];support_modes=[]
    if include_environment:
        # Only first derivatives are used for the slip direction. They are not
        # acceleration-derived wrench targets.
        from scipy.interpolate import CubicSpline
        from scipy.spatial.transform import RotationSpline
        translation = CubicSpline(times, positions)
        rotation_curve = RotationSpline(times, Rotation.from_quat(wxyz, scalar_first=True))
    for frame in range(count):
        scene.set_state(q[frame],positions[frame],wxyz[frame])
        p,_,jac,_=contact_kinematics(scene,points)
        inward=Rotation.from_quat(wxyz[frame],scalar_first=True).apply(-np.array([p.object_outward for p in points]))
        com=scene.data.xipos[scene.book_body]
        support = environment_contacts(scene, slots) if include_environment else []
        env_p = np.tile(com, (slots, 1))
        env_p[:len(support)] = [c['position'] for c in support] if support else np.empty((0,3))
        all_p = np.vstack((p, env_p))
        operator=np.column_stack([np.vstack((np.eye(3),cross_matrix(x-com))) for x in all_p])
        scale=np.array([10.,10.,10.,1.,1.,1.])
        operators.append(sparse.csc_matrix(operator/scale[:,None]))
        demand_scale = 1. if include_environment else blend[frame]
        wanted.append(np.r_[-scene.model.body_mass[scene.book_body]*scene.model.opt.gravity*demand_scale,np.zeros(3)]/scale)
        rows=[];l=[];u=[]
        for k,n in enumerate(inward):
            basis=contact_basis(n);block=np.zeros((5,width))
            block[:,3*k:3*k+3]=np.array([[1,0,0]]+[[-.85*friction,a,b] for a,b in [(1,1),(1,-1),(-1,1),(-1,-1)]])@basis.T
            rows.extend(block);l.extend([preload*blend[frame],-np.inf,-np.inf,-np.inf,-np.inf])
            u.extend([normal_cap*blend[frame],0.,0.,0.,0.])
        env_v=np.zeros((slots,3));modes=[]
        for slot in range(slots):
            section=slice(3*(contacts+slot),3*(contacts+slot+1))
            if slot>=len(support):
                block=np.zeros((3,width));block[:,section]=np.eye(3)
                rows.extend(block);l.extend([0.]*3);u.extend([0.]*3);modes.append('absent')
                continue
            c=support[slot];basis=contact_basis(c['normal']);mu=c['friction']
            angular=rotation_curve(times[frame]).apply(rotation_curve(times[frame],1))
            env_v[slot]=translation(times[frame],1)+np.cross(angular,c['position']-positions[frame])
            tangent=basis[:,1:].T@env_v[slot]
            if np.linalg.norm(tangent) > .001:
                # Sliding friction opposes motion; it is not an arbitrary
                # tangential support wrench selected to make tracking easy.
                direction=tangent/np.linalg.norm(tangent)
                local=np.array([[1.,0,0],[mu*direction[0],1.,0],[mu*direction[1],0,1.]])
                block=np.zeros((3,width));block[:,section]=local@basis.T
                rows.extend(block);l.extend([0.,0.,0.]);u.extend([scene.model.body_mass[scene.book_body]*20.,0.,0.])
                modes.append('sliding')
            else:
                local=np.array([[1,0,0]]+[[-.85*mu,a,b] for a,b in [(1,1),(1,-1),(-1,1),(-1,-1)]])
                block=np.zeros((5,width));block[:,section]=local@basis.T
                rows.extend(block);l.extend([0.,-np.inf,-np.inf,-np.inf,-np.inf])
                u.extend([scene.model.body_mass[scene.book_body]*20.,0.,0.,0.,0.]);modes.append('sticking_candidate')
        jt=np.zeros((len(scene.caps),width));jt[:,:3*contacts]=jac.reshape(3*contacts,-1).T
        bias=scene.data.qfrc_bias[scene.active_v].copy()
        rows.extend(jt);l.extend(-torque_fraction*scene.caps-bias);u.extend(torque_fraction*scene.caps-bias)
        # Unloaded joints cannot be fixed by this force problem. Their gravity
        # compensation must already fit the chosen reserve; infeasibility is reported.
        constraints.append(sparse.csc_matrix(np.asarray(rows)));low.extend(l);high.extend(u)
        jt_all.append(jt);bias_all.append(bias)
        support_counts.append(len(support));support_positions.append(env_p);support_velocities.append(env_v);support_modes.append(modes)
    operator=sparse.block_diag(operators,format='csc');constraint=sparse.block_diag(constraints,format='csc')
    size=count*width
    difference=sparse.kron(sparse.diags([-1/np.diff(times),1/np.diff(times)],[0,1],shape=(count-1,count)),sparse.eye(width),format='csc')
    hessian=operator.T@operator+1e-5*sparse.eye(size)+2e-5*(difference.T@difference)
    problem=osqp.OSQP();problem.setup(P=sparse.triu(2*hessian,format='csc'),q=-2*np.asarray(operator.T@np.asarray(wanted).ravel()).ravel(),
        A=constraint,l=np.asarray(low),u=np.asarray(high),verbose=False,eps_abs=1e-6,eps_rel=1e-6,
        max_iter=8000,time_limit=8.,polishing=True)
    fit=problem.solve(raise_error=False);solved=fit.info.status_val in (1,2) and fit.x is not None and np.isfinite(fit.x).all()
    forces=fit.x.reshape(count,contacts+slots,3) if solved else np.zeros((count,contacts+slots,3))
    value=constraint@forces.ravel();violation=float(max(0.,np.max(np.asarray(low)-value),np.max(value-np.asarray(high))))
    residual=(operator@forces.ravel()-np.asarray(wanted).ravel()).reshape(count,6)*[10,10,10,1,1,1]
    torque=np.asarray(bias_all)+np.einsum('tij,tj->ti',np.asarray(jt_all),forces.reshape(count,width))
    audit=dict(kind='time-coupled quasistatic allocation; excludes inertial/tracking torque',status=fit.info.status,
        solved=bool(solved),constraint_violation=violation,preload_n=preload,torque_fraction_limit=torque_fraction,
        max_force_residual_n=float(np.linalg.norm(residual[:,:3],axis=1).max()),
        max_moment_residual_nm=float(np.linalg.norm(residual[:,3:],axis=1).max()),
        task_wrench_satisfied=bool(np.linalg.norm(residual[:,:3],axis=1).max()<=.1 and np.linalg.norm(residual[:,3:],axis=1).max()<=.01),
        max_static_torque_fraction=float(np.max(abs(torque)/scene.caps)),
        include_environment=include_environment,
        environment_contact_count=support_counts,
        environment_support_source='active physical collision manifold' if include_environment else 'disabled',
        zero_hand_blend_keeps_gravity_demand=include_environment,
        wall_seconds=perf_counter()-start,physics_success=False)
    if not solved or violation>1e-4:raise ValueError('Force-profile allocation rejected: '+str(audit))
    if include_environment:
        audit['environment_plan']=dict(forces_world=forces[:,contacts:],positions_world=np.asarray(support_positions),
                                       velocities_world=np.asarray(support_velocities),modes=np.asarray(support_modes))
    return forces[:,:contacts],audit
