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


def allocate_profile(scene,points,q,positions,wxyz,times,blend,*,friction,preload=2.,normal_cap=15.,torque_fraction=.9):
    start=perf_counter();count=len(q);contacts=len(points);width=3*contacts
    operators=[];constraints=[];low=[];high=[];wanted=[];jt_all=[];bias_all=[]
    for frame in range(count):
        scene.set_state(q[frame],positions[frame],wxyz[frame])
        p,_,jac,_=contact_kinematics(scene,points)
        inward=Rotation.from_quat(wxyz[frame],scalar_first=True).apply(-np.array([p.object_outward for p in points]))
        com=scene.data.xipos[scene.book_body]
        operator=np.column_stack([np.vstack((np.eye(3),cross_matrix(x-com))) for x in p])
        scale=np.array([10.,10.,10.,1.,1.,1.])
        operators.append(sparse.csc_matrix(operator/scale[:,None]))
        wanted.append(np.r_[-scene.model.body_mass[scene.book_body]*scene.model.opt.gravity*blend[frame],np.zeros(3)]/scale)
        rows=[];l=[];u=[]
        for k,n in enumerate(inward):
            basis=contact_basis(n);block=np.zeros((5,width))
            block[:,3*k:3*k+3]=np.array([[1,0,0]]+[[-.85*friction,a,b] for a,b in [(1,1),(1,-1),(-1,1),(-1,-1)]])@basis.T
            rows.extend(block);l.extend([preload*blend[frame],-np.inf,-np.inf,-np.inf,-np.inf])
            u.extend([normal_cap*blend[frame],0.,0.,0.,0.])
        jt=jac.reshape(width,-1).T;bias=scene.data.qfrc_bias[scene.active_v].copy()
        rows.extend(jt);l.extend(-torque_fraction*scene.caps-bias);u.extend(torque_fraction*scene.caps-bias)
        # Unloaded joints cannot be fixed by this force problem. Their gravity
        # compensation must already fit the chosen reserve; infeasibility is reported.
        constraints.append(sparse.csc_matrix(np.asarray(rows)));low.extend(l);high.extend(u)
        jt_all.append(jt);bias_all.append(bias)
    operator=sparse.block_diag(operators,format='csc');constraint=sparse.block_diag(constraints,format='csc')
    size=count*width
    difference=sparse.kron(sparse.diags([-1/np.diff(times),1/np.diff(times)],[0,1],shape=(count-1,count)),sparse.eye(width),format='csc')
    hessian=operator.T@operator+1e-5*sparse.eye(size)+2e-5*(difference.T@difference)
    problem=osqp.OSQP();problem.setup(P=sparse.triu(2*hessian,format='csc'),q=-2*np.asarray(operator.T@np.asarray(wanted).ravel()).ravel(),
        A=constraint,l=np.asarray(low),u=np.asarray(high),verbose=False,eps_abs=1e-6,eps_rel=1e-6,
        max_iter=8000,time_limit=8.,polishing=True)
    fit=problem.solve(raise_error=False);solved=fit.info.status_val in (1,2) and fit.x is not None and np.isfinite(fit.x).all()
    forces=fit.x.reshape(count,contacts,3) if solved else np.zeros((count,contacts,3))
    value=constraint@forces.ravel();violation=float(max(0.,np.max(np.asarray(low)-value),np.max(value-np.asarray(high))))
    residual=(operator@forces.ravel()-np.asarray(wanted).ravel()).reshape(count,6)*[10,10,10,1,1,1]
    torque=np.asarray(bias_all)+np.einsum('tij,tj->ti',np.asarray(jt_all),forces.reshape(count,width))
    audit=dict(kind='time-coupled quasistatic allocation; excludes inertial/tracking torque',status=fit.info.status,
        solved=bool(solved),constraint_violation=violation,preload_n=preload,torque_fraction_limit=torque_fraction,
        max_force_residual_n=float(np.linalg.norm(residual[:,:3],axis=1).max()),
        max_moment_residual_nm=float(np.linalg.norm(residual[:,3:],axis=1).max()),
        task_wrench_satisfied=bool(np.linalg.norm(residual[:,:3],axis=1).max()<=.1 and np.linalg.norm(residual[:,3:],axis=1).max()<=.01),
        max_static_torque_fraction=float(np.max(abs(torque)/scene.caps)),
        wall_seconds=perf_counter()-start,physics_success=False)
    if not solved or violation>1e-4:raise ValueError('Force-profile allocation rejected: '+str(audit))
    return forces,audit
