"""Small collision-audited catalog of inactive regrasp finger paths.

Contact fingers retain their phase-two endpoint. Unused thumbs may remain open.
Opening and reshaping are separate operations to avoid interpolating through
another finger. This is a bounded path search, not an endpoint IK certificate.
"""
import numpy as np
import osqp
from scipy import sparse
from scipy.interpolate import PchipInterpolator, CubicHermiteSpline
from scipy.spatial.transform import Rotation, Slerp
from time import perf_counter
from ..dynamics.collision_distance import distance_row


def candidates(scene, reference, *, start=525, end=600):
    active = reference['contact_active']
    if np.any(active[start:end]):
        raise ValueError('Finger reshaping requires an inactive contact interval')
    ids = 7+scene.movable[scene.hand]
    names = [scene.kin.joint_names[i] for i in scene.movable[scene.hand]]
    thumb = np.array(['thumb' in n for n in names])
    distal = np.array(['thumb_1' in n or 'thumb_2' in n for n in names])
    first, last = reference['q'][start,ids], reference['q'][end,ids].copy()
    last[thumb] = 0.
    # Verify that an open unused thumb does not erase a loaded contract.
    for channel, point in enumerate(reference['_points']):
        if 'thumb' in scene.model.body(point.body).name and np.any(active[end:,channel]):
            raise ValueError('Cannot open a thumb with a later loaded contact')
    neutral = np.zeros_like(first)
    unfold = first.copy(); unfold[distal] = 0.
    open_index = first.copy(); open_index[~thumb] = 0.
    options = dict(
        all_open=[first, neutral, neutral, last],
        unfold_first=[first, unfold, neutral, last],
        index_open_first=[first, open_index, neutral, last])
    times = reference['timestamps']
    for name, values in options.items():
        q = reference['q'].copy()
        knots = np.linspace(times[start],times[end],4)
        spline = PchipInterpolator(knots,np.array(values),axis=0)
        q[start:end+1,ids] = spline(times[start:end+1])
        q[end:,ids[thumb]] = 0.
        command = PchipInterpolator(times,q[:,7+scene.movable],axis=0)
        result = {k:v for k,v in reference.items() if k!='_points'}
        result.update(q=q, command_velocity=command(times,1))
        yield name, result


def audit_path(scene, reference, *, start=525, end=600, frequency=150):
    """Check actual mesh geometry over the executed Hermite interpolation."""
    t = reference['timestamps']
    trajectory = CubicHermiteSpline(t,reference['q'][:,7+scene.movable],reference['command_velocity'])
    samples = np.linspace(t[start],t[end],int(np.ceil((t[end]-t[start])*frequency))+1)
    roots = Slerp(t,Rotation.from_quat(reference['q'][:,3:7]))(samples).as_quat()
    objects = Slerp(t,Rotation.from_quat(reference['object_wxyz'],scalar_first=True))(samples).as_quat(scalar_first=True)
    q = reference['q'][start].copy()
    maximum, pair, worst_time = 0., '', None
    max_robot, late_robot = 0., 0.
    late_penetration = 0.
    points = []
    for i,time in enumerate(samples):
        q[:3] = [np.interp(time,t,reference['q'][:,j]) for j in range(3)]
        q[3:7] = roots[i]; q[7+scene.movable] = trajectory(time)
        position = np.array([np.interp(time,t,reference['object_positions'][:,j]) for j in range(3)])
        scene.set_state(q,position,objects[i])
        for row in scene.contacts(scene.movable):
            if np.linalg.norm(row['jacobian'])>1e-10:
                depth=max(0.,-row['distance'])
                max_robot=max(max_robot,depth)
                if time>t[start]+.3: late_robot=max(late_robot,depth)
            if 'hand' not in row['pair']: continue
            depth = max(0., -row['distance'])
            if depth>maximum: maximum,pair,worst_time = depth,row['pair'],float(time)
            if time>t[start]+.3: late_penetration = max(late_penetration,depth)
            if depth>.0003:
                points.append(dict(time=float(time),pair=row['pair'],depth_m=depth))
    return dict(samples=len(samples), frequency_hz=frequency, max_hand_penetration_m=maximum,
        late_hand_penetration_m=late_penetration, worst_pair=pair, worst_time=worst_time,
        max_robot_penetration_m=max_robot, late_robot_penetration_m=late_robot,
        accepted=bool(max_robot<=.0003 and late_robot<=1e-6),
        initial_endpoint_allowance_m=.0003, initial_exit_seconds=.3,
        violations=points[:10], scope='All movable robot geometry on this inactive path; full robot dynamics still required')


def refine_free_path(scene, reference, *, start=525, end=600, iterations=20, seconds=25.,
                     dense_pairs=False):
    """Sequential convex collision repair for all free upper-body joints.

    No contact forces or object states are optimized here. This local repair
    uses soft linearized clearance and hard joint/speed/trust bounds. Its
    output requires dense nonlinear geometry and then unassisted replay.
    """
    began = perf_counter()
    t=reference['timestamps'][start:end+1]; count=len(t); width=len(scene.movable)
    if np.any(reference['contact_active'][start:end+1]):
        raise ValueError('Free-path refinement must not alter loaded motion')
    nominal=reference['q'][start:end+1,7+scene.movable].copy()
    x=nominal.copy(); dt=float(np.median(np.diff(t)))
    first=sparse.kron(sparse.diags([-np.ones(count-1),np.ones(count-1)], [0,1],
                        shape=(count-1,count))/dt,sparse.eye(width),format='csc')
    second=sparse.kron(sparse.diags([np.ones(count-2),-2*np.ones(count-2),np.ones(count-2)],[0,1,2],
                         shape=(count-2,count))/dt**2,sparse.eye(width),format='csc')
    weight=np.tile(np.where(scene.hand,.1,1.),count)
    base_h=sparse.diags(weight)+.005*(first.T@first)+.0001*(second.T@second)
    qlinear=-2*weight*nominal.ravel()
    lower=scene.kin.robot.model.lowerPositionLimit[7+scene.movable]
    upper=scene.kin.robot.model.upperPositionLimit[7+scene.movable]
    speed=scene.kin.robot.model.velocityLimit[6+scene.movable]
    history=[]
    remembered=[set() for _ in range(count)]
    remembered_mid=[set() for _ in range(count-1)]
    for iteration in range(iterations):
        if perf_counter()-began>seconds: break
        mid_constraints=[]
        if dense_pairs:
            # Recompute Hermite velocities each SQP iteration, then freeze
            # those derivatives in the midpoint linearization. Its position
            # sensitivity to each adjacent knot is 1/2. This extra collocation
            # constraint is necessary: two clear endpoints can sweep through
            # an obstacle even with 2mm clearance at each endpoint.
            sample_q=reference['q'].copy()
            sample_q[start:end+1,7+scene.movable]=x
            interpolator=PchipInterpolator(reference['timestamps'],sample_q[:,7+scene.movable],axis=0)
            mid=(t[:-1]+t[1:])/2
            root_rot=Slerp(reference['timestamps'],Rotation.from_quat(sample_q[:,3:7]))(mid).as_quat()
            object_rot=Slerp(reference['timestamps'],Rotation.from_quat(
                reference['object_wxyz'],scalar_first=True))(mid).as_quat(scalar_first=True)
            for i,time in enumerate(mid):
                q=reference['q'][start+i].copy()
                q[:3]=[np.interp(time,reference['timestamps'],sample_q[:,j]) for j in range(3)]
                q[3:7]=root_rot[i];q[7+scene.movable]=interpolator(time)
                p=[np.interp(time,reference['timestamps'],reference['object_positions'][:,j]) for j in range(3)]
                scene.set_state(q,p,object_rot[i])
                pairs={(int(c.geom1),int(c.geom2)) for c in scene.data.contact}
                remembered[i].update(pairs);remembered[i+1].update(pairs)
                remembered_mid[i].update(pairs)
                for geom1,geom2 in remembered_mid[i]:
                    contact=distance_row(scene,geom1,geom2)
                    if contact is None: continue
                    columns=np.r_[np.arange(i*width,(i+1)*width),
                                  np.arange((i+1)*width,(i+2)*width)]
                    coefficients=np.tile(.5*contact['jacobian'],2)
                    row=sparse.coo_matrix((coefficients,(np.zeros(2*width),columns)),
                                          shape=(1,count*width)).tocsc()
                    clearance=.002*min(1.,(time-t[0])/.3)
                    bound=clearance-contact['distance']+contact['jacobian']@(.5*(x[i]+x[i+1]))
                    mid_constraints.append((row,bound,contact))
        entries=[]; rhs=[]; maximum=0.; keys=[]
        for i,frame in enumerate(range(start,end+1)):
            q=reference['q'][frame].copy(); q[7+scene.movable]=x[i]
            scene.set_state(q,reference['object_positions'][frame],reference['object_wxyz'][frame])
            # Deduplicate solver contact points by geom pair and retain the
            # most restrictive linearized clearance row for each pair.
            remembered[i].update((int(c.geom1),int(c.geom2)) for c in scene.data.contact)
            rows={}
            for geom1,geom2 in remembered[i]:
                row=distance_row(scene,geom1,geom2)
                if row is None: continue
                maximum=max(maximum,max(0.,-row['distance']))
                rows[row['pair']]=row
            for pair,contact in rows.items():
                row=sparse.coo_matrix((contact['jacobian'],(np.zeros(width),np.arange(i*width,(i+1)*width))),
                                      shape=(1,count*width)).tocsc()
                # Preserve the pre-existing endpoint rather than teleporting
                # out of its 0.25mm overlap. Demand clearance after the exit.
                clearance=-.0003 if i==0 else .002*min(1.,(t[i]-t[0])/.3)
                entries.append(row); rhs.append(clearance-contact['distance']+contact['jacobian']@x[i])
                keys.append(pair)
        for row,bound,contact in mid_constraints:
            entries.append(row);rhs.append(bound);keys.append(contact['pair'])
            maximum=max(maximum,max(0.,-contact['distance']))
        n=count*width; nc=len(entries)
        collision=sparse.vstack(entries,format='csc') if entries else sparse.csc_matrix((0,n))
        # Explicit nonnegative collision slack prevents numerical infeasibility
        # from disguising a local dead end; nonlinear audit decides acceptance.
        identity=sparse.eye(n,format='csc'); zeros=sparse.csc_matrix((n,nc))
        matrix=sparse.vstack([
            sparse.hstack([identity,zeros]),
            sparse.hstack([first,sparse.csc_matrix((first.shape[0],nc))]),
            sparse.hstack([collision,sparse.eye(nc)]),
            sparse.hstack([sparse.csc_matrix((nc,n)),sparse.eye(nc)])],format='csc')
        lo=np.maximum(lower,x-.15); hi=np.minimum(upper,x+.15)
        lo[0]=hi[0]=nominal[0]; lo[-1]=hi[-1]=nominal[-1]
        low=np.r_[lo.ravel(),np.tile(-speed,count-1),rhs,np.zeros(nc)]
        high=np.r_[hi.ravel(),np.tile(speed,count-1),np.full(nc,np.inf),np.full(nc,np.inf)]
        hessian=sparse.block_diag([base_h,1e5*sparse.eye(nc)],format='csc')
        solver=osqp.OSQP()
        solver.setup(P=sparse.triu(2*hessian,format='csc'),q=np.r_[qlinear,np.zeros(nc)],
            A=matrix,l=low,u=high,eps_abs=1e-5,eps_rel=1e-5,max_iter=10000,
            time_limit=min(2.,max(.01,seconds-(perf_counter()-began))),verbose=False,polishing=True)
        result=solver.solve(raise_error=False)
        row=dict(iteration=iteration,status=result.info.status,pre_step_max_penetration_m=maximum,
                 collision_rows=nc,midpoint_collision_rows=len(mid_constraints))
        if result.info.status_val not in (1,2) or result.x is None:
            history.append(row);break
        value=matrix@result.x
        violation=max(0.,float(np.max(low-value)),float(np.max(value-high)))
        row.update(constraint_violation=violation,max_clearance_slack_m=float(np.max(result.x[n:],initial=0.)))
        history.append(row)
        if violation>1e-3: break
        x=result.x[:n].reshape(count,width)
    command=dict(reference); command['q']=reference['q'].copy()
    command['q'][start:end+1,7+scene.movable]=x
    spline=PchipInterpolator(reference['timestamps'],command['q'][:,7+scene.movable],axis=0)
    command['command_velocity']=spline(reference['timestamps'],1)
    return command,dict(iterations=history,wall_seconds=perf_counter()-began,
        midpoint_collision_pair_discovery=dense_pairs,
        midpoint_collision_constraints=dense_pairs,
        scope='Inactive-path collision/smoothness SQP; actuator dynamics and dense collisions remain separate gates')
