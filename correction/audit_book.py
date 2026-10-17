"""Recheck force balance at attained FK contacts, including contact gaps."""
from pathlib import Path
import json
import re
import numpy as np
from qpsolvers import solve_qp
from scipy.spatial.transform import Rotation
from .retarget_bridge import WORKSPACE
from .dynamics import world_points_to_object
from .optimizer import skew
from v2d_retarget.core.surface import ObjectSurface
from v2d_retarget.output import load_result


def basis(normal):
    axis=np.eye(3)[np.argmin(np.abs(normal))]
    u=np.cross(normal,axis);u/=np.linalg.norm(u)
    return np.column_stack((normal,u,np.cross(normal,u)))


def audit(directory):
    plan=np.load(directory/'wrench_plan.npz')
    ids=plan['ids']; times=plan['times']
    results={};saved={}
    for label,path in [('before',directory/'retargeted/contact_aware.npz'),('after',directory/'dynamics_guided.npz')]:
        data,metadata=load_result(path)
        surface=ObjectSurface(Path(str(data['object_mesh'])))
        pose=np.column_stack((data['object_positions'][ids],data['object_wxyz'][ids]))
        local=world_points_to_object(pose,data['fk_tip_positions'][ids])
        if label=='before':
            support=Path(metadata['source']['support_surface']).read_text()
            radius=float(re.search(r'double radius = ([\d.eE+-]+)',support).group(1))
            center=np.array(list(map(float,re.search(r'xformOp:translate = \(([^)]+)\)',support).group(1).split(','))))
            floor=[];table=[]
            for transform,position in zip(Rotation.from_quat(pose[:,3:],scalar_first=True).as_matrix(),pose[:,:3]):
                v=surface.mesh.vertices@transform.T
                floor.append(float(v[:,2].min()+position[2]))
                table.append(float(np.linalg.norm(position[:2]-center[:2])-np.linalg.norm(v[:,:2],axis=1).max()-radius))
            results['environment_clearance']={'minimum_object_floor_clearance_m':min(floor),
                'minimum_conservative_horizontal_table_clearance_m':min(table),
                'free_flight_separation_confirmed':bool(min(floor)>0 and min(table)>0)}
        nearest,distance,normals=surface.nearest(local)
        signed=np.sum((local-nearest)*normals,axis=-1)
        gap=signed-.002
        eligible=np.abs(gap)<=.003
        residual=[]; forces=[]
        scale=np.array([10.,10.,10.,1.,1.,1.])
        for t in range(len(ids)):
            bases=[basis(-n) for n in normals[t]]
            grasp=np.column_stack([np.vstack((b,skew(p-plan['com'])@b)) for p,b in zip(nearest[t],bases)])
            operator=grasp/scale[:,None]
            target=plan['required_wrench'][t]/scale
            G=np.zeros((24,18))
            for j in range(6):
                G[4*j:4*j+4,3*j:3*j+3]=np.array([[-.6*.85,a,b] for a,b in [(1,1),(1,-1),(-1,1),(-1,-1)]])
            caps=np.where(eligible[t],15.,0.)
            lower=np.array([[0.,-.6*c,-.6*c] for c in caps]).ravel()
            upper=np.array([[c,.6*c,.6*c] for c in caps]).ravel()
            f=solve_qp(2*(operator.T@operator+1e-9*np.eye(18)),-2*operator.T@target,G=G,h=np.zeros(24),lb=lower,ub=upper,solver='daqp')
            if f is None: raise RuntimeError('FK force audit QP failed')
            residual.append(grasp@f-plan['required_wrench'][t])
            forces.append(np.array([b@fk for b,fk in zip(bases,f.reshape(6,3))]))
        residual=np.array(residual)
        force_error=np.linalg.norm(residual[:,:3],axis=1)
        torque_error=np.linalg.norm(residual[:,3:],axis=1)
        good=(force_error<=.02)&(torque_error<=.002)
        collision_ok=data['probe_penetration_m'][ids].max(axis=1)<=.005
        results[label]={'contact_gap_tolerance_m':.003,'contact_gap_absolute_mean_mm':float(np.abs(gap).mean()*1000),
            'contact_gap_absolute_max_mm':float(np.abs(gap).max()*1000),'tip_frames_with_contact':int(eligible.sum()),'total_tip_frames':int(eligible.size),
            'mean_eligible_tips':float(eligible.sum(axis=1).mean()),'wrench_feasible_frames':int(good.sum()),'total_frames':len(ids),
            'max_force_residual_n':float(force_error.max()),'max_torque_residual_nm':float(torque_error.max()),
            'mean_force_residual_n':float(force_error.mean()),'mean_torque_residual_nm':float(torque_error.mean()),
            'frames_with_probe_penetration_at_most_5mm':int(collision_ok.sum()),
            'frames_passing_wrench_and_probe_checks':int((good&collision_ok).sum())}
        slip=np.linalg.norm(np.diff(local,axis=0),axis=-1)/np.diff(times)[:,None]
        results[label]['mean_object_relative_tip_speed_m_s']=float(slip.mean())
        results[label]['max_object_relative_tip_speed_m_s']=float(slip.max())
        results[label]['max_tip_distance_from_planned_anchor_m']=float(np.linalg.norm(local-data['contact_anchors_local'][ids],axis=-1).max())
        saved.update({label+'_gap_m':gap,label+'_eligible':eligible,label+'_residual':residual,label+'_forces':np.array(forces),label+'_probe_max_m':data['probe_penetration_m'][ids].max(axis=1)})
    before,_=load_result(directory/'retargeted/contact_aware.npz');after,_=load_result(directory/'dynamics_guided.npz')
    results['invariants']={key:bool(np.array_equal(before[key],after[key])) for key in ['object_positions','object_wxyz','timestamps']}
    results['invariants']['root_unchanged']=bool(np.array_equal(before['q'][:,:7],after['q'][:,:7]))
    results['accepted_as_feasible_manipulation']=False
    results['scope']='Instantaneous friction-cone wrench existence at attained FK; 3 mm gap allowance, 2 mm tip radius; provisional inertial/friction/force parameters. Does not enforce sticking or sliding friction direction. No actuation, balance or rollout validation.'
    (directory/'fk_force_audit.json').write_text(json.dumps(results,indent=2))
    np.savez_compressed(directory/'fk_force_audit.npz',times=times,**saved)
    print(json.dumps(results,indent=2))


if __name__=='__main__':
    audit(WORKSPACE/'correction/outputs/tier1_book')
