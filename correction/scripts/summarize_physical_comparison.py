"""Compare saved full episodes after asserting the common physical contract."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def read(path):
    with np.load(path,allow_pickle=False) as f:
        return {k:f[k].copy() for k in f.files}


def run(args):
    if args.output.exists(): raise ValueError('Use a fresh output directory')
    sources={'original':args.original,'geometry':args.geometry,'mechanics':args.mechanics}
    reports={k:json.loads((p/'report.json').read_text()) for k,p in sources.items()}
    refs={k:read(p/'reference.npz') for k,p in sources.items()}
    data={k:read(p/f'{k}.npz') for k,p in sources.items()}
    ref=refs['mechanics']; report=reports['mechanics']; initial=None
    for name in sources:
        for field in ('object_positions','object_wxyz','timestamps','baseline_command_q','initial_q','initial_velocity'):
            np.testing.assert_array_equal(refs[name][field],ref[field])
        for field in ('physics','controller','kp_hand','inertia_ff_scale','input_directory'):
            if reports[name][field]!=report[field]:
                raise ValueError(f'Mismatched physical contract: {name}, {field}')
        state=np.r_[data[name]['q'][0],data[name]['joint_velocities'][0],
                    data[name]['object_positions'][0],data[name]['object_wxyz'][0]]
        if initial is None: initial=state
        np.testing.assert_array_equal(initial,state)
        if len(data[name]['q'])!=len(ref['q']) or data[name]['max_object_applied_force']!=0:
            raise ValueError('Every case must be complete and unassisted')
    for field in ('q','command_velocity','force_ff','contact_active','robot_local_points','object_local_points'):
        np.testing.assert_array_equal(refs['geometry'][field],ref[field])
    args.output.mkdir(parents=True)
    result=dict(scope='Matched physical comparison assembled from preserved complete episodes',
        frames=len(ref['q']),duration_s=float(ref['timestamps'][-1]-ref['timestamps'][0]),
        common_initial_state=True,common_object_model_and_controller=True,
        original_reused=True,object_assistance=False,ready_for_training=False,
        original_reuse_note='Original command, initial state, target, model and controller match exactly. Inactive force-channel bookkeeping differs but commands no contact torque.',
        cases={k:reports[k]['cases'][k] for k in sources},
        sources={k:dict(directory=str(p.resolve()),reference_sha256=hashlib.sha256((p/'reference.npz').read_bytes()).hexdigest(),
                        rollout_sha256=hashlib.sha256((p/f'{k}.npz').read_bytes()).hexdigest()) for k,p in sources.items()},
        summary_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (args.output/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:dict(position_rms_mm=v['rms_position_error_mm'],rotation_rms_deg=v['rms_rotation_error_deg'],
                            accepted=v['accepted_physics']) for k,v in result['cases'].items()},indent=2))
    if not args.plot:
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,1,figsize=(11,9),layout='constrained')
    colors={'original':'#515b67','geometry':'#bd7b2e','mechanics':'#007e9b'}
    labels={'original':'Original retargeting','geometry':'Geometry only','mechanics':'Corrected grasp + force guidance'}
    t=ref['timestamps']
    for name in sources:
        axes[0].plot(t,1000*data[name]['position_error'],label=labels[name],color=colors[name])
        axes[1].plot(t,np.rad2deg(data[name]['rotation_error']),color=colors[name])
        axes[2].plot(data[name]['object_positions'][:,0],data[name]['object_positions'][:,1],color=colors[name],alpha=.85)
    axes[2].plot(ref['object_positions'][:,0],ref['object_positions'][:,1],color='#f14b1a',
                 linewidth=2,linestyle='--',label='Intended object path')
    for ax in axes[:2]:
        for a,b in ((235,480),(645,725)):
            ax.axvspan(t[a],t[b],color='#007e9b',alpha=.06)
        ax.set_xlabel('Episode time (s)');ax.grid(alpha=.2)
    axes[0].set_ylabel('Object position error (mm)');axes[0].legend(loc='upper left',ncols=3,fontsize=9)
    axes[1].set_ylabel('Object rotation error (degrees)')
    axes[2].set(xlabel='World X (m)',ylabel='World Y (m)')
    axes[2].set_aspect('equal',adjustable='datalim');axes[2].grid(alpha=.2);axes[2].legend(loc='upper right')
    fig.suptitle('Complete 858-frame physical episode · supported pelvis · source-derived 1 kg object\n'
                 'All candidates still fail acceptance; intended and simulated object trajectories are separate',fontsize=12)
    fig.savefig(args.output/'comparison.png',dpi=160)
    fig.savefig(args.output/'comparison.pdf')
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('original','geometry','mechanics','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--plot',action='store_true',help='Optional Matplotlib PNG/PDF; requires an existing installation')
    run(parser.parse_args())
