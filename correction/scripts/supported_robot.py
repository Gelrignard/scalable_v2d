"""Run supported G1/Dex3 book manipulation with free object/upper-body physics."""
import argparse
import json
from pathlib import Path
import numpy as np

from correction.src.preprocessing.supported_reference import load_supported_reference
from correction.src.optimization.robot_manipulation import run,metrics,tracking_accepted
from correction.src.dynamics.supported_robot import SupportedRobotScene
from correction.src.pass_traj.robot_reference import save_robot_experiment


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=Path('correction/outputs/reorganization_check_20261017/retargeted/contact_aware.npz'))
    parser.add_argument('--output',type=Path,default=Path('correction/outputs/supported_g1_20261019'))
    parser.add_argument('--start',type=int,default=520)
    parser.add_argument('--stop',type=int,default=566)
    parser.add_argument('--time-scale',type=float,default=2.)
    parser.add_argument('--iterations',type=int,default=8)
    parser.add_argument('--prepared-reference',type=Path,help='Optional cached G1 reference with q and local arrays; timing excludes its construction')
    parser.add_argument('--verify',action='store_true')
    args=parser.parse_args()
    if args.time_scale<=0 or args.iterations<0:parser.error('Invalid time scale or iteration count')
    ref=load_supported_reference(args.input,args.start,args.stop)
    prepared=None
    if args.prepared_reference:
        with np.load(args.prepared_reference,allow_pickle=False) as data:
            prepared=dict(q=data['prepared_q' if 'prepared_q' in data.files else 'q'].copy(),local=data['local'].copy())
    scene,report,records,interface=run(ref,time_scale=args.time_scale,iterations=args.iterations,prepared=prepared)
    if args.verify:
        checks={}
        for name,parameters in [('half_timestep',dict(dt=.0005)),('mass_0.9kg',dict(mass=.9)),
                                ('mass_1.1kg',dict(mass=1.1)),('friction_0.5',dict(friction=.5))]:
            other=SupportedRobotScene(scene.kin,ref.bounds,interface['support_path'],interface['fixed_q'],move_waist=True,**parameters)
            result=other.rollout(interface['q'],ref.positions,ref.quaternions,interface['timestamps'],
                                 force_ff=interface['force_ff'],kp_arm=300.,kp_hand=20.,inertia_ff=True,
                                 initial_q=interface['initial_q'],initial_velocity=interface['initial_velocity'])
            checks[name]=dict(parameters=parameters,accepted_tracking=tracking_accepted(result),**metrics(result))
            records['verify_'+name]=result
        report['held_out_replays_without_reoptimization']=checks
    save_robot_experiment(args.output,report,records,interface,scene.xml)
    print(json.dumps({k:report[k] for k in ('method_seconds','baseline','wrench_seed','corrected','accepted_tracking')},indent=2),flush=True)
    print(args.output.resolve(),flush=True)

if __name__=='__main__':main()
