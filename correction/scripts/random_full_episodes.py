"""Evaluate the recorded random selection; never replace episodes after failure."""
import argparse
import json
from pathlib import Path
from correction.src.optimization.full_episode import run
from correction.src.pass_traj.robot_reference import save_robot_experiment

# Explicit diagnostic assumptions, fixed before any physical rollout.
MASSES = {'blue_trash_can':2., 'white_desk':10., 'toy_stroller':3.}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,default=Path('correction/outputs/random_three_full_20261019'))
    parser.add_argument('--episode',type=int)
    args = parser.parse_args()
    selection = json.loads((args.directory/'selection.json').read_text())
    for episode in selection['selected']:
        if args.episode is not None and args.episode != episode['episode_index']: continue
        directory = args.directory/f'episode_{episode["episode_index"]:06d}'
        print(f'FULL EPISODE {episode["episode_index"]}: {episode["object"]}',flush=True)
        scene,report,records,interface = run(directory/'retargeted/contact_aware.npz',directory,mass=MASSES[episode['object']])
        report.update(episode=episode,random_seed=selection['seed'])
        if report['frames'] != episode['frames']: raise ValueError('Incomplete episode')
        save_robot_experiment(directory,report,records,interface,scene.xml)
        print(json.dumps(report,indent=2),flush=True)

if __name__ == '__main__':main()
