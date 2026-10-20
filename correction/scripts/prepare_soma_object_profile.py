"""Create a separate source-derived object/friction diagnostic configuration.

Matches the pinned SOMA exporter inertial defaults and FlashCHORD friction.
Other solver, robot, collision and fixture settings remain local: this is NOT
official-runtime parity and does not replace the prior 2kg stress benchmark.
"""
import argparse
import json
from pathlib import Path
import shutil
from correction.src.preprocessing.strategy_bundle import load_strategy_scene


def run(source,output):
    if output.exists():raise ValueError('Fresh output required')
    prior=json.loads((source/'report.json').read_text())
    sha='0593d80de5d39ab6f56f73ed16ca464ba5053499'
    base=f'https://github.com/nvidia-isaac/video_to_data/blob/{sha}/robotic_grounding/'
    report={k:prior[k] for k in ('input_directory','episode','physics')}
    report.update(scope='Source-derived object/friction profile; incomplete official parity',derived_from=str(source.resolve()),
        sources=[base+'scripts/retarget/soma_to_g1.py',base+'flash_chord/src/flash_chord/configs/experiment/g1_recon_body_flash_sac.yaml'],
        remaining_mismatches=['V-HACD vs official CoACD collision parts','local robot gains and gravity/inertia compensation',
            'local MuJoCo solver/timestep','supported pelvis rather than official whole-body control'])
    report['physics'].update(mass_kg=1.,friction=1.,object_inertial=dict(com_local_m=[0.,0.,0.],diagonal_kg_m2=[.01,.01,.01]))
    output.mkdir(parents=True);shutil.copy2(source/'reference.npz',output/'reference.npz')
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    _,_,_,scene=load_strategy_scene(output)
    import numpy as np
    np.testing.assert_allclose(scene.model.body_mass[scene.book_body],1.)
    np.testing.assert_allclose(scene.model.body_ipos[scene.book_body],0.,atol=1e-14)
    np.testing.assert_allclose(scene.model.body_inertia[scene.book_body],[.01]*3,atol=1e-14)
    report['compiled_model_verified']=True
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('correction/outputs/g1_controller_full_20261019_v2'))
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();run(args.source,args.output)
