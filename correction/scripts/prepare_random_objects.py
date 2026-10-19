"""Prepare the three already-selected object collision models, without resampling."""
import json
from pathlib import Path
import numpy as np
from correction.src.preprocessing.convex_object import decompose

if __name__ == '__main__':
    root = Path('correction/outputs/random_three_full_20261019')
    for episode in json.loads((root/'selection.json').read_text())['selected']:
        directory = root/f'episode_{episode["episode_index"]:06d}'
        with np.load(directory/'human_motion.npz') as data:
            mesh = str(data['object_mesh'])
        print(f'Decomposing {episode["object"]}', flush=True)
        parts = decompose(mesh, directory/'collision')
        print(f'{len(parts)} convex pieces', flush=True)
