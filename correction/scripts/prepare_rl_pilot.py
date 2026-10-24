"""Decode and retarget the fixed paired pilot, with separate tier provenance."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
from time import perf_counter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--jobs', type=int, default=2)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=False)
    repo = Path.cwd()
    config = json.loads(Path('experiments/rl_pilot_tier12_six.json').read_text())
    (args.root/'selection.json').write_text(json.dumps(config, indent=2)+'\n')
    env = os.environ.copy()
    env.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONPATH=str(repo), PYTHONUNBUFFERED='1')
    soma = str(repo/'.soma-env/Scripts/python.exe')
    retarget = [str(repo/'.tools/Library/bin/micromamba.exe'), 'run', '-p', str(repo/'.retarget-env'), str(repo/'.retarget-env/python.exe')]
    def job(pair):
        tier, episode = pair
        folder = args.root/f'tier{tier}'/f'episode_{episode:06d}'
        folder.mkdir(parents=True)
        report = dict(tier=tier, episode=episode, status='running', stages=[], process_id=os.getpid())
        started = perf_counter()
        def write():
            report['wall_seconds'] = perf_counter()-started
            (folder/'prepare_report.json').write_text(json.dumps(report, indent=2)+'\n')
        write()
        commands = [
            ('decode', [soma, '-m', 'data_tools.decode_tier1', '--tier', str(tier), '--all-frames',
                        '--data', f'correction/data/tier{tier}', '--episode', str(episode), '--output', str(folder/'human_motion.npz')]),
            ('retarget', [*retarget, 'retargeting/scripts/run.py', '--input', str(folder/'human_motion.npz'),
                         '--method', 'contact_aware', '--output', str(folder/'retargeted')])]
        try:
            for name, command in commands:
                start = perf_counter()
                row = dict(name=name, command=command, status='running')
                report['stages'].append(row); write()
                print(tier, episode, name, 'start', flush=True)
                with (folder/f'{name}.log').open('w', encoding='utf-8') as stream:
                    child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, env=env)
                    row['pid'] = child.pid; write()
                    try:
                        code = child.wait(timeout=1800)
                    except subprocess.TimeoutExpired:
                        subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'], capture_output=True)
                        raise
                row.update(exit_code=code, wall_seconds=perf_counter()-start, status='complete' if code == 0 else 'failed')
                write()
                if code != 0: raise RuntimeError(f'{name} exited {code}; see {name}.log')
                print(tier, episode, name, 'complete', round(row['wall_seconds'], 2), flush=True)
            report['status'] = 'complete'
        except Exception as error:
            report.update(status='failed', error=f'{type(error).__name__}: {error}')
        finally:
            write()
        return report
    pairs = [(tier, episode) for episode in config['episodes'] for tier in config['tiers']]
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(job, pairs))
    (args.root/'prepare_summary.json').write_text(json.dumps(results, indent=2)+'\n')
    print('Prepared',sum(r['status']=='complete' for r in results),'of',len(results),flush=True)


if __name__ == '__main__':
    main()
