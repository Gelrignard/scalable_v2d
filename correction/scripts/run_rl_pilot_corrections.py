"""Run remaining prepared pilot episodes once, with bounded CPU concurrency."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
from time import perf_counter


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--jobs',type=int,default=2)
    parser.add_argument('--skip',nargs='*',default=[])
    args=parser.parse_args();repo=Path.cwd()
    config=json.loads((args.root/'selection.json').read_text())
    env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONPATH=str(repo),PYTHONUNBUFFERED='1')
    runtime=[str(repo/'.tools/Library/bin/micromamba.exe'),'run','-p',str(repo/'.retarget-env'),str(repo/'.retarget-env/python.exe')]
    def job(pair):
        tier,episode=pair;key=f'{tier}:{episode}'
        folder=args.root/f'tier{tier}'/f'episode_{episode:06d}'
        if key in args.skip:return dict(tier=tier,episode=episode,status='skipped_existing_process')
        prepared=folder/'prepare_report.json'
        if not prepared.exists() or json.loads(prepared.read_text())['status']!='complete':
            return dict(tier=tier,episode=episode,status='not_prepared')
        output=folder/'correction'
        if output.exists():
            return dict(tier=tier,episode=episode,status='existing_not_restarted',output=str(output))
        command=[*runtime,'-u','-m','correction.scripts.correct_rl_pilot_episode','--input',str(folder),'--output',str(output)]
        started=perf_counter();row=dict(tier=tier,episode=episode,status='running',command=command)
        path=folder/'correction_process.json'
        def write():path.write_text(json.dumps(row,indent=2)+'\n')
        with (folder/'correction.log').open('w',encoding='utf-8') as stream:
            child=subprocess.Popen(command,env=env,stdout=stream,stderr=subprocess.STDOUT)
            row['pid']=child.pid;write();print(tier,episode,'correction start',child.pid,flush=True)
            try:
                code=child.wait(timeout=3600)
                row.update(exit_code=code,status='complete' if code==0 else 'failed')
            except subprocess.TimeoutExpired:
                subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],capture_output=True)
                row.update(status='timeout',error='3600-second process budget exceeded')
        row['wall_seconds']=perf_counter()-started;write()
        print(tier,episode,row['status'],round(row['wall_seconds'],1),flush=True)
        return row
    pairs=[(tier,ep) for ep in config['episodes'] for tier in config['tiers']]
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:results=list(pool.map(job,pairs))
    (args.root/'correction_process_summary.json').write_text(json.dumps(results,indent=2)+'\n')


if __name__=='__main__':main()
