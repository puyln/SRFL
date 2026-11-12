import json,os,subprocess,sys,time
from pathlib import Path
root=Path(__file__).resolve().parents[2]
results=[]
for name,config,parallel in [('parallel_views','views',True),('parallel_views_fused','views_fused',True),('parallel_views_sdpa','views_sdpa',True),('serial_views','views',False)]:
    output=root/'runs'/('benchmark_'+name)
    command=[sys.executable,'scripts/train.py','--config',str(root/f'config_benchmark_{config}.json'),'--output',str(output)]
    command+=['--devices','cuda:0','cuda:1','cuda:2','cuda:3'] if parallel else ['--device','cuda:0']
    began=time.perf_counter()
    with (root/'reports'/f'benchmark_{name}.log').open('w') as log:
        completed=subprocess.run(command,cwd=root/'SRFL',stdout=log,stderr=subprocess.STDOUT)
    results.append(dict(name=name,exit_code=completed.returncode,wall_seconds=time.perf_counter()-began))
    (root/'reports'/'benchmark_suite.json').write_text(json.dumps(results,indent=2))
    print(json.dumps(results[-1]),flush=True)
