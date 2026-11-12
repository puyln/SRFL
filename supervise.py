import argparse,json,os,subprocess,sys,time,traceback
from pathlib import Path

def write(path,data):
    tmp=path.with_suffix('.tmp'); tmp.write_text(json.dumps(data,indent=2)); tmp.replace(path)

def main():
    a=argparse.ArgumentParser(); a.add_argument('--root',required=True); a.add_argument('--method',required=True); a.add_argument('--gpu',type=int,required=True); a.add_argument('--dataset',default='mots'); args=a.parse_args()
    root=Path(args.root).resolve(); config=root/(args.dataset+'_config.json'); cfg=json.loads(config.read_text())
    status_path=root/f'{args.dataset}_{args.method}_status.json'
    state={'method':args.method,'dataset':args.dataset,'gpu':args.gpu,'supervisor_pid':os.getpid(),'state':'starting','runs':[]}
    write(status_path,state)
    for seed in cfg['benchmark']['seeds']:
        out=root/'runs'/args.dataset/args.method/str(seed); out.mkdir(parents=True,exist_ok=True)
        if (out/'completed.json').exists():
            state['runs'].append({'seed':seed,'state':'previously_completed'}); continue
        command=[sys.executable,str(Path(__file__).resolve().with_name('convergence.py')),'--config',str(config),'--method',args.method,'--gpu',str(args.gpu),'--seed',str(seed),'--out',str(out)]
        if (out/'last.pt').exists(): command.append('--resume')
        elif (out/'metrics.jsonl').exists(): raise RuntimeError('Partial baseline record without checkpoint; inspect before restarting')
        with (out/'stdout.log').open('a') as log:
            p=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,cwd=root,env={**os.environ,'PYTHONUNBUFFERED':'1','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4'})
            entry={'seed':seed,'pid':p.pid,'state':'running','started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'command':command}
            state['runs'].append(entry); state['state']='running'; write(status_path,state)
            code=p.wait()
        entry.update(exit_code=code,state='completed' if code==0 and (out/'completed.json').exists() else 'failed',ended_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
        if entry['state']=='failed':
            state['state']='failed'; write(status_path,state); return 1
        write(status_path,state)
    state['state']='completed'; write(status_path,state); return 0

if __name__=='__main__': sys.exit(main())
