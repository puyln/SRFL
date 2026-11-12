import argparse, copy, hashlib, json, os, random, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from srfl.models import SRFL
from srfl.data import read_manifest, SliceDataset, CaseShuffleSampler, load_case, normalize_volume, image_tensor
from srfl.regions import partial_region_loss
from srfl.usgr import USGRLoss
from srfl.aggregation import trainable_state, load_trainable_state, aggregate
from srfl.training import seed_everything, atomic_save, file_sha256, rng_state, restore_rng


def mean_states(states):
    return {n:torch.stack([s[n] for s in states]).mean(0) for n in states[0]}


def new_control(old, server, global_state, local_state, steps, lr):
    return {n:old[n]-server[n]+(global_state[n]-local_state[n])/(steps*lr) for n in old}


def local_train(model, dataset, config, method, global_state, control, server_control, device, seed, cap):
    settings=config['training']; lr=settings['learning_rate']
    seed_everything(seed,settings['threads'])
    gen=torch.Generator().manual_seed(seed)
    loader=DataLoader(dataset,batch_size=settings['batch_size'],sampler=CaseShuffleSampler(dataset,gen),num_workers=0,generator=gen)
    model.train()
    params={n:p for n,p in model.named_parameters() if p.requires_grad}
    reference={n:t.to(device) for n,t in global_state.items()} if method=='fedprox' else None
    correction={n:(server_control[n]-control[n]).to(device) for n in params} if method=='scaffold' else None
    optimizer=torch.optim.SGD(params.values(),lr=lr,momentum=0,weight_decay=0)
    usgr=USGRLoss(**config['usgr']).to(device) if method=='srfl' else None
    steps=min(cap,len(loader)); iterator=iter(loader); totals=np.zeros(3); count=0
    started=time.perf_counter()
    for step in range(steps):

        batch_seed=(seed+step*997)%2**32
        random.seed(batch_seed); np.random.seed(batch_seed); torch.manual_seed(batch_seed); torch.cuda.manual_seed_all(batch_seed)
        image,target,annotated=(v.to(device) for v in next(iterator))
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            output=model(image)
        seg=partial_region_loss(output.logits,target,annotated,dataset.cases[0]['_regions'],settings['ce_weight'],settings['dice_weight'])
        reg=usgr(output.usgr_features.float()).loss if usgr is not None else seg.new_zeros(())
        prox=0.5*config['benchmark']['prox_mu']*sum((p-reference[n]).square().sum() for n,p in params.items()) if reference is not None else seg.new_zeros(())
        loss=seg+settings['lambda_usgr']*reg+prox
        if not torch.isfinite(loss): raise FloatingPointError('Nonfinite objective')
        loss.backward()
        with torch.no_grad():
            if correction is not None:
                for n,p in params.items():
                    if p.grad is None: p.grad=torch.zeros_like(p)
                    p.grad.add_(correction[n])
            if not torch.stack([torch.isfinite(p.grad).all() for p in params.values() if p.grad is not None]).all():
                raise FloatingPointError('Nonfinite gradient')
        optimizer.step()
        totals+=torch.stack([seg.detach(),reg.detach(),prox.detach()]).cpu().numpy()*len(image); count+=len(image)
        if step==0 or step+1==steps:
            print(json.dumps({'event':'local_step','method':method,'step':step+1,'segmentation_loss':float(seg.detach()),'seconds':time.perf_counter()-started}),flush=True)
    torch.cuda.synchronize(device)
    local=trainable_state(model)
    updated=new_control(control,server_control,global_state,local,steps,lr) if method=='scaffold' else None
    return local,updated,{'steps':steps,'samples':count,'segmentation_loss':float(totals[0]/count),'usgr_loss':float(totals[1]/count),'proximal_loss':float(totals[2]/count),'seconds':time.perf_counter()-started}


def counts_to_score(predicted, truth, intersection):

    return 2*intersection/(predicted+truth) if truth>0 else None


@torch.no_grad()
def validate_dice(model,cases,config,device,path):
    model.eval(); records=[]; started=time.perf_counter(); regions=cases[0]['_regions']
    for index,case in enumerate(cases):
        image,label,_,_=load_case(case)
        image=normalize_volume(image,case['modality'],config['preprocessing'])
        stack=np.moveaxis(image,config['data']['slice_axis'],0)
        labels=np.moveaxis(label,config['data']['slice_axis'],0)
        channels=case['annotated_regions']; counts=np.zeros((len(channels),3),dtype=np.int64)
        for start in range(0,len(stack),config['training']['batch_size']):
            end=min(start+config['training']['batch_size'],len(stack))
            batch=torch.stack([image_tensor(s,config['model']['image_size']) for s in stack[start:end]]).to(device)
            with torch.autocast('cuda',dtype=torch.bfloat16): logits=model(batch).logits[:,channels]
            prediction=F.interpolate(logits.float(),size=stack.shape[1:],mode='bilinear',align_corners=False)>0
            target=torch.as_tensor(np.ascontiguousarray(labels[start:end]),device=device)
            valid=target!=-1
            for j,c in enumerate(channels):
                truth=torch.zeros_like(valid)
                for v in regions[c]: truth|=(target==v)
                pm=prediction[:,j]&valid; tm=truth&valid
                counts[j]+=torch.stack([pm.sum(),tm.sum(),(pm&tm).sum()]).cpu().numpy()
        for j,c in enumerate(channels):
            p,t,i=map(int,counts[j]); records.append({'case_id':case['case_id'],'region':c,'prediction_voxels':p,'truth_voxels':t,'intersection_voxels':i,'dice':counts_to_score(p,t,i)})
        if (index+1)%10==0 or index+1==len(cases):
            print(json.dumps({'event':'validation_case','completed':index+1,'total':len(cases),'seconds':time.perf_counter()-started}),flush=True)
    means={str(c):float(np.mean([r['dice'] for r in records if r['region']==c and r['dice'] is not None])) for c in range(len(regions)) if any(r['region']==c and r['dice'] is not None for r in records)}
    if len(means)!=len(regions): raise ValueError('Validation lacks positive ground truth for an output region')
    result={'macro_dice':float(np.mean(list(means.values()))),'region_dice':means,'case_count':len(cases),'seconds':time.perf_counter()-started,
            'empty_reference_pairs':sum(r['truth_voxels']==0 for r in records),'empty_reference_false_positive_pairs':sum(r['truth_voxels']==0 and r['prediction_voxels']>0 for r in records)}
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps({'summary':result,'per_case':records},indent=2,allow_nan=False))
    return result


def trajectory_summary(records,budget):
    vals=[r for r in records if r.get('validation') is not None]
    out={'completed_round':records[-1]['round'],'budget':budget,'validation_points':len(vals),'attainment':{}}
    for threshold in (0.5,0.6,0.7):
        hit=next((r for r in vals if r['validation']['macro_dice']>=threshold),None)
        out['attainment'][str(threshold)]={'round':hit['round'],'seconds':hit['elapsed_seconds']} if hit else {'status':'not_reached' if records[-1]['round']==budget else 'pending'}
    if vals and vals[-1]['round']==budget:
        x=[r['round'] for r in vals]; y=[r['validation']['macro_dice'] for r in vals]
        out['nauc_percent']=float(np.trapezoid(y,x)/budget*100)
        out['endpoint_dice_percent']=y[-1]*100
    return out


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',required=True); ap.add_argument('--method',choices=['fedavg','fedprox','scaffold','srfl'],required=True)
    ap.add_argument('--gpu',type=int,required=True); ap.add_argument('--seed',type=int,required=True); ap.add_argument('--out',required=True)
    ap.add_argument('--smoke',action='store_true'); ap.add_argument('--resume',action='store_true'); args=ap.parse_args()
    cfg=json.loads(Path(args.config).read_text()); cfg['seed']=args.seed
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    device=f'cuda:{args.gpu}'; torch.cuda.set_device(args.gpu); torch.set_num_interop_threads(1)
    seed_everything(args.seed,cfg['training']['threads'])
    cases=read_manifest(cfg['data']['manifest'],cfg['model']['num_classes']); val=[c for c in cases if c['split']=='val']
    names=sorted({c['client_id'] for c in cases if c['split']=='train'})
    groups={name:[c for c in cases if c['split']=='train' and c['client_id']==name] for name in names}
    if len(groups)!=cfg['data']['expected_clients']: raise ValueError('Client count mismatch')
    if args.smoke:
        groups={n:cs[:1] for n,cs in list(groups.items())[:2]}; val=[]
    model=SRFL(cfg['model'],load_pretrained=True).to(device)
    initial=trainable_state(model)
    controls={n:{p:torch.zeros_like(t) for p,t in initial.items()} for n in groups} if args.method=='scaffold' else {}
    server_control={p:torch.zeros_like(t) for p,t in initial.items()}
    init_hash=hashlib.sha256(b''.join(t.numpy().tobytes() for t in initial.values())).hexdigest()
    cfg_hash=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest(); manifest_hash=file_sha256(cfg['data']['manifest'])
    provenance={'initial_trainable_sha256':init_hash,'config_sha256':cfg_hash,'manifest_sha256':manifest_hash,'sam_sha256':file_sha256(cfg['model']['sam_checkpoint']),
                'torch':str(torch.__version__),'gpu':torch.cuda.get_device_name(args.gpu),'gpu_index':args.gpu,'method':args.method,'seed':args.seed,'smoke':args.smoke,'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    start_round=0; elapsed_before=0.; records=[]; best=-1.
    if args.resume:
        saved=torch.load(out/'last.pt',weights_only=True,map_location='cpu')
        if saved['config_sha256']!=cfg_hash or saved['manifest_sha256']!=manifest_hash: raise ValueError('Resume provenance mismatch')
        load_trainable_state(model,saved['trainable']); controls=saved['controls']; server_control=saved['server_control']; start_round=saved['round']; elapsed_before=saved['elapsed_seconds']; best=saved['best_score']; restore_rng(saved['rng'])
        records=[json.loads(l) for l in (out/'metrics.jsonl').read_text().splitlines() if json.loads(l)['round']<=start_round]
        (out/'metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    elif (out/'metrics.jsonl').exists(): raise ValueError('Refusing to overwrite an existing run')
    else:
        (out/'provenance.json').write_text(json.dumps(provenance,indent=2)); (out/'config.json').write_text(json.dumps(cfg,indent=2))
    datasets={n:SliceDataset(cs,cfg,augment=True) for n,cs in groups.items()}
    def log(record):
        records.append(record)
        with (out/'metrics.jsonl').open('a') as h: h.write(json.dumps(record,allow_nan=False)+'\n')
        (out/'summary.json').write_text(json.dumps(trajectory_summary(records,cfg['training']['rounds']),indent=2))
    if not args.resume and val:
        baseline=validate_dice(model,val,cfg,device,out/'validation'/'round_0000.json')
        log({'round':0,'elapsed_seconds':0.,'validation':baseline})
    started=time.perf_counter(); rounds=1 if args.smoke else cfg['training']['rounds']; cap=2 if args.smoke else cfg['training']['max_batches_per_client']
    for rnd in range(start_round+1,rounds+1):
        global_state=trainable_state(model); states=[]; new_controls={}; client_metrics={}
        round_start=time.perf_counter()
        for index,(name,ds) in enumerate(datasets.items()):
            print(json.dumps({'event':'client_start','round':rnd,'client':name,'method':args.method}),flush=True)
            load_trainable_state(model,global_state)
            local,control,metrics=local_train(model,ds,cfg,args.method,global_state,controls.get(name),server_control,device,(args.seed+(rnd-1)*100003+index)%2**32,cap)
            states.append(local); client_metrics[name]=metrics
            if control is not None: new_controls[name]=control
        agg_start=time.perf_counter()
        if args.method=='srfl': combined,diagnostic=aggregate(global_state,states,[1]*len(states),cfg['aggregation'])
        else: combined=mean_states(states); diagnostic={'method':args.method,'weights':[1/len(states)]*len(states)}
        if args.method=='scaffold': controls=new_controls; server_control=mean_states(list(controls.values()))
        load_trainable_state(model,combined); torch.cuda.synchronize(device)
        diagnostic['seconds']=time.perf_counter()-agg_start
        result=validate_dice(model,val,cfg,device,out/'validation'/f'round_{rnd:04d}.json') if val and (rnd%cfg['training']['validate_every']==0 or rnd==rounds) else None
        elapsed=elapsed_before+time.perf_counter()-started
        log({'round':rnd,'elapsed_seconds':elapsed,'round_seconds':time.perf_counter()-round_start,'clients':client_metrics,'aggregation':diagnostic,'validation':result})
        if result and result['macro_dice']>best:
            best=result['macro_dice']; atomic_save({'trainable':combined,'round':rnd,'validation':result,'config':cfg},out/'best.pt')
        atomic_save({'trainable':combined,'controls':controls,'server_control':server_control,'round':rnd,'elapsed_seconds':elapsed_before+time.perf_counter()-started,'best_score':best,'rng':rng_state(),'config_sha256':cfg_hash,'manifest_sha256':manifest_hash},out/'last.pt')
        print(json.dumps({'event':'round_complete','round':rnd,'method':args.method,'seconds':time.perf_counter()-round_start,'validation':result}),flush=True)
    (out/'completed.json').write_text(json.dumps({'completed_round':rounds,'smoke':args.smoke,'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}))

if __name__=='__main__': main()
