import json
import math
from pathlib import Path
import random
import time
import hashlib
import platform
import resource
from contextlib import ExitStack
import numpy as np
import torch
from torch.utils.data import DataLoader
from .models import SRFL
from .data import read_manifest, SliceDataset, CaseShuffleSampler
from .losses import partial_segmentation_loss
from .regions import partial_region_loss
from .usgr import USGRLoss
from .aggregation import trainable_state, load_trainable_state, aggregate
from .evaluation import evaluate


def seed_everything(seed, threads):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True


def rng_state():
    state=np.random.get_state()
    return {'python':random.getstate(),'numpy':[state[0],state[1].tolist(),state[2],state[3],state[4]],
            'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state['python'])
    npstate=state['numpy']
    np.random.set_state((npstate[0],np.asarray(npstate[1],dtype=np.uint32),*npstate[2:]))
    torch.set_rng_state(state['torch'])
    if torch.cuda.is_available() and state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])


def atomic_save(payload,path):
    temporary=path.with_suffix('.tmp')
    torch.save(payload,temporary)
    temporary.replace(path)


def file_sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def train_local(model, loader, usgr, settings, device, regions=None, stop_event=None):
    model.train()
    optimizer=torch.optim.Adam(model.trainable_parameters(),lr=settings['learning_rate'],betas=tuple(settings['betas']),
                               fused=True if settings.get('fused_adam',False) and str(device).startswith('cuda') else None)
    sums={'loss':0.,'segmentation_loss':0.,'usgr_loss':0.,'coupling_mass':0.}
    samples=0
    use_cuda=str(device).startswith('cuda')
    if use_cuda:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started=time.perf_counter()
    batches=0
    data_wait_seconds=0.
    last_step_end=started
    for _ in range(settings['local_epochs']):
        for image,target,annotated in loader:
            if stop_event is not None and stop_event.is_set():
                raise RuntimeError("Peer worker failed; cancelling this client")
            data_wait_seconds+=time.perf_counter()-last_step_end
            image,target,annotated=(value.to(device) for value in (image,target,annotated))
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda' if use_cuda else 'cpu',dtype=torch.bfloat16,enabled=settings['amp'] and use_cuda):
                output=model(image)
            if settings['loss_mode']=='regions':
                seg=partial_region_loss(output.logits,target,annotated,regions,
                    settings['ce_weight'],settings['dice_weight'])
            else:
                seg=partial_segmentation_loss(output.logits.float(),target,annotated,mode=settings['loss_mode'],
                    ce_weight=settings['ce_weight'],dice_weight=settings['dice_weight'])
            if settings['lambda_usgr'] > 0:
                result=usgr(output.usgr_features.float())
                uniformity=result.loss
                mass=result.coupling.sum()
            else:
                uniformity=seg.new_zeros(())
                mass=0.
            loss=seg+settings['lambda_usgr']*uniformity
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite local objective')
            loss.backward()
            if not torch.stack([torch.isfinite(p.grad).all() for p in model.trainable_parameters() if p.grad is not None]).all():
                raise FloatingPointError('Non-finite parameter gradient')
            optimizer.step()
            batches+=1
            count=len(image)
            samples+=count
            values=torch.stack([loss.detach(),seg.detach(),uniformity.detach(),torch.as_tensor(mass,device=device)]).cpu().tolist()
            for key,value in zip(sums,values):
                sums[key]+=value*count
            if batches==1 or batches%settings['log_every']==0:
                print(json.dumps({'local_batch':batches,'loss':values[0],'elapsed_seconds':time.perf_counter()-started}),flush=True)
            last_step_end=time.perf_counter()
            if settings['max_batches_per_client'] is not None and batches>=settings['max_batches_per_client']:
                break
        if settings['max_batches_per_client'] is not None and batches>=settings['max_batches_per_client']:
            break
    if use_cuda:
        torch.cuda.synchronize(device)
    return {**{key:value/samples for key,value in sums.items()},'sample_updates':samples,'batches':batches,
            'seconds':time.perf_counter()-started,'data_wait_seconds':data_wait_seconds,'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(device) if use_cuda else 0}


def train(config, output, device='cpu', resume=None, devices=None):
    if devices:
        device='cpu'
    output=Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output directory must be empty; resume into a new directory')
    if str(device).startswith('cuda') and not torch.cuda.is_available():
        raise ValueError('CUDA is unavailable')
    cases=read_manifest(config['data']['manifest'],config['model']['num_classes'])
    regions=cases[0].get('_regions')
    if (config['training']['loss_mode']=='regions') != (regions is not None):
        raise ValueError('Region manifest and region loss must be used together')
    training=[c for c in cases if c['split']=='train']
    validation=[c for c in cases if c['split']=='val']
    groups={client:[c for c in training if c['client_id']==client] for client in sorted({c['client_id'] for c in training})}
    if not groups:
        raise ValueError('No training clients in manifest')
    expected=config['data']['expected_clients']
    if expected is not None and len(groups)!=expected:
        raise ValueError(f'Expected {expected} clients, found {len(groups)}')
    if config['aggregation']['method']=='centralized':
        groups={'pooled':training}
    manifest_hash=file_sha256(config['data']['manifest'])
    snapshot=torch.load(resume,map_location='cpu',weights_only=True) if resume else None
    if snapshot and snapshot.get('kind')!='training':
        raise ValueError('Resume requires last.pt, not an inference-only best.pt')
    if snapshot and snapshot['manifest_sha256']!=manifest_hash:
        raise ValueError('Manifest changed since the saved checkpoint')
    seed_everything(config['seed'],config['training']['threads'])
    model=SRFL(config['model'],load_pretrained=not bool(snapshot)).to(device)
    usgr=USGRLoss(**config['usgr']).to(device)
    start=0
    best_score=None
    best_state=None
    best_round=None
    if snapshot:
        model.load_state_dict(snapshot['model_state'],strict=True)
        start=snapshot['completed_round']
        best_score=snapshot['best_score']
        best_state=snapshot['best_trainable']
        best_round=snapshot['best_round']
        restore_rng(snapshot['rng'])
    if start>=config['training']['rounds']:
        raise ValueError('Requested total rounds must exceed completed rounds')
    output.mkdir(parents=True,exist_ok=True)
    (output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    environment={'python':platform.python_version(),'torch':str(torch.__version__),'numpy':np.__version__,
                 'device':str(device),'parameter_counts':model.parameter_counts(),'manifest_sha256':manifest_hash,
                 'parallel_devices':devices,'resume_from_round':start,'client_sampling_unit':'2D slices','original_clients':len({c['client_id'] for c in training})}
    if not snapshot and config['model']['backbone']!='tiny':
        environment['sam_checkpoint_sha256']=file_sha256(config['model']['sam_checkpoint'])
    (output/'environment.json').write_text(json.dumps(environment,indent=2)+'\n')
    datasets={name:SliceDataset(group,config,augment=config['augmentation']['enabled']) for name,group in groups.items()}
    with ExitStack() as stack:
        log=stack.enter_context((output/'metrics.jsonl').open('w'))
        pool=None
        if devices:
            from .parallel import ClientPool
            pool=stack.enter_context(ClientPool(config,model,devices))
        for round_index in range(start,config['training']['rounds']):
            round_started=time.perf_counter()
            global_state=trainable_state(model)
            locals_,counts,metrics=[],[],{}
            if pool:
                locals_,counts,metrics=pool.run(groups,datasets,global_state,round_index)
            else:
                for client_index,(name,dataset) in enumerate(datasets.items()):
                    if config['training'].get('client_seeded',False):
                        seed_everything(config['seed']+round_index*100003+client_index,config['training']['threads'])
                    load_trainable_state(model,global_state)
                    generator=torch.Generator().manual_seed(config['seed']+round_index*100003+client_index)
                    print(json.dumps({'round':round_index+1,'client':name,'slices':len(dataset)}),flush=True)
                    sampler=CaseShuffleSampler(dataset,generator) if config['training']['casewise_sampling'] else None
                    loader=DataLoader(dataset,batch_size=config['training']['batch_size'],shuffle=sampler is None,
                                      sampler=sampler,num_workers=0,generator=generator)
                    metrics[name]=train_local(model,loader,usgr,config['training'],device,regions)
                    locals_.append(trainable_state(model))
                    counts.append(len(dataset))
            if str(device).startswith('cuda'):
                torch.cuda.synchronize(device)
            started=time.perf_counter()
            combined,agg=aggregate(global_state,locals_,counts,config['aggregation'])
            load_trainable_state(model,combined)
            if str(device).startswith('cuda'):
                torch.cuda.synchronize(device)
            agg['seconds']=time.perf_counter()-started
            agg['process_lifetime_peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if platform.system()=='Darwin' else 1024)
            agg['logical_upload_bytes']=sum(t.numel()*t.element_size() for state in locals_ for t in state.values())
            agg['logical_download_bytes']=len(locals_)*sum(t.numel()*t.element_size() for t in global_state.values())
            agg['payload_policy']='Tensor payload estimate in virtual federation; not measured network traffic'
            print(json.dumps({'round':round_index+1,'aggregation_seconds':agg['seconds'],'optimality_gap':agg.get('optimality_gap')}),flush=True)
            val=None
            if validation and ((round_index+1)%config['training']['validate_every']==0 or round_index+1==config['training']['rounds']):
                evaluation_device=devices[0] if devices else device
                model.to(evaluation_device)
                val=evaluate(model,validation,config,evaluation_device,output/'validation'/f'round_{round_index+1:04d}',distance_unit='mm')
                model.to(device)
                if devices and str(evaluation_device).startswith('cuda'):
                    torch.cuda.empty_cache()
                score=val['macro_dice']
                if score is not None and (best_score is None or score>best_score):
                    best_score,best_state,best_round=score,trainable_state(model),round_index+1
                    atomic_save({'kind':'inference','model_state':{n:t.detach().cpu() for n,t in model.state_dict().items()},
                        'config':config,'selected_round':best_round,'validation_dice':best_score},output/'best.pt')
            record={'round':round_index+1,'clients':metrics,'aggregation':agg,'validation':val,'round_seconds_before_checkpoint':time.perf_counter()-round_started}
            log.write(json.dumps(record,allow_nan=False)+'\n')
            log.flush()
            print(json.dumps({'round':round_index+1,'mean_client_loss':sum(m['loss'] for m in metrics.values())/len(metrics),
                              'validation_dice':None if val is None else val['macro_dice'],
                              'applied_global_step':agg.get('applied_step')},allow_nan=False),flush=True)
            payload={'kind':'training','model_state':{n:t.detach().cpu() for n,t in model.state_dict().items()},
                     'config':config,'completed_round':round_index+1,'rng':rng_state(),'manifest_sha256':manifest_hash,
                     'best_score':best_score,'best_trainable':best_state,'best_round':best_round}
            atomic_save(payload,output/'last.pt')
    if best_state is not None:
        load_trainable_state(model,best_state)
        atomic_save({'kind':'inference','model_state':{n:t.detach().cpu() for n,t in model.state_dict().items()},
                     'config':config,'selected_round':best_round,'validation_dice':best_score},output/'best.pt')
    return output/'last.pt'
