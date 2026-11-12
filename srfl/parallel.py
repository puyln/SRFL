from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import time
import tempfile
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from .models import SRFL
from .data import SliceDataset, CaseShuffleSampler
from .usgr import USGRLoss
from .aggregation import trainable_state, load_trainable_state

_WORKER=None


def _initialize(config, initial, device_queue, stop_event):
    global _WORKER
    from .training import seed_everything
    device=device_queue.get(timeout=120)
    if device.startswith('cuda'):
        torch.cuda.set_device(device)
    seed_everything(config['seed'],config['training']['threads'])
    model=SRFL(config['model'],load_pretrained=False)
    model.load_state_dict(torch.load(initial,map_location='cpu',weights_only=True),strict=True)
    model.to(device)
    _WORKER=(config,model,USGRLoss(**config['usgr']).to(device),device,{},stop_event)


def _run_client(name,cases,state,round_index,client_index):
    from .training import seed_everything,train_local
    config,model,usgr,device,datasets,stop_event=_WORKER
    if stop_event.is_set():
        raise RuntimeError("Parallel client execution was cancelled")
    began=time.perf_counter()
    seed=config['seed']+round_index*100003+client_index
    seed_everything(seed,config['training']['threads'])
    load_trainable_state(model,{n:torch.from_numpy(v) for n,v in state.items()})
    if name not in datasets:
        datasets[name]=SliceDataset(cases,config,augment=config['augmentation']['enabled'])
    dataset=datasets[name]
    generator=torch.Generator().manual_seed(seed)
    sampler=CaseShuffleSampler(dataset,generator) if config['training']['casewise_sampling'] else None
    loader=DataLoader(dataset,batch_size=config['training']['batch_size'],shuffle=sampler is None,
                      sampler=sampler,num_workers=0,generator=generator)
    metrics=train_local(model,loader,usgr,config['training'],device,cases[0].get('_regions'),stop_event=stop_event)
    updated=trainable_state(model)
    metrics.update(device=device,worker_seconds=time.perf_counter()-began)

    model.zero_grad(set_to_none=True)
    if device.startswith('cuda'):
        torch.cuda.empty_cache()
    return {n:v.numpy() for n,v in updated.items()},metrics


class ClientPool:
    def __init__(self,config,model,devices):
        if not devices or (len(set(devices))!=len(devices) and devices!=['cpu']*len(devices)):
            raise ValueError('Parallel GPU devices must be nonempty and distinct')
        if not config['training'].get('client_seeded',False):
            raise ValueError('Parallel training requires training.client_seeded=true for schedule-independent RNG')
        if any(d!='cpu' and not d.startswith('cuda:') for d in devices):
            raise ValueError('Use explicit device names such as cuda:0')
        context=mp.get_context('spawn')
        self.queue=context.Queue()
        self.stop_event=context.Event()
        for device in devices:
            self.queue.put(device)
        self.temporary=tempfile.TemporaryDirectory(prefix='srfl_worker_init_')
        initial=str(Path(self.temporary.name)/'initial.pt')
        torch.save({n:t.detach().cpu() for n,t in model.state_dict().items()},initial)
        self.executor=ProcessPoolExecutor(max_workers=len(devices),mp_context=context,
            initializer=_initialize,initargs=(config,initial,self.queue,self.stop_event))
        self.config=config

    def __enter__(self):
        return self

    def __exit__(self,*exc):
        if exc[0] is not None:
            self.stop_event.set()
        self.executor.shutdown(wait=True,cancel_futures=True)
        self.queue.close()
        self.queue.join_thread()
        self.temporary.cleanup()

    def run(self,groups,datasets,state,round_index):
        state={n:v.numpy() for n,v in state.items()}
        names=list(groups)
        cap=self.config['training']['max_batches_per_client']
        batch=self.config['training']['batch_size']

        ordered=sorted(enumerate(names),key=lambda item: min((len(datasets[item[1]])+batch-1)//batch,cap or float('inf')),reverse=True)
        pending={self.executor.submit(_run_client,name,groups[name],state,round_index,i):name for i,name in ordered}
        result={}
        for future in as_completed(pending):
            result[pending[future]]=future.result()
        return ([{key:torch.from_numpy(value) for key,value in result[n][0].items()} for n in names],[len(datasets[n]) for n in names],
                {n:result[n][1] for n in names})
