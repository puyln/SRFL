from copy import deepcopy
import torch
from test_pipeline import config_for
from srfl.training import train


def test_parallel_matches_serial_and_resumes(tmp_path):
    cfg=config_for(tmp_path)
    cfg['training']['client_seeded']=True
    serial=train(cfg,tmp_path/'serial')
    parallel=train(cfg,tmp_path/'parallel',devices=['cpu','cpu'])
    first=deepcopy(cfg); first['training']['rounds']=1
    part=train(first,tmp_path/'first',devices=['cpu','cpu'])
    resumed=train(cfg,tmp_path/'resumed',resume=part,devices=['cpu','cpu'])
    expected=torch.load(serial,weights_only=True)['model_state']
    for checkpoint in (parallel,resumed):
        actual=torch.load(checkpoint,weights_only=True)['model_state']
        for name,value in expected.items():
            torch.testing.assert_close(value,actual[name],rtol=0,atol=0)


def test_slice_views_preserve_samples_and_augmentation(tmp_path):
    from srfl.data import SliceDataset,read_manifest
    cfg=config_for(tmp_path)
    cases=read_manifest(cfg['data']['manifest'],cfg['model']['num_classes'])
    cases=[c for c in cases if c['split']=='train'][:1]
    for axis in (0,1,2):
        cfg['data']['slice_axis']=axis
        candidate=deepcopy(cfg);candidate['data']['slice_views']=True
        original=SliceDataset(cases,cfg,augment=True)
        optimized=SliceDataset(cases,candidate,augment=True)
        for index in (0,len(original)-1):
            torch.manual_seed(45);a=original[index]
            torch.manual_seed(45);b=optimized[index]
            for x,y in zip(a,b):
                torch.testing.assert_close(x,y,rtol=0,atol=0)
