from copy import deepcopy
from pathlib import Path
import importlib.util
import numpy as np
import torch
import pytest
from srfl.config import load_config
from srfl.training import train
from srfl.cli import load_model
from srfl.data import read_manifest
from srfl.evaluation import evaluate, predict_volume

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('make_demo_data',ROOT/'scripts/make_demo_data.py')
demo=importlib.util.module_from_spec(spec);spec.loader.exec_module(demo)


def config_for(tmp_path):
    manifest=demo.make_demo(tmp_path/'data')
    cfg=load_config(ROOT/'configs/smoke.json')
    cfg['data']['manifest']=str(manifest)
    return cfg


def test_resume_matches_uninterrupted_training_and_native_prediction(tmp_path):
    cfg=config_for(tmp_path)
    uninterrupted=train(cfg,tmp_path/'full')
    first=deepcopy(cfg);first['training']['rounds']=1
    part=train(first,tmp_path/'first')
    resumed=train(cfg,tmp_path/'resumed',resume=part)
    original=torch.load(uninterrupted,weights_only=True)
    continued=torch.load(resumed,weights_only=True)
    for name,value in original['model_state'].items():
        torch.testing.assert_close(value,continued['model_state'][name],rtol=0,atol=0)
    model,loaded=load_model(tmp_path/'resumed/best.pt','cpu')
    cases=read_manifest(loaded['data']['manifest'],3)
    test_cases=[c for c in cases if c['split']=='test']
    summary=evaluate(model,test_cases,loaded,'cpu',tmp_path/'evaluation',True,'mm')
    assert summary['macro_dice'] is not None and 0<=summary['macro_dice']<=1
    prediction=np.load(tmp_path/'evaluation'/f"{test_cases[0]['case_id']}_prediction.npy")
    assert prediction.shape==np.load(test_cases[0]['image']).shape
    assert set(np.unique(prediction)).issubset({0,1,2})
    assert (tmp_path/'evaluation/per_case.csv').is_file()


@pytest.mark.parametrize('method',['fedavg','centralized'])
def test_alternative_training_modes_run(tmp_path,method):
    cfg=config_for(tmp_path)
    cfg['training']['rounds']=1
    cfg['training']['lambda_usgr']=0
    cfg['model']['adapters']=False
    cfg['aggregation']['method']=method
    checkpoint=train(cfg,tmp_path/'run')
    assert checkpoint.is_file()
