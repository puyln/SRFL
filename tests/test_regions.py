import json
import numpy as np
import torch
from srfl.regions import partial_region_loss,region_targets
from srfl.config import load_config
from srfl.data import read_manifest,SliceDataset,CaseShuffleSampler
from srfl.training import train
from srfl.cli import load_model
from srfl.evaluation import predict_volume,evaluate
from pathlib import Path


def test_overlap_and_missing_region_gradients():
    labels=torch.tensor([[[0,1],[2,-1]]])
    regions=[[1,2],[2],[3]]
    target=region_targets(labels,regions)
    assert target[0,0,1,0] and target[0,1,1,0]
    x=torch.zeros(1,3,2,2,requires_grad=True)
    loss=partial_region_loss(x,labels,torch.tensor([[1,1,0]],dtype=torch.bool),regions)
    loss.backward()
    assert x.grad[0,2].abs().sum()==0
    assert x.grad[:,:,1,1].abs().sum()==0
    assert x.grad[0,0,1,0]<0 and x.grad[0,1,1,0]<0


def test_region_pipeline_and_native_axis(tmp_path):
    cfg=load_config(Path(__file__).parents[1]/'configs/smoke.json')
    cfg['model'].update(num_classes=2,usgr_feature='decoder')
    cfg['training'].update(loss_mode='regions',rounds=1,casewise_sampling=True,max_batches_per_client=2)
    cfg['data']['expected_clients']=1
    cases=[]
    for split in ['train','val','test']:
        image=np.random.default_rng(7).random((4,5,3),dtype=np.float32)
        label=np.zeros((4,5,3),dtype=np.int16);label[1:3,1:4,:]=1;label[2,2,:]=2
        np.save(tmp_path/(split+'_image.npy'),image);np.save(tmp_path/(split+'_label.npy'),label)
        cases.append(dict(case_id=split,patient_id=split,client_id='a',split=split,image=split+'_image.npy',label=split+'_label.npy',modality='normalized',spacing=[1,1,2],annotated_classes=[0,1,2],annotated_regions=[0,1]))
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps({'regions':[[1,2],[2]],'cases':cases}))
    cfg['data']['manifest']=str(manifest)
    parsed=read_manifest(manifest,2)
    ds=SliceDataset(parsed[:1],cfg)
    indices=list(CaseShuffleSampler(ds,torch.Generator().manual_seed(0)))
    assert sorted(indices)==list(range(len(ds)))
    train(cfg,tmp_path/'run')
    model,loaded=load_model(tmp_path/'run/best.pt','cpu')
    for axis in [0,1,2]:
        loaded['data']['slice_axis']=axis
        prediction=predict_volume(model,image,'normalized',loaded,'cpu')
        assert prediction.shape==(2,4,5,3)
    summary=evaluate(model,parsed[2:],loaded,'cpu',tmp_path/'eval',True)
    assert set(summary['per_class'])=={'0','1'}
    assert np.load(tmp_path/'eval/test_prediction.npy').shape==(2,4,5,3)


def test_nifti_axis_reorientation_keeps_image_and_mask_aligned(tmp_path):
    import nibabel as nib
    from srfl.data import load_case,volume_shape
    data=np.arange(24,dtype=np.float32).reshape(2,3,4)
    label=(data>10).astype(np.int16)
    affine=np.array([[0,0,1,0],[0,-2,0,0],[3,0,0,0],[0,0,0,1]],dtype=float)
    image=tmp_path/'image.nii.gz';mask=tmp_path/'mask.nii.gz'
    nib.save(nib.Nifti1Image(data,affine),image);nib.save(nib.Nifti1Image(label,affine),mask)
    case={'image':str(image),'label':str(mask),'annotated_classes':[0,1]}
    x,y,spacing,out_affine=load_case(case)
    assert x.shape==volume_shape(case)==(4,3,2)
    assert np.array_equal(x>10,y>0)
    assert nib.aff2axcodes(out_affine)==('R','A','S')
    np.testing.assert_allclose(spacing,[1,2,3])
