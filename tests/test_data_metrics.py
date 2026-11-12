import json
import numpy as np
import pytest
import nibabel as nib
from srfl.metrics import binary_metrics, summarize
from srfl.data import read_manifest, load_case, normalize_volume, SliceDataset
from srfl.config import merge, DEFAULTS


def test_dice_and_spacing_aware_hausdorff_analytic_shift():
    a=np.zeros((5,5,5),bool); b=a.copy()
    a[1,2,2]=1; b[2,2,2]=1
    result=binary_metrics(a,b,[2.,1.,1.])
    assert result['dice']==0 and result['hd']==2. and result['hd95']==2.
    perfect=binary_metrics(a,a,[2.,1.,1.])
    assert perfect['dice']==1. and perfect['hd']==0


def test_empty_and_partial_masks_are_not_silently_good_distances():
    a=np.zeros((3,3,3),bool); b=a.copy(); b[1,1,1]=True
    assert binary_metrics(a,a,[1,1,1])['status']=='both_empty'
    result=binary_metrics(a,b,[1,1,1])
    assert result['dice']==0 and result['hd'] is None and result['status']=='one_empty'
    valid=np.ones_like(a); valid[0]=False
    assert binary_metrics(b,b,[1,1,1],valid)['hd'] is None
    assert binary_metrics(b,b,[1,1,1],np.zeros_like(a))['dice'] is None


def test_ct_preprocessing_uses_paper_window():
    value=normalize_volume(np.array([-1000.,-500.,250.,1000.,2000.]),'ct',DEFAULTS['preprocessing'])
    np.testing.assert_allclose(value,[0,0,0.5,1,1])


def test_manifest_rejects_patient_leakage(tmp_path):
    image=tmp_path/'a.npy'; label=tmp_path/'b.npy'
    np.save(image,np.zeros((3,3,3)));np.save(label,np.zeros((3,3,3)))
    second=tmp_path/'c.npy'; np.save(second,np.zeros((3,3,3)))
    base={'patient_id':'same','client_id':'0','image':str(image),'label':str(label),'annotated_classes':[0,1],
          'modality':'ct','spacing':[1,1,1]}
    cases=[{**base,'case_id':'a','split':'train'},{**base,'case_id':'b','split':'test','image':str(second)}]
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps({'cases':cases}))
    with pytest.raises(ValueError,match='boundary'):
        read_manifest(manifest,2)


def test_nifti_affine_and_labels_checked(tmp_path):
    image=tmp_path/'image.nii.gz'; label=tmp_path/'label.nii.gz'
    affine=np.diag([1.,2.,3.,1.])
    nib.save(nib.Nifti1Image(np.ones((4,5,6),np.float32),affine),image)
    nib.save(nib.Nifti1Image(np.zeros((4,5,6),np.int16),affine),label)
    case={'image':str(image),'label':str(label),'annotated_classes':[0,1]}
    _,_,spacing,actual_affine=load_case(case)
    np.testing.assert_allclose(spacing,[1,2,3]);np.testing.assert_allclose(actual_affine,affine)
    nib.save(nib.Nifti1Image(np.zeros((4,5,6),np.int16),np.eye(4)),label)
    with pytest.raises(ValueError,match='affines'):
        load_case(case)


def test_local_labels_map_to_global_ids(tmp_path):
    image=tmp_path/'image.npy';label=tmp_path/'label.npy'
    np.save(image,np.ones((3,3,3),dtype=np.float32))
    raw=np.zeros((3,3,3),dtype=np.int16);raw[1]=1
    np.save(label,raw)
    case={'image':str(image),'label':str(label),'spacing':[1,1,1],
          'annotated_classes':[0,4],'label_map':{'0':0,'1':4}}
    _,mapped,_,_=load_case(case)
    assert set(np.unique(mapped))=={0,4}
    assert np.all(mapped[1]==4)
    case['label_map']={'0':0}
    with pytest.raises(ValueError,match='raw IDs'):
        load_case(case)
