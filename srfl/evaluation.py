import csv
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from .data import load_case, normalize_volume, image_tensor
from .metrics import binary_metrics, summarize


@torch.no_grad()
def predict_volume(model, image, modality, config, device):
    image = normalize_volume(image, modality, config['preprocessing'])
    axis = config['data']['slice_axis']
    stack = np.moveaxis(image, axis, 0)
    predictions = []
    model.eval()
    batch_size = config['training']['batch_size']
    for start in range(0,len(stack),batch_size):
        batch = torch.stack([image_tensor(s,config['model']['image_size']) for s in stack[start:start+batch_size]]).to(device)
        with torch.autocast(device_type='cuda' if str(device).startswith('cuda') else 'cpu',dtype=torch.bfloat16,enabled=config['training']['amp'] and str(device).startswith('cuda')):
            logits = model(batch).logits
        logits=logits.float()

        logits = F.interpolate(logits,size=stack.shape[1:],mode='bilinear',align_corners=False)
        if config['training']['loss_mode']=='regions':
            predictions.append((logits>0).cpu().numpy())
        else:
            predictions.append(logits.argmax(1).cpu().numpy())
    if config['training']['loss_mode']=='regions':
        stack=np.concatenate(predictions)
        return np.moveaxis(np.moveaxis(stack,1,0),1,axis+1).astype(np.uint8)
    return np.moveaxis(np.concatenate(predictions),0,axis).astype(np.int16)


def evaluate(model, cases, config, device, output=None, save_predictions=False, distance_unit='mm'):
    if not cases:
        raise ValueError('Evaluation split has no cases')
    if distance_unit not in ('mm','voxel'):
        raise ValueError('distance_unit must be mm or voxel')
    if output is not None:
        output=Path(output)
        output.mkdir(parents=True,exist_ok=True)
    records=[]
    for case in cases:
        image,label,spacing,affine=load_case(case)
        prediction=predict_volume(model,image,case['modality'],config,device)
        regions=case.get('_regions')
        for class_id in (case['annotated_regions'] if regions else case['annotated_classes']):
            if regions is None and class_id == 0:
                continue
            pm=prediction[class_id].astype(bool) if regions else prediction==class_id
            tm=np.isin(label,regions[class_id]) if regions else label==class_id
            scores=binary_metrics(pm,tm,
                spacing if distance_unit=='mm' else np.ones(3),valid=label!=-1)
            records.append({'case_id':case['case_id'],'patient_id':case['patient_id'],
                            'client_id':case['client_id'],'class_id':class_id,'distance_unit':distance_unit,**scores})
        if save_predictions and output is not None:
            if affine is not None:
                import nibabel as nib
                saved=np.moveaxis(prediction,0,-1) if regions else prediction
                nib.save(nib.Nifti1Image(saved,affine),output/(case['case_id']+'_prediction.nii.gz'))
            else:
                np.save(output/(case['case_id']+'_prediction.npy'),prediction,allow_pickle=False)
    summary=summarize(records)
    summary['region_definitions']=cases[0].get('_regions')
    summary['prediction_format']='region channels on last NIfTI axis; independent threshold 0.5' if cases[0].get('_regions') else 'class IDs'
    summary['distance_unit']=distance_unit
    summary['dice_scale']='0 to 1'
    summary['hd_policy']='Undefined distances are null; finite means and failure counts are both reported.'
    if output is not None:
        (output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
        with (output/'per_case.csv').open('w',newline='') as handle:
            fields=['case_id','patient_id','client_id','class_id','distance_unit','dice','hd','hd95','status']
            writer=csv.DictWriter(handle,fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)
    return summary
