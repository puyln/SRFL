import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .config import load_config, validate
from .training import train, seed_everything
from .models import SRFL
from .data import read_manifest
from .evaluation import evaluate, predict_volume


def train_main():
    parser=argparse.ArgumentParser(description='Train the SRFL model on a medical-volume manifest')
    parser.add_argument('--config',type=Path)
    parser.add_argument('--manifest',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--devices',nargs='+',help='Parallel client devices, e.g. cuda:0 cuda:1 cuda:2 cuda:3')
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--rounds',type=int,help='Total number of rounds, including completed rounds')
    args=parser.parse_args()
    if args.resume:
        if args.config:
            parser.error('--config cannot override a resumed run; only --rounds may change')
        snapshot=torch.load(args.resume,map_location='cpu',weights_only=True)
        config=snapshot['config']
    else:
        if not args.config:
            parser.error('--config is required for a new run')
        config=load_config(args.config)
    if args.manifest:
        config['data']['manifest']=str(args.manifest.resolve())
    if args.rounds is not None:
        config['training']['rounds']=args.rounds
    validate(config)
    train(config,args.output,args.device,args.resume,devices=args.devices)


def load_model(checkpoint,device):
    snapshot=torch.load(checkpoint,map_location='cpu',weights_only=True)
    config=validate(snapshot['config'])
    seed_everything(config['seed'],config['training']['threads'])
    model=SRFL(config['model'],load_pretrained=False)
    model.load_state_dict(snapshot['model_state'],strict=True)
    return model.to(device).eval(), config


def evaluate_main():
    parser=argparse.ArgumentParser(description='Evaluate SRFL predictions on native 3D volumes')
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--manifest',type=Path)
    parser.add_argument('--split',choices=['val','test'],default='test')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--distance-unit',choices=['mm','voxel'],default='mm')
    parser.add_argument('--save-predictions',action='store_true')
    args=parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Output directory must be empty')
    model,config=load_model(args.checkpoint,args.device)
    cases=read_manifest(args.manifest or config['data']['manifest'],config['model']['num_classes'])
    summary=evaluate(model,[c for c in cases if c['split']==args.split],config,args.device,
        args.output,args.save_predictions,args.distance_unit)
    print(json.dumps(summary,indent=2,allow_nan=False))


def predict_main():
    parser=argparse.ArgumentParser(description='Segment one unlabeled 3D NPY or NIfTI volume')
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--image',type=Path,required=True)
    parser.add_argument('--modality',choices=['ct','mri','normalized'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cpu')
    args=parser.parse_args()
    if args.output.exists():
        parser.error('Output exists; choose a new path')
    nifti=None
    if str(args.image).endswith(('.nii','.nii.gz')):
        import nibabel as nib
        nifti=nib.as_closest_canonical(nib.load(args.image))
        image=np.asarray(nifti.dataobj,dtype=np.float32)
        if not str(args.output).endswith(('.nii','.nii.gz')):
            parser.error('NIfTI input requires a NIfTI output to retain spatial geometry')
    elif args.image.suffix=='.npy':
        image=np.load(args.image,allow_pickle=False)
        if args.output.suffix!='.npy':
            parser.error('NPY input requires an .npy output')
    else:
        parser.error('Input must be .npy, .nii or .nii.gz')
    if image.ndim!=3 or not np.isfinite(image).all():
        parser.error('Input must be a finite single-channel 3D volume')
    model,config=load_model(args.checkpoint,args.device)
    prediction=predict_volume(model,image,args.modality,config,args.device)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if nifti is not None:
        import nibabel as nib
        if config['training']['loss_mode']=='regions':
            prediction=np.moveaxis(prediction,0,-1)
        header=nifti.header.copy()
        header.set_data_dtype(np.int16)
        nib.save(nib.Nifti1Image(prediction,nifti.affine,header),args.output)
    else:
        np.save(args.output,prediction,allow_pickle=False)
    print(args.output.resolve())
