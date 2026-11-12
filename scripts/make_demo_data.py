import argparse
import json
from pathlib import Path
import numpy as np


def make_demo(output):
    output=Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Choose an empty output directory')
    output.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(19)
    x,y,z=np.mgrid[:20,:24,:3]
    cases=[]
    for client in range(2):
        for split in ('train','val','test'):
            case_id=f'client{client}_{split}'
            labels=np.zeros(x.shape,dtype=np.int16)
            labels[(x-7-client)**2+(y-8)**2<20]=1
            labels[(x-13)**2+(y-16-client)**2<15]=2
            image=np.clip(0.1+rng.normal(0,0.02,x.shape)+0.35*(labels==1)+0.7*(labels==2)+0.04*client,0,1).astype(np.float32)
            annotated=[0,client+1] if split=='train' else [0,1,2]
            labels[~np.isin(labels,annotated)]=0
            np.save(output/(case_id+'_image.npy'),image,allow_pickle=False)
            np.save(output/(case_id+'_label.npy'),labels,allow_pickle=False)
            cases.append({'case_id':case_id,'patient_id':case_id,'client_id':str(client),'split':split,
                'image':case_id+'_image.npy','label':case_id+'_label.npy','annotated_classes':annotated,
                'modality':'normalized','spacing':[0.8,0.9,2.5]})
    path=output/'manifest.json'
    path.write_text(json.dumps({'description':'Artificial volumes, not medical data','cases':cases},indent=2)+'\n')
    return path


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    print(make_demo(parser.parse_args().output).resolve())
