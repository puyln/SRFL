import copy
import json
from pathlib import Path
import math

DEFAULTS = {
 'seed':20260924,
 'model':{'backbone':'vit_b','image_size':1024,'num_classes':12,'sam_checkpoint':'../checkpoints/sam_vit_b_01ec64.pth',
          'adapters':True,'adapter_dim':64,'encoder_channels':32,'encoder_depth':2,
          'decoder_channels':128,'decoder_heads':4,'decoder_depth':2,'query_grid':16,
          'sam_normalization':True,'usgr_feature':'encoder','fusion':'multiply','attention_backend':'original'},
 'data':{'manifest':'../data/manifest.json','slice_axis':2,'cache_volumes':1,'expected_clients':7,'slice_views':False},
 'preprocessing':{'ct_window':[-500.,1000.],'mri_percentiles':[0.5,99.5]},
 'augmentation':{'enabled':True,'rotation_degrees':15.,'scale_range':[0.9,1.1],'intensity_jitter':0.1},
 'training':{'rounds':200,'local_epochs':1,'batch_size':4,'learning_rate':1e-4,'betas':[0.9,0.999],
             'lambda_usgr':0.05,'loss_mode':'marginal','ce_weight':1.,'dice_weight':1.,
             'validate_every':1,'threads':1,'max_batches_per_client':None,'casewise_sampling':False,'log_every':20,'amp':False,'fused_adam':False,'client_seeded':False},
 'usgr':{'random_sample_count':128,'tau_alpha':1000.,'tau_beta':1000.,
         'transport_iterations':100,'transport_learning_rate':None,'max_features':128},
 'aggregation':{'method':'gua','iterations':500,'tolerance':1e-8,'eps':1e-12,
                'global_step':0.1,'normalize_direction':True,'safeguard':False,'safety_fraction':0.9},
}


def merge(default, supplied, path=''):
    output=copy.deepcopy(default)
    for key,value in supplied.items():
        if key not in default:
            raise ValueError(f'Unknown configuration key: {path}{key}')
        if isinstance(default[key],dict):
            if not isinstance(value,dict):
                raise ValueError(f'{path}{key} must be an object')
            output[key]=merge(default[key],value,path+key+'.')
        else:
            output[key]=value
    return output


def validate(config):
    m,d,t,u,a = (config[k] for k in ('model','data','training','usgr','aggregation'))
    if type(config['seed']) is not int or not 0 <= config['seed'] < 2**32:
        raise ValueError('seed must be an integer in [0,2**32)')
    positive_int=[m[k] for k in ('image_size','num_classes','adapter_dim','encoder_channels','encoder_depth','decoder_channels','decoder_heads','decoder_depth','query_grid')]
    positive_int += [d['cache_volumes']] + [t[k] for k in ('rounds','local_epochs','batch_size','validate_every','threads')]
    positive_int += [u['random_sample_count'],u['transport_iterations'],u['max_features'],a['iterations']]
    if any(type(v) is not int or v <= 0 for v in positive_int):
        raise ValueError('Counts and sizes must be positive integers')
    if m['num_classes'] < 2 or m['image_size'] < 16:
        raise ValueError('At least two classes and image size >=16 are required')
    if m['backbone'] == 'tiny' and (m['image_size'] % 4 or m['encoder_channels'] % 4):
        raise ValueError('Tiny image size and encoder channels must be divisible by 4')
    if m['backbone'] not in ('vit_b','vit_l','tiny') or m['usgr_feature'] not in ('encoder','decoder'):
        raise ValueError('Unsupported backbone or USGR feature layer')
    if m.get('attention_backend','original') not in ('original','sdpa_global'):
        raise ValueError('Unsupported attention backend')
    if m['fusion'] not in ('multiply','concat'):
        raise ValueError('fusion must be multiply or concat')
    if d['slice_axis'] not in (0,1,2):
        raise ValueError('slice_axis must be 0,1,2')
    if d['expected_clients'] is not None and (type(d['expected_clients']) is not int or d['expected_clients'] <= 0):
        raise ValueError('expected_clients must be a positive integer or null')
    if t['loss_mode'] not in ('marginal','ignore','regions') or a['method'] not in ('gua','fedavg','centralized'):
        raise ValueError('Unsupported loss or aggregation')
    for value in [t['learning_rate'],u['tau_alpha'],u['tau_beta'],a['global_step'],a['eps']]:
        if not math.isfinite(value) or value <= 0:
            raise ValueError('Learning rate, penalties, global step and eps must be finite and positive')
    for value in [t['lambda_usgr'],t['ce_weight'],t['dice_weight'],a['tolerance']]:
        if not math.isfinite(value) or value < 0:
            raise ValueError('Loss coefficients and tolerance must be finite and nonnegative')
    if t['ce_weight'] + t['dice_weight'] <= 0:
        raise ValueError('At least one segmentation loss coefficient must be positive')
    if len(t['betas']) != 2 or any(not 0 <= beta < 1 for beta in t['betas']):
        raise ValueError('Adam betas must be in [0,1)')
    if not 0 < a['safety_fraction'] < 1:
        raise ValueError('safety_fraction must lie in (0,1)')
    for key in ('ct_window','mri_percentiles'):
        limits=config['preprocessing'][key]
        if len(limits)!=2 or not all(math.isfinite(v) for v in limits) or limits[0]>=limits[1]:
            raise ValueError(f'Invalid {key}')
    if not 0 <= config['preprocessing']['mri_percentiles'][0] < config['preprocessing']['mri_percentiles'][1] <= 100:
        raise ValueError('MRI percentiles must lie in [0,100]')
    aug=config['augmentation']
    if len(aug['scale_range'])!=2 or not 0 < aug['scale_range'][0] <= aug['scale_range'][1] < float('inf'):
        raise ValueError('Invalid augmentation scale range')
    if not 0 <= aug['rotation_degrees'] <= 180 or not 0 <= aug['intensity_jitter'] <= 1:
        raise ValueError('Invalid augmentation magnitude')
    cap=t['max_batches_per_client']
    if cap is not None and (type(cap) is not int or cap <= 0):
        raise ValueError('max_batches_per_client must be positive or null')
    return config


def load_config(path):
    path=Path(path).resolve()
    result=validate(merge(DEFAULTS,json.loads(path.read_text())))
    for section,key in [('data','manifest'),('model','sam_checkpoint')]:
        location=Path(result[section][key])
        if not location.is_absolute():
            result[section][key]=str((path.parent/location).resolve())
    return result
