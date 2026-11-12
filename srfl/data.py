from collections import OrderedDict
import json
from pathlib import Path
import re
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F


def read_manifest(path, num_classes):
    path = Path(path).resolve()
    document = json.loads(path.read_text())
    cases = document.get('cases', [])
    if not cases:
        raise ValueError('manifest.cases must be a non-empty list')
    regions=document.get('regions')
    if regions is not None:
        if len(regions)!=num_classes or any(not r or any(type(v) is not int or v<=0 for v in r) for r in regions):
            raise ValueError('regions must define one nonempty positive-label set per output channel')
    label_classes=max(v for r in regions for v in r)+1 if regions else num_classes
    ids, paths, patients = set(), set(), {}
    for original in cases:
        for field in ('case_id', 'patient_id', 'client_id', 'split', 'image', 'label', 'annotated_classes', 'modality'):
            if field not in original:
                raise ValueError(f'Manifest case missing {field}')
        case_id = original['case_id']
        if not isinstance(case_id, str) or not re.fullmatch(r'[A-Za-z0-9_.-]+', case_id) or case_id in ('.', '..'):
            raise ValueError('case_id must be a safe filename component')
        if case_id in ids:
            raise ValueError(f'Duplicate case_id: {case_id}')
        ids.add(case_id)
        if original['split'] not in ('train', 'val', 'test'):
            raise ValueError('split must be train, val or test')
        if original['modality'] not in ('ct', 'mri', 'normalized'):
            raise ValueError('modality must be ct, mri or normalized')
        labels = original['annotated_classes']
        if not labels or 0 not in labels or len(labels) != len(set(labels)) or any(type(v) is not int or v < 0 or v >= label_classes for v in labels):
            raise ValueError('annotated_classes must contain unique valid class IDs including 0')
        if 'label_map' in original:
            mapping = original['label_map']
            if not isinstance(mapping, dict) or any(type(v) is not int or v < -1 or v >= label_classes for v in mapping.values()):
                raise ValueError('label_map values must be global class IDs or -1')
            try:
                for key in mapping:
                    int(key)
            except (TypeError, ValueError) as exc:
                raise ValueError('label_map keys must be integer strings') from exc
        if regions is not None:
            annotated=original.get('annotated_regions',[])
            if not annotated or len(set(annotated))!=len(annotated) or any(type(v) is not int or not 0<=v<num_classes for v in annotated):
                raise ValueError('Valid annotated_regions required for region segmentation')
            original['_regions']=regions
        original['client_id'] = str(original['client_id'])

        key = str(original['patient_id'])
        identity = (original['client_id'], original['split'])
        if key in patients and patients[key] != identity:
            raise ValueError(f'Patient {key} crosses a client or split boundary')
        patients[key] = identity
        for field in ('image', 'label'):
            value = Path(original[field])
            original[field] = str((path.parent / value).resolve() if not value.is_absolute() else value.resolve())
            if not Path(original[field]).is_file():
                raise FileNotFoundError(original[field])
        if original['image'] in paths:
            raise ValueError('An image volume appears more than once in the manifest')
        paths.add(original['image'])
        if original['image'].endswith('.npy'):
            spacing = np.asarray(original.get('spacing', []), dtype=float)
            if spacing.shape != (3,) or not np.isfinite(spacing).all() or (spacing <= 0).any():
                raise ValueError('NPY volumes require positive spacing [axis0, axis1, axis2]')
    return cases


def _read_array(path):
    if path.endswith('.npy'):
        return np.load(path, mmap_mode='r', allow_pickle=False), None
    if path.endswith(('.nii', '.nii.gz')):
        import nibabel as nib
        volume = nib.load(path)
        return volume, volume
    raise ValueError('Supported volume formats: .npy, .nii, .nii.gz')


def volume_shape(case):
    image, nifti = _read_array(case['image'])
    shape = image.shape
    if nifti is not None:
        import nibabel as nib
        order=np.argsort(nib.orientations.io_orientation(nifti.affine)[:,0]).astype(int)
        shape=tuple(shape[i] for i in order)
    if len(shape) != 3:
        raise ValueError('Only 3D single-channel volumes are supported')
    return shape


def load_case(case):
    image_obj, image_nifti = _read_array(case['image'])
    label_obj, label_nifti = _read_array(case['label'])
    if (image_nifti is None) != (label_nifti is None):
        raise ValueError('Image and label must use the same volume format')
    if image_nifti is not None:
        if not np.allclose(image_nifti.affine, label_nifti.affine, atol=1e-4):
            raise ValueError('Image and label NIfTI affines differ; align before training')
        import nibabel as nib
        image_obj=nib.as_closest_canonical(image_obj)
        label_obj=nib.as_closest_canonical(label_obj)
        image_nifti=image_obj
        image = np.asarray(image_obj.dataobj, dtype=np.float32)
        labels = np.asarray(label_obj.dataobj)
        spacing = np.asarray(image_nifti.header.get_zooms()[:3], dtype=float)
        affine = image_nifti.affine
    else:
        image = np.asarray(image_obj, dtype=np.float32)
        labels = np.asarray(label_obj)
        spacing = np.asarray(case['spacing'], dtype=float)
        affine = None
    if image.ndim != 3 or labels.shape != image.shape:
        raise ValueError('Image and label must be aligned 3D volumes of equal shape')
    if not np.isfinite(image).all() or not np.isfinite(labels).all():
        raise ValueError('Volume contains NaN or infinity')
    if not np.equal(labels, np.round(labels)).all():
        raise ValueError('Labels must contain integer class IDs')
    labels = labels.astype(np.int64)
    if 'label_map' in case:
        mapping = {int(k): int(v) for k,v in case['label_map'].items()}
        mapping.setdefault(-1, -1)
        missing = set(np.unique(labels)) - mapping.keys()
        if missing:
            raise ValueError(f'label_map does not define raw IDs: {sorted(missing)}')
        mapped = np.full_like(labels, -1)
        for raw_id, global_id in mapping.items():
            mapped[labels == raw_id] = global_id
        labels = mapped
    legal = set(case['annotated_classes']) | {-1}
    if not set(np.unique(labels)).issubset(legal):
        raise ValueError('Labels contain classes absent from annotated_classes; map labels explicitly')
    if not np.isfinite(spacing).all() or (spacing <= 0).any():
        raise ValueError('Spacing must be finite and positive')
    if affine is not None:
        axes = affine[:3, :3]
        norms = np.linalg.norm(axes, axis=0)
        if not np.allclose((axes / norms).T @ (axes / norms), np.eye(3), atol=1e-4):
            raise ValueError('Sheared NIfTI affine: resample to an orthogonal grid before spacing-based HD')
    return image, labels, spacing, affine


def normalize_volume(image, modality, preprocessing):
    image = np.asarray(image, dtype=np.float32)
    if modality == 'ct':
        lo, hi = preprocessing['ct_window']
    elif modality == 'mri':
        lo, hi = np.percentile(image, preprocessing['mri_percentiles'])
    else:
        if image.min() < 0 or image.max() > 1:
            raise ValueError('normalized images must lie in [0,1]')
        return image.copy()
    if hi <= lo:
        return np.zeros_like(image)
    return np.clip((image - lo) / (hi - lo), 0, 1).astype(np.float32)


def image_tensor(slice_image, size):
    image = torch.from_numpy(np.ascontiguousarray(slice_image)).float()[None, None]
    image = F.interpolate(image, size=(size,size), mode='bilinear', align_corners=False)
    return image[0].repeat(3,1,1)


class SliceDataset(Dataset):
    def __init__(self, cases, config, *, augment=False):
        self.cases = cases
        self.config = config
        self.augment = augment
        self.cache = OrderedDict()
        self.index = [(i, z) for i,c in enumerate(cases) for z in range(volume_shape(c)[config['data']['slice_axis']])]
        if not self.index:
            raise ValueError('Dataset has no slices')

    def __len__(self):
        return len(self.index)

    def _volume(self, index):
        if index not in self.cache:
            image, label, _, _ = load_case(self.cases[index])
            image = normalize_volume(image, self.cases[index]['modality'], self.config['preprocessing'])
            self.cache[index] = (image, label)
            while len(self.cache) > self.config['data']['cache_volumes']:
                self.cache.popitem(last=False)
        self.cache.move_to_end(index)
        return self.cache[index]

    def __getitem__(self, index):
        case_index, z = self.index[index]
        image, label = self._volume(case_index)
        axis = self.config['data']['slice_axis']
        size = self.config['model']['image_size']
        if self.config['data'].get('slice_views',False):
            selection=tuple(z if i==axis else slice(None) for i in range(3))
            image_slice,label_slice=image[selection],label[selection]
        else:
            image_slice,label_slice=np.take(image,z,axis=axis),np.take(label,z,axis=axis)
        image = image_tensor(image_slice, size)
        target = torch.from_numpy(np.ascontiguousarray(label_slice)).float()[None,None]
        target = F.interpolate(target, size=(size,size), mode='nearest')[0,0].long()
        if self.augment:
            options = self.config['augmentation']
            degrees = (torch.rand(()).item() * 2 - 1) * options['rotation_degrees']
            radians = degrees * np.pi / 180.
            scale = options['scale_range'][0] + torch.rand(()).item() * (options['scale_range'][1] - options['scale_range'][0])
            matrix = torch.tensor([[[np.cos(radians)/scale, -np.sin(radians)/scale, 0.],
                                     [np.sin(radians)/scale, np.cos(radians)/scale, 0.]]],dtype=torch.float32)
            grid = F.affine_grid(matrix, image[None].shape, align_corners=False)
            image = F.grid_sample(image[None],grid,mode='bilinear',padding_mode='zeros',align_corners=False)[0]

            target = F.grid_sample((target.float()+1)[None,None],grid,mode='nearest',padding_mode='zeros',align_corners=False)[0,0].long()-1
            jitter = options['intensity_jitter']
            image = (image * (1 + (torch.rand(()).item()*2-1)*jitter) + (torch.rand(()).item()*2-1)*jitter).clamp(0,1)
        annotated = torch.zeros(self.config['model']['num_classes'],dtype=torch.bool)
        annotated[self.cases[case_index].get('annotated_regions',self.cases[case_index]['annotated_classes'])] = True
        return image, target, annotated


class CaseShuffleSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, generator):
        self.dataset, self.generator = dataset, generator
    def __len__(self):
        return len(self.dataset)
    def __iter__(self):
        groups = {}
        for index, (case_index, _) in enumerate(self.dataset.index):
            groups.setdefault(case_index, []).append(index)
        keys=list(groups)
        for i in torch.randperm(len(keys),generator=self.generator).tolist():
            indices=groups[keys[i]]
            for j in torch.randperm(len(indices),generator=self.generator).tolist():
                yield indices[j]
