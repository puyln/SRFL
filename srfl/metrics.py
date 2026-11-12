import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure


def binary_metrics(prediction, target, spacing, valid=None):
    prediction, target = np.asarray(prediction, bool), np.asarray(target, bool)
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError('Metrics expect equal 3D mask shapes')
    if valid is not None:
        valid = np.asarray(valid, bool)
        if valid.shape != target.shape:
            raise ValueError('valid mask has wrong shape')
        if not valid.any():
            return {'dice':None,'hd':None,'hd95':None,'status':'no_valid_voxels'}
        prediction, target = prediction & valid, target & valid
    total = int(prediction.sum() + target.sum())
    if total == 0:

        return {'dice':None,'hd':None,'hd95':None,'status':'both_empty'}
    dice = float(2 * np.count_nonzero(prediction & target) / total)
    if not prediction.any() or not target.any():
        return {'dice':dice,'hd':None,'hd95':None,'status':'one_empty'}
    if valid is not None and not valid.all():

        return {'dice':dice,'hd':None,'hd95':None,'status':'partial_voxel_annotation'}
    spacing = np.asarray(spacing, float)
    if spacing.shape != (3,) or not np.isfinite(spacing).all() or (spacing <= 0).any():
        raise ValueError('Spacing must have three finite positive values')
    structure = generate_binary_structure(3,1)
    ps = prediction ^ binary_erosion(prediction,structure=structure,border_value=0)
    ts = target ^ binary_erosion(target,structure=structure,border_value=0)
    p_to_t = distance_transform_edt(~ts,sampling=spacing)[ps]
    t_to_p = distance_transform_edt(~ps,sampling=spacing)[ts]
    distances = np.concatenate([p_to_t,t_to_p])
    return {'dice':dice, 'hd':float(distances.max()),
            'hd95':float(np.percentile(distances,95)), 'status':'ok'}


def summarize(records):
    per_class = {}
    for class_id in sorted({row['class_id'] for row in records}):
        rows = [row for row in records if row['class_id'] == class_id]
        summary = {'case_count':len(rows), 'one_empty_count':sum(r['status']=='one_empty' for r in rows),
                   'both_empty_count':sum(r['status']=='both_empty' for r in rows)}
        for metric in ('dice','hd','hd95'):
            available = [r[metric] for r in rows if r[metric] is not None]
            summary[metric] = float(np.mean(available)) if available else None
            summary[metric+'_valid_count'] = len(available)
        per_class[str(class_id)] = summary
    result = {'per_class':per_class}
    for metric in ('dice','hd','hd95'):
        values = [r[metric] for r in per_class.values() if r[metric] is not None]
        result['macro_'+metric] = float(np.mean(values)) if values else None
    return result
