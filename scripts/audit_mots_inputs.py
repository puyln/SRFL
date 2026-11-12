import argparse
import json
from pathlib import Path


def inspect_mask(path):
    import numpy as np
    if path.suffix == '.npy':
        data = np.load(path, mmap_mode='r', allow_pickle=False)
        geometry = {}
    elif str(path).endswith(('.nii', '.nii.gz')):
        import nibabel as nib
        volume = nib.load(path)
        data = np.asanyarray(volume.dataobj)
        geometry = {'spacing': list(map(float, volume.header.get_zooms())),
                    'affine': volume.affine.tolist(),
                    'orientation': list(nib.aff2axcodes(volume.affine))}
    else:
        raise ValueError('Unsupported mask format')
    if data.ndim != 3:
        raise ValueError(f'Expected a 3D mask, found {data.shape}; clarify channels/overlap first')
    if not np.isfinite(data).all():
        raise ValueError('Non-finite mask values')
    if not np.equal(data, np.rint(data)).all():
        raise ValueError('Non-integer mask; may be image intensities or probabilities')
    values, counts = np.unique(data, return_counts=True)
    if len(values) > 1024:
        raise ValueError('Too many label IDs; check whether this is a mask')
    return {'shape': list(data.shape), 'dtype': str(data.dtype), **geometry,
            'label_voxel_counts': {str(int(v)): int(n) for v, n in zip(values, counts)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mask-glob', action='append', default=[],
                        help='Explicit pattern relative to root, e.g. **/labelsTr/*.nii.gz')
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        parser.error('Data root is not accessible')
    output = args.output.resolve()
    if output == root or root in output.parents:
        parser.error('Write reports outside the original dataset')
    if output.exists():
        parser.error('Choose a new output filename')
    files = sorted(p for p in root.rglob('*') if p.is_file() and p.name != '.DS_Store')
    inventory = [{'path': str(p.relative_to(root)), 'bytes': p.stat().st_size} for p in files]
    metadata = [str(p.relative_to(root)) for p in files
                if p.suffix.lower() in {'.json', '.csv', '.txt', '.yaml', '.yml', '.md'}]
    selected = sorted({p for pattern in args.mask_glob for p in root.glob(pattern) if p.is_file()})
    if args.mask_glob and not selected:
        parser.error('No masks matched the explicit patterns')
    masks, errors = [], []
    for p in selected:
        if root not in p.resolve().parents:
            parser.error('Mask symlink resolves outside the data root')
        try:
            masks.append({'path': str(p.relative_to(root)), **inspect_mask(p)})
        except Exception as exc:
            errors.append({'path': str(p.relative_to(root)), 'error': str(exc)})
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as handle:
        json.dump({'root': str(root), 'inventory': inventory, 'metadata_files': metadata,
                   'masks': masks, 'errors': errors,
                   'anatomical_mapping': 'Unresolved: verify label metadata and masks; do not infer from IDs.'},
                  handle, ensure_ascii=False, indent=2)
    print(json.dumps({'files': len(files), 'masks_checked': len(masks),
                      'errors': len(errors), 'report': str(output)}))
    if errors:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
