# Data setup and evaluation

## Volume format

Inputs are single-channel 3D `.nii`, `.nii.gz`, or `.npy` volumes. Images and labels must share the same shape, orientation, and spatial grid. Alignment and DICOM conversion are performed before loading. Select one MRI sequence or phase per input volume.

NIfTI spacing is read from the header. NumPy inputs require `spacing` in millimeters, ordered by array axis. `data.slice_axis` selects the slice axis and defaults to 2. Use an orthogonal spatial grid for spacing-based surface distances.

## Manifest

Paths are relative to the manifest directory.

```json
{
  "cases": [
    {
      "case_id": "site0_patient001",
      "patient_id": "site0_patient001",
      "client_id": "0",
      "split": "train",
      "image": "images/patient001.nii.gz",
      "label": "labels/patient001.nii.gz",
      "modality": "ct",
      "annotated_classes": [0, 1, 2],
      "label_map": {"0": 0, "1": 1, "2": 2}
    }
  ]
}
```

A complete manifest contains training, validation, and test cases for the intended clients. Supported splits are `train`, `val`, and `test`; modalities are `ct`, `mri`, and `normalized`. Patient identifiers are globally unique and remain within one client and split.

`annotated_classes` identifies classes with complete annotation coverage and includes background 0. Unknown voxels use -1. `label_map` maps original IDs to global IDs in memory; it may be omitted when IDs already match. Every observed ID requires a mapping when a map is supplied. NPY records also provide, for example, `"spacing": [0.8, 0.8, 2.5]`.

CT class configurations use 12 mutually exclusive labels, including background; MRI class configurations use seven. The benchmark configurations use the region-based loss with an explicit `regions` list and per-case `annotated_regions`. Configure class or region identities consistently across clients.

## Preprocessing

CT intensities are clipped to [-500, 1000] HU and scaled to [0, 1]. MRI intensities use configurable volume-level percentile clipping. `normalized` inputs are already in [0, 1]. Image slices are resized bilinearly and repeated over three channels; labels use nearest-neighbor interpolation. SAM normalization is controlled by the model configuration.

Rotation, scaling, and intensity augmentation apply to training inputs. Geometric padding has label -1. Evaluation restores the original volume grid.

## Partial annotations

`marginal` loss combines probabilities of background and unannotated classes into a local background category. `ignore` loss evaluates known pixels and excludes label -1. `regions` loss uses independent region channels and per-case annotated-region coverage, supporting overlapping regions. Configure supervision using the corresponding data representation.

## Metrics

- Dice is calculated per case and annotated foreground class, on a 0-1 scale.
- HD is the maximum bidirectional nearest-surface distance.
- HD95 is the 95th percentile of pooled bidirectional surface distances.
- Surfaces use a one-voxel boundary from six-neighbor binary erosion.
- Distance units are `voxel` or `mm`.
- If both masks are empty, Dice, HD, and HD95 are null with status `both_empty`.
- If one mask is empty, Dice is zero; HD and HD95 are null with status `one_empty`.
- With ignored voxels, Dice uses valid voxels and surface distances are null.
- The standard evaluator averages defined case scores within each class, then averages defined class scores. Valid counts and empty-mask statuses accompany the results.

The separate SGD benchmark reports GT-positive case Dice per region. Its aggregation rule is distinct from the standard evaluator. Compare results using the same metric convention and data partition.
