# SRFL

Code project for **SRFL: Stable and Robust Federated Learning for Segment Anything Model with Non-IID Medical Images**.

SRFL is a federated medical image segmentation framework that adapts the Segment Anything Model (SAM) to heterogeneous client data. The implementation brings together three components:

1. 🧩 **SAM adaptation** with a frozen image encoder, trainable bottleneck adapters, and a dual-scale segmentation decoder.
2. 🎯 **USGR feature regularization** during local client training.
3. 🔄 **GUA parameter aggregation** for global trainable-model updates.

This repository provides training and evaluation code, model configurations, implementation notes, and regression tests. Datasets, pretrained weights, experiment checkpoints, and manuscript files are supplied separately.

## 1. 📂 Repository Layout

```text
SRFL/
├── configs/                 Model and CPU example configurations
├── docs/                    Data setup, method notes, and references
├── scripts/                 Training, evaluation, and utility entry points
├── srfl/
│   ├── models/              SAM encoder, adapters, and decoder
│   ├── usgr.py              Feature regularization
│   ├── aggregation.py       GUA parameter aggregation
│   ├── training.py          Federated training and checkpoint resume
│   ├── data.py              Volume loading and preprocessing
│   ├── losses.py            Segmentation supervision
│   ├── regions.py           Region-based supervision utilities
│   ├── evaluation.py        Native-volume prediction and evaluation
│   └── metrics.py           Dice, HD, and HD95
├── tests/                   Regression tests
├── .github/workflows/       Automated CPU test workflow
├── convergence.py           SGD comparison of federated methods
├── supervise.py             Single-GPU benchmark queue
├── mots_config.json         MOTS benchmark configuration
├── lidc_config.json         LIDC benchmark configuration
├── pyproject.toml           Package metadata and dependencies
├── requirements.txt         Core dependency list
├── requirements-sam.txt     SAM dependency list
├── LICENSE_NOTICE.md        Current licensing status
└── SHA256SUMS               File integrity checksums
```

## 2. 📌 Current Release Status

✅ The repository currently provides:

- SAM ViT-B and ViT-L integration with trainable adapters and a segmentation decoder.
- USGR regularization and GUA aggregation modules.
- Federated training with validation, checkpoint saving, and round-boundary resume.
- Native-volume prediction and evaluation with Dice, HD, and HD95.
- Configuration files for MOTS and liver-lesion MRI, plus a small CPU example.
- A separate SGD benchmark for FedAvg, FedProx, SCAFFOLD, and SRFL.
- Regression tests and a GitHub Actions CPU workflow.

📦 The following items are not included:

- Patient imaging volumes and clinical labels.
- Pretrained SAM weights and trained experiment checkpoints.
- Complete historical training logs and manuscript assets.

Clients are simulated within local processes. Reproducing manuscript results requires the corresponding data partitions, checkpoints, and experiment settings. **This package has not independently reproduced the manuscript's reported numerical results.**

## 3. ⚙️ Installation

Use **Python 3.10 or later**. Open a terminal in the repository root and create an environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

On Windows, activate the environment with `.venv\Scripts\activate` instead.

Install a PyTorch build suitable for your hardware, then install the project with SAM support and test dependencies:

```bash
python -m pip install -e '.[sam,test]'
```

For the lightweight CPU example and tests without SAM support:

```bash
python -m pip install -e '.[test]'
```

For SAM experiments, download the appropriate ViT-B or ViT-L checkpoint separately and set `model.sam_checkpoint` in the selected configuration. Pretrained weights are not bundled with this repository.

## 4. 🚀 Quick Start: CPU Example

Run the following commands from the repository root:

```bash
# Generate a small example dataset.
python scripts/make_demo_data.py --output data/demo

# Train with the lightweight CPU configuration.
python scripts/train.py --config configs/smoke.json --output outputs/smoke

# Evaluate the selected checkpoint and save predictions.
python scripts/evaluate.py --checkpoint outputs/smoke/best.pt --output outputs/eval --save-predictions

# Predict one example volume.
python scripts/predict.py --checkpoint outputs/smoke/best.pt --image data/demo/client0_test_image.npy --modality normalized --output outputs/prediction.npy
```

The example uses **synthetic volumes and a small encoder** to exercise the pipeline. It does not use patient data or pretrained SAM and is not a benchmark of medical segmentation performance. Use an empty directory for generated demo data and evaluation outputs, and a new file path for prediction outputs.

## 5. 🗂️ Data Preparation

Prepare a patient-level JSON manifest describing each case's image, label, client, split, modality, and annotation coverage.

| Item | Expected format |
| --- | --- |
| Input volumes | Single-channel 3D `.nii`, `.nii.gz`, or `.npy` files |
| Data splits | `train`, `val`, and `test` |
| Modalities | `ct`, `mri`, or `normalized` |
| Patient assignment | Each patient belongs to one client and one split |
| Image–label alignment | Same shape, orientation, and spatial grid |
| NumPy spacing | Explicit spacing in millimeters, in array-axis order |
| Annotation coverage | Class or region coverage consistent across clients |

See **[Data Setup and Evaluation](docs/DATA.md)** for the manifest schema, preprocessing, partial annotations, and metric conventions.

Paths inside a configuration are relative to that configuration file. Manifest image and label paths are relative to the manifest directory. Command-line paths are relative to the current working directory.

## 6. 🏋️ Federated Training

Choose a configuration matching your data and SAM backbone:

| Configuration | Use |
| --- | --- |
| `configs/mots_vit_b.json` | MOTS with SAM ViT-B |
| `configs/mots_vit_l.json` | MOTS with SAM ViT-L |
| `configs/liver_mri_vit_b.json` | Liver-lesion MRI with SAM ViT-B |
| `configs/smoke.json` | Lightweight CPU pipeline example |

After preparing the manifest and setting the SAM checkpoint path, start training:

```bash
python scripts/train.py --config configs/mots_vit_b.json --manifest data/mots/manifest.json --output outputs/mots --device cuda
```

Resume from a round checkpoint:

```bash
python scripts/train.py --resume outputs/mots/last.pt --rounds 200 --output outputs/mots_resumed --device cuda
```

- `last.pt` supports resume at round boundaries.
- `best.pt` contains the model selected by validation Dice.
- `--rounds` specifies the total target round count, including completed rounds.
- Adam state is reset at the start of each local round.

## 7. 📊 Evaluation and Prediction

Evaluate on the test split and save native-volume predictions:

```bash
python scripts/evaluate.py --checkpoint outputs/mots/best.pt --split test --output outputs/evaluation --device cuda --distance-unit voxel --save-predictions
```

Predict an unlabeled volume:

```bash
python scripts/predict.py --checkpoint outputs/mots/best.pt --image data/example.nii.gz --modality ct --output outputs/prediction.nii.gz --device cuda
```

Evaluation reconstructs predictions on the native volume grid and reports per-case **Dice, HD, and HD95**. Use `--distance-unit mm` for physical distances when valid spacing is available. Empty-mask handling, valid-score counts, and aggregation rules are documented in [Data Setup and Evaluation](docs/DATA.md#metrics).

## 8. 🔬 Federated Method Benchmark

`convergence.py` provides a separate SGD comparison of **FedAvg, FedProx, SCAFFOLD, and SRFL** using the supplied benchmark configurations. Set the manifest and checkpoint paths before running:

```bash
python convergence.py --config mots_config.json --method srfl --gpu 0 --seed 20261015 --out outputs/benchmark_mots
```

Queue the benchmark seeds on one GPU:

```bash
python supervise.py --root . --dataset mots --method srfl --gpu 0
```

`--root` points to the directory containing the benchmark configuration. Relative data and checkpoint paths in that configuration resolve from this directory.

The benchmark has separate optimizer, sampling, and metric settings from the standard pipeline. Its configurations specify 64 batches per client per round, rather than a full local epoch. See [Implementation Details](docs/METHOD_MAPPING.md) before comparing results across entry points.

## 9. 🧪 Tests

Run the regression suite from the repository root:

```bash
python -m pytest -q
```

The included [GitHub Actions workflow](.github/workflows/tests.yml) is configured to run the tests and CPU training, evaluation, and prediction example on pushes and pull requests. It also supports manual dispatch.

## 10. 📚 Documentation and Attribution

| Document | Contents |
| --- | --- |
| [Data Setup and Evaluation](docs/DATA.md) | Input formats, manifests, preprocessing, supervision, and metrics |
| [Implementation Details](docs/METHOD_MAPPING.md) | Component mapping, implementation choices, and benchmark scope |
| [References and Attribution](docs/REFERENCES.md) | SAM, FedU², and FedFMS references |

SRFL uses the installed SAM image encoder. FedU² FUR/EUA provides mathematical precedents for USGR/GUA, and FedFMS provides a methodological precedent for federated SAM adaptation. See the references document for attribution details.

## 11. 📄 License

A release license has **not yet been specified** for this code package. See [LICENSE_NOTICE.md](LICENSE_NOTICE.md) for the current status. Third-party dependencies and pretrained weights retain their respective license terms.
