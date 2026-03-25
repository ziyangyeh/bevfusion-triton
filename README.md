# BEVFusion-Triton

BEVFusion-Triton is a local BEVFusion-style codebase focused on a clean training stack, Triton-first operator implementations, and a Hydra + Lightning workflow for 3D detection and BEV map segmentation.

## What This Repository Contains

- A local BEVFusion-style model stack for:
  - 3D object detection
  - BEV map segmentation
- Triton-first custom operators under [`ops/`](ops/)
- Local CPU fallback paths for the operators that need them
- Lightning training, validation, and visualization tooling
- Hydra configs organized by:
  - dataset
  - model modules
  - optimizer
  - runtime
  - experiment

## Current Scope

The main supported experiment entry points are:

- Detection:
  - [`configs/experiment/det/swint_v0p075.yaml`](configs/experiment/det/swint_v0p075.yaml)
- Segmentation:
  - [`configs/experiment/seg/fusion_bev256d2_lss.yaml`](configs/experiment/seg/fusion_bev256d2_lss.yaml)

The project currently prioritizes:

- NuScenes-based workflows
- single-GPU validation and evaluation
- Triton/local operator correctness against legacy reference implementations in tests

## Repository Layout

- [`configs/`](configs/)
  - Hydra config tree
- [`core/`](core/)
  - boxes, points, geometry, and related utilities
- [`datasets/`](datasets/)
  - dataset definitions and pipelines
- [`lit/`](lit/)
  - Lightning data/module glue
- [`models/`](models/)
  - encoders, decoders, heads, losses, and fusion models
- [`ops/`](ops/)
  - local Triton operators and legacy reference backends for tests
- [`tools/`](tools/)
  - data creation and visualization commands
- [`tests/`](tests/)
  - operator-focused regression and parity tests
- [`docs/`](docs/)
  - installation and secondary documentation

## Documentation
- [Installation Guide](docs/INSTALL_en.md)
- [Project TODO](docs/TODO_en.md)

## Installation

See:

- [docs/INSTALL_en.md](docs/INSTALL_en.md)
- [docs/INSTALL_zh.md](docs/INSTALL_zh.md)

Typical setup:

```bash
uv sync --dev
```

## Pretrained Checkpoints

Expected checkpoint roles:

- `pretrained/swin_tiny_patch4_window7_224.pth`
  - camera Swin backbone initialization
- `pretrained/bevfusion-det.pth`
  - full detection model initialization
- `pretrained/bevfusion-seg.pth`
  - full segmentation model initialization

The default experiments already point to the detection and segmentation checkpoints through `load_from`.

## Training

Run the default experiment:

```bash
uv run python train.py
```

Run detection explicitly:

```bash
uv run python train.py --config-name experiment/det/swint_v0p075
```

Run segmentation explicitly:

```bash
uv run python train.py --config-name experiment/seg/fusion_bev256d2_lss
```

Override config values with Hydra:

```bash
uv run python train.py \
  --config-name experiment/det/swint_v0p075 \
  dataset_root=data/nuscenes-mini \
  trainer.max_epochs=1 \
  dataloader.train.batch_size=1
```

## Data Preparation

NuScenes info and GT database generation:

```bash
uv run python -m tools.create_data nuscenes \
  --root_path ./data/nuscenes \
  --out_dir ./data/nuscenes \
  --version v1.0 \
  --max_sweeps 10 \
  --extra_tag nuscenes
```

Mini split generation:

```bash
uv run python -m tools.create_data nuscenes \
  --root_path ./data/nuscenes-mini \
  --out_dir ./data/nuscenes-mini \
  --version v1.0-mini \
  --max_sweeps 10 \
  --extra_tag nuscenes
```

## Visualization

Detection visualization:

```bash
uv run python -m tools.visualize \
  --mode val-pred \
  --config configs/experiment/det/swint_v0p075.yaml \
  --dataset-root data/nuscenes-mini \
  --index 0 \
  --out-dir outputs/visualize_det
```

Segmentation visualization:

```bash
uv run python -m tools.visualize \
  --mode val-pred \
  --config configs/experiment/seg/fusion_bev256d2_lss.yaml \
  --dataset-root data/nuscenes-mini \
  --index 0 \
  --out-dir outputs/visualize_seg
```

Streamlit app:

```bash
uv run streamlit run visualization_toolkit/app.py
```

The Streamlit 3D view now uses Plotly directly inside the app:

- no Open3D WebRTC viewer
- no nginx requirement
- no TURN / ICE setup
- browser access only needs the Streamlit port

## Notes On Operators

The model runtime is intended to use the local implementations in [`ops/`](ops/):

- voxelization: local Triton implementation with local CPU fallback
- BEV pooling: local Triton implementation with local CPU fallback
- sparse convolution path: local `ops.spconv`

Legacy operator backends are still kept in the repository as reference implementations for tests and parity checks. They are not the intended primary runtime path.

## Notes On Evaluation

- Single-GPU validation and evaluation are the current main path.
- Multi-GPU aggregation for detection evaluation is still listed as future work in [docs/TODO_en.md](docs/TODO_en.md).

## Development

Sync development dependencies:

```bash
uv sync --dev
```

Run tests:

```bash
uv run python -m pytest
```

## Language

- The default top-level README is English.
- Chinese documentation lives in files with the `_zh` suffix under [`docs/`](docs/).
- English secondary docs use the `_en` suffix where needed.
