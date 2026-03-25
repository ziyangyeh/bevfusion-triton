# BEVFusion-Triton Installation Guide

## Requirements

- Python >= 3.12
- CUDA 13.0
- GPU: NVIDIA Blackwell (sm_120, for example RTX 5090)
- [uv](https://docs.astral.sh/uv/) package manager

## Installation

### 1. Export environment variables

Before installation, export the following variables. They control the `cumm` and `spconv` build behavior, disable JIT, and set the target CUDA architecture.

```bash
export CUMM_CUDA_VERSION="13.0"
export CUMM_DISABLE_JIT="1"
export CUMM_CUDA_ARCH_LIST="12.0"
export SPCONV_DISABLE_JIT="1"
export MAX_JOBS=16
```

Notes:

- Re-export them in each new shell session.
- `MAX_JOBS` controls compile parallelism and can be adjusted to your CPU core count.

### 2. Install dependencies

```bash
uv sync --dev
```

`uv` will:

- build `cumm` and `spconv` from git sources
- install CUDA 13.0 builds of `torch` and `torchvision` from the official PyTorch index

### 3. Verify the installation

```bash
uv run python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
uv run python -c "import spconv; print('spconv OK')"
```

## Visualization

Start the app:

```bash
uv run streamlit run visualization_toolkit/app.py
```

The visualization toolkit now uses Plotly for interactive 3D point cloud and 3D box rendering directly inside Streamlit.

This means:

- no Open3D WebRTC setup
- no nginx requirement
- no TURN / ICE configuration
- only the Streamlit port needs to be exposed

## Troubleshooting

### `uv sync` reports version conflicts

`pyproject.toml` already uses `override-dependencies` to pin `cumm-cu130>=0.8.2`, which works around the upstream `spconv` metadata upper bound.

### CUDA is not found during build

Check that `nvcc` is available:

```bash
nvcc --version
```

If needed, add CUDA to your environment:

```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### `sm_120` architecture is not supported

Make sure you exported:

```bash
export CUMM_CUDA_ARCH_LIST="12.0"
```

and that your local CUDA version is at least 13.0.

### The Streamlit 3D plot is slow

Large point clouds can be expensive in a browser. If needed, reduce the number of displayed points in the visualization code or test on smaller samples first.
