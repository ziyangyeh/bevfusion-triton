# BEVFusion-Triton 安装指南

## 环境要求

- Python >= 3.12
- CUDA 13.0
- GPU: NVIDIA Blackwell (sm_120, 例如 RTX 5090)
- [uv](https://docs.astral.sh/uv/) 包管理器

## 安装步骤

### 1. 设置环境变量

安装前请先导出以下环境变量。这些变量控制 `cumm` 和 `spconv` 的编译行为，禁用 JIT 并指定目标 CUDA 架构。

```bash
export CUMM_CUDA_VERSION="13.0"
export CUMM_DISABLE_JIT="1"
export CUMM_CUDA_ARCH_LIST="12.0"
export SPCONV_DISABLE_JIT="1"
export MAX_JOBS=16
```

说明：

- 每次新开终端都需要重新导出
- `MAX_JOBS` 控制编译并行数，可按 CPU 核心数调整

### 2. 安装依赖

```bash
uv sync --dev
```

`uv` 会自动：

- 从 git 源编译 `cumm` 和 `spconv`
- 从 PyTorch 官方源拉取 CUDA 13.0 版本的 `torch` 与 `torchvision`

### 3. 验证安装

```bash
uv run python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
uv run python -c "import spconv; print('spconv OK')"
```

## 可视化

启动方式：

```bash
uv run streamlit run visualization_toolkit/app.py
```

现在 3D 点云和 3D 框已经改成直接用 Plotly 在 Streamlit 内交互显示。

这意味着：

- 不再需要 Open3D WebRTC
- 不再需要 nginx
- 不再需要 TURN / ICE 配置
- 浏览器只需要访问 Streamlit 端口

## 常见问题

### `uv sync` 报版本冲突

`pyproject.toml` 已通过 `override-dependencies` 强制指定 `cumm-cu130>=0.8.2`，用来绕过 `spconv` 的上游版本上限限制。

### 编译时找不到 CUDA

确认 `nvcc` 可用：

```bash
nvcc --version
```

如果不在 `PATH` 中，可追加：

```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### `sm_120` 架构不支持

确认已经正确导出：

```bash
export CUMM_CUDA_ARCH_LIST="12.0"
```

并且本机 CUDA 版本至少为 13.0。

### Streamlit 里的 3D 图较慢

浏览器端显示大规模点云本来就会有压力。如果需要，可以后续在可视化代码里继续下采样点云，或者先用较小样本进行查看。
