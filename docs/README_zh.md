# BEVFusion-Triton

[English](../README.md) | [中文](README_zh.md)

## 项目简介

BEVFusion-Triton 是一个本地化的 BEVFusion 风格代码库，核心特点包括：

- 以 Triton 为主的自定义算子实现
- Lightning 训练与验证流程
- Hydra 配置系统
- 3D 检测与 BEV 地图分割实验
- 检测与分割可视化工具

## 目录说明

- [`configs/`](../configs/)：Hydra 配置树
- [`datasets/`](../datasets/)：数据集与 pipeline
- [`lit/`](../lit/)：Lightning 数据与训练封装
- [`models/`](../models/)：模型组件
- [`ops/`](../ops/)：Triton 与 reference 算子实现
- [`tools/`](../tools/)：数据准备与可视化工具
- [`tests/`](../tests/)：算子与集成测试

## 文档索引

- [安装指南](INSTALL_zh.md)
- [项目 TODO](TODO_en.md)

## 快速开始

安装依赖：

```bash
uv sync --dev
```

运行默认实验：

```bash
uv run python train.py
```

运行检测实验：

```bash
uv run python train.py --config-name experiment/det/swint_v0p075
```

运行分割实验：

```bash
uv run python train.py --config-name experiment/seg/fusion_bev256d2_lss
```

## 可视化

检测可视化：

```bash
uv run python -m tools.visualize \
  --mode val-pred \
  --config configs/experiment/det/swint_v0p075.yaml \
  --dataset-root data/nuscenes-mini \
  --index 0 \
  --out-dir outputs/visualize_det
```

分割可视化：

```bash
uv run python -m tools.visualize \
  --mode val-pred \
  --config configs/experiment/seg/fusion_bev256d2_lss.yaml \
  --dataset-root data/nuscenes-mini \
  --index 0 \
  --out-dir outputs/visualize_seg
```

## 预训练权重

- `pretrained/swin_tiny_patch4_window7_224.pth`：相机 Swin backbone 初始化权重
- `pretrained/bevfusion-det.pth`：检测整模初始化权重
- `pretrained/bevfusion-seg.pth`：分割整模初始化权重

## 配置入口

- 检测：[`configs/experiment/det/swint_v0p075.yaml`](../configs/experiment/det/swint_v0p075.yaml)
- 分割：[`configs/experiment/seg/fusion_bev256d2_lss.yaml`](../configs/experiment/seg/fusion_bev256d2_lss.yaml)
