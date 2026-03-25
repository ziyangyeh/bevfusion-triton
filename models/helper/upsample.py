from __future__ import annotations

from typing import Any

import torch.nn as nn

__all__ = ['build_upsample_layer']


def build_upsample_layer(
	upsample_cfg: dict[str, Any] | None,
	in_channels: int,
	out_channels: int,
	kernel_size: int,
	stride: int,
) -> nn.Module:
	cfg = {} if upsample_cfg is None else dict(upsample_cfg)
	layer_type = str(cfg.pop('type', 'deconv'))
	bias = bool(cfg.pop('bias', False))
	if layer_type != 'deconv':
		raise KeyError(f'Unsupported upsample type {layer_type}')
	return nn.ConvTranspose2d(
		in_channels=in_channels,
		out_channels=out_channels,
		kernel_size=kernel_size,
		stride=stride,
		bias=bias,
		**cfg,
	)
