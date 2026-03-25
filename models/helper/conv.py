from __future__ import annotations

from typing import Any

import torch.nn as nn

__all__ = ['build_conv_layer']


def build_conv_layer(
	conv_cfg: dict[str, Any] | None,
	in_channels: int,
	out_channels: int,
	kernel_size: int | tuple[int, ...],
	stride: int | tuple[int, ...] = 1,
	padding: int | tuple[int, ...] = 0,
	dilation: int | tuple[int, ...] = 1,
	groups: int = 1,
	bias: bool = False,
) -> nn.Module:
	cfg = {} if conv_cfg is None else dict(conv_cfg)
	conv_type = str(cfg.pop('type', 'Conv2d'))
	bias = bool(cfg.pop('bias', bias))

	conv_map = {
		'Conv1d': nn.Conv1d,
		'Conv2d': nn.Conv2d,
		'Conv3d': nn.Conv3d,
		'Conv': nn.Conv2d,
	}
	if conv_type not in conv_map:
		raise KeyError(f'Unrecognized conv type {conv_type}')

	conv_cls = conv_map[conv_type]
	return conv_cls(
		in_channels=in_channels,
		out_channels=out_channels,
		kernel_size=kernel_size,
		stride=stride,
		padding=padding,
		dilation=dilation,
		groups=groups,
		bias=bias,
		**cfg,
	)
