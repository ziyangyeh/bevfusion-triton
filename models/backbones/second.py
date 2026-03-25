# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from models.helper import build_conv_layer, build_norm_layer

__all__ = ['SECOND', 'SECONDConfig']


@dataclass
class SECONDConfig:
	"""Hydra-friendly config for SECOND backbone."""

	in_channels: int = 256
	out_channels: tuple[int, ...] = (128, 256)
	layer_nums: tuple[int, ...] = (5, 5)
	layer_strides: tuple[int, ...] = (1, 2)
	norm_cfg: dict[str, Any] | None = field(
		default_factory=lambda: dict(type='BN', eps=1e-3, momentum=0.01)
	)
	conv_cfg: dict[str, Any] | None = field(default_factory=lambda: dict(type='Conv2d', bias=False))
	init_cfg: dict[str, Any] | None = None
	pretrained: str | None = None


class SECOND(nn.Module):
	"""Backbone network for SECOND/PointPillars/PartA2/MVXNet."""

	def __init__(
		self,
		in_channels: int = 256,
		out_channels: tuple[int, ...] = (128, 256),
		layer_nums: tuple[int, ...] = (5, 5),
		layer_strides: tuple[int, ...] = (1, 2),
		norm_cfg: dict[str, Any] | None = dict(type='BN', eps=1e-3, momentum=0.01),
		conv_cfg: dict[str, Any] | None = dict(type='Conv2d', bias=False),
		init_cfg: dict[str, Any] | None = None,
		pretrained: str | None = None,
	):
		super().__init__()
		if len(layer_strides) != len(layer_nums):
			raise ValueError('layer_strides and layer_nums must have same length')
		if len(out_channels) != len(layer_nums):
			raise ValueError('out_channels and layer_nums must have same length')
		if init_cfg is not None and pretrained is not None:
			raise ValueError('init_cfg and pretrained cannot be set at the same time')

		self.init_cfg = init_cfg
		self.pretrained = pretrained
		if isinstance(pretrained, str):
			warnings.warn(
				'DeprecationWarning: "pretrained" is deprecated, use "init_cfg" instead',
				stacklevel=2,
			)
			self.init_cfg = {'type': 'Pretrained', 'checkpoint': pretrained}
		elif self.init_cfg is None:
			self.init_cfg = {'type': 'Kaiming', 'layer': 'Conv2d'}

		in_filters = [in_channels, *out_channels[:-1]]
		blocks = []
		for i, layer_num in enumerate(layer_nums):
			block_layers: list[nn.Module] = [
				build_conv_layer(
					conv_cfg,
					in_filters[i],
					out_channels[i],
					3,
					stride=layer_strides[i],
					padding=1,
				),
				build_norm_layer(norm_cfg, out_channels[i])[1],
				nn.ReLU(inplace=True),
			]
			# Keep parity with original implementation: extra `layer_num` conv-norm-relu.
			for _ in range(layer_num):
				block_layers.append(
					build_conv_layer(conv_cfg, out_channels[i], out_channels[i], 3, padding=1)
				)
				block_layers.append(build_norm_layer(norm_cfg, out_channels[i])[1])
				block_layers.append(nn.ReLU(inplace=True))

			blocks.append(nn.Sequential(*block_layers))

		self.blocks = nn.ModuleList(blocks)
		self._init_weights_if_needed()

	@classmethod
	def from_config(cls, cfg: SECONDConfig | dict[str, Any]) -> 'SECOND':
		"""Hydra/OmegaConf friendly constructor."""
		if isinstance(cfg, SECONDConfig):
			cfg_dict = vars(cfg).copy()
		else:
			cfg_dict = dict(cfg)
		return cls(**cfg_dict)

	def _init_weights_if_needed(self) -> None:
		cfg = self.init_cfg or {}
		if cfg.get('type') != 'Kaiming':
			return
		layer = cfg.get('layer', 'Conv2d')
		if layer != 'Conv2d':
			return
		for module in self.modules():
			if isinstance(module, nn.Conv2d):
				nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
				if module.bias is not None:
					nn.init.zeros_(module.bias)

	def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
		outs = []
		for block in self.blocks:
			x = block(x)
			outs.append(x)
		return tuple(outs)


if __name__ == '__main__':
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model = SECOND().to(device).eval()

	x = torch.randn(2, 256, 128, 128, device=device)

	with torch.no_grad():
		outs = model(x)

	print([out.shape for out in outs])
