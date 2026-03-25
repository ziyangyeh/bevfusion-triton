from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from models.helper import build_conv_layer, build_norm_layer, build_upsample_layer

__all__ = ['SECONDFPN', 'SECONDFPNConfig']


@dataclass
class SECONDFPNConfig:
	in_channels: tuple[int, ...] = (128, 256)
	out_channels: tuple[int, ...] = (256, 256)
	upsample_strides: tuple[int, ...] = (1, 2)
	norm_cfg: dict[str, Any] = field(
		default_factory=lambda: dict(type='BN', eps=1e-3, momentum=0.01)
	)
	upsample_cfg: dict[str, Any] = field(default_factory=lambda: dict(type='deconv', bias=False))
	conv_cfg: dict[str, Any] = field(default_factory=lambda: dict(type='Conv2d', bias=False))
	use_conv_for_no_stride: bool = True


class SECONDFPN(nn.Module):
	"""FPN used in SECOND/PointPillars style lidar backbones."""

	def __init__(
		self,
		in_channels: tuple[int, ...] = (128, 256),
		out_channels: tuple[int, ...] = (256, 256),
		upsample_strides: tuple[int, ...] = (1, 2),
		norm_cfg: dict[str, Any] | None = dict(type='BN', eps=1e-3, momentum=0.01),
		upsample_cfg: dict[str, Any] | None = dict(type='deconv', bias=False),
		conv_cfg: dict[str, Any] | None = dict(type='Conv2d', bias=False),
		use_conv_for_no_stride: bool = True,
	) -> None:
		super().__init__()
		if not (len(in_channels) == len(out_channels) == len(upsample_strides)):
			raise ValueError(
				'in_channels, out_channels and upsample_strides must have the same length'
			)

		self.in_channels = tuple(in_channels)
		self.out_channels = tuple(out_channels)
		self.upsample_strides = tuple(upsample_strides)
		self.norm_cfg = norm_cfg
		self.upsample_cfg = upsample_cfg
		self.conv_cfg = conv_cfg
		self.use_conv_for_no_stride = use_conv_for_no_stride

		deblocks = []
		for i, out_channel in enumerate(self.out_channels):
			stride = self.upsample_strides[i]
			if stride > 1 or (stride == 1 and not self.use_conv_for_no_stride):
				upsample_layer = build_upsample_layer(
					self.upsample_cfg,
					in_channels=self.in_channels[i],
					out_channels=out_channel,
					kernel_size=stride,
					stride=stride,
				)
			else:
				upsample_layer = build_conv_layer(
					self.conv_cfg,
					in_channels=self.in_channels[i],
					out_channels=out_channel,
					kernel_size=1,
					stride=1,
					bias=False,
				)

			deblocks.append(
				nn.Sequential(
					upsample_layer,
					build_norm_layer(self.norm_cfg, out_channel)[1],
					nn.ReLU(inplace=True),
				)
			)
		self.deblocks = nn.ModuleList(deblocks)
		self._init_weights()

	@classmethod
	def from_config(cls, cfg: SECONDFPNConfig | dict[str, Any]) -> 'SECONDFPN':
		if isinstance(cfg, SECONDFPNConfig):
			cfg_dict = vars(cfg).copy()
		else:
			cfg_dict = dict(cfg)
		return cls(**cfg_dict)

	def _init_weights(self) -> None:
		for module in self.modules():
			if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
				nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
				if module.bias is not None:
					nn.init.zeros_(module.bias)

	def forward(self, x: list[torch.Tensor] | tuple[torch.Tensor, ...]) -> list[torch.Tensor]:
		if len(x) != len(self.in_channels):
			raise ValueError(f'Expected {len(self.in_channels)} input features, got {len(x)}')
		ups = [deblock(x[i]) for i, deblock in enumerate(self.deblocks)]
		out = torch.cat(ups, dim=1) if len(ups) > 1 else ups[0]
		return [out]


if __name__ == '__main__':
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model = SECONDFPN().to(device).eval()

	inputs = [
		torch.randn(2, 128, 128, 128, device=device),
		torch.randn(2, 256, 64, 64, device=device),
	]

	with torch.no_grad():
		outputs = model(inputs)

	print([out.shape for out in outputs])
