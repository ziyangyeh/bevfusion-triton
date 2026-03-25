from __future__ import annotations

import warnings
from typing import Any

import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.instancenorm import _InstanceNorm

from .conv import build_conv_layer
from .norm import build_norm_layer

__all__ = ['ConvModule']


class ConvModule(nn.Module):
	"""Pure PyTorch version of MMCV-style ConvModule."""

	_abbr_ = 'conv_block'

	def __init__(
		self,
		in_channels: int,
		out_channels: int,
		kernel_size: int | tuple[int, ...],
		stride: int | tuple[int, ...] = 1,
		padding: int | tuple[int, ...] = 0,
		dilation: int | tuple[int, ...] = 1,
		groups: int = 1,
		bias: bool | str = 'auto',
		conv_cfg: dict[str, Any] | None = None,
		norm_cfg: dict[str, Any] | None = None,
		act_cfg: dict[str, Any] | None = dict(type='ReLU'),
		inplace: bool = True,
		with_spectral_norm: bool = False,
		padding_mode: str = 'zeros',
		order: tuple[str, str, str] = ('conv', 'norm', 'act'),
	):
		super().__init__()
		assert conv_cfg is None or isinstance(conv_cfg, dict)
		assert norm_cfg is None or isinstance(norm_cfg, dict)
		assert act_cfg is None or isinstance(act_cfg, dict)
		assert isinstance(order, tuple) and len(order) == 3
		assert set(order) == {'conv', 'norm', 'act'}

		self.conv_cfg = conv_cfg
		self.norm_cfg = norm_cfg
		self.act_cfg = act_cfg
		self.inplace = inplace
		self.with_spectral_norm = with_spectral_norm
		self.order = order
		self.with_norm = norm_cfg is not None
		self.with_activation = act_cfg is not None
		self.with_explicit_padding = padding_mode not in {'zeros', 'circular'}

		# Match MMCV behavior: disable conv bias when a norm layer follows by default.
		if bias == 'auto':
			bias = not self.with_norm
		self.with_bias = bool(bias)

		# Explicit padding is kept as a separate module when PyTorch conv padding_mode
		# does not cover the requested behavior.
		if self.with_explicit_padding:
			if padding_mode == 'reflect':
				self.padding_layer = nn.ReflectionPad2d(padding)
			elif padding_mode == 'replicate':
				self.padding_layer = nn.ReplicationPad2d(padding)
			elif padding_mode == 'zero':
				self.padding_layer = nn.ZeroPad2d(padding)
			else:
				raise ValueError(f'Unsupported padding type: {padding_mode}')

		conv_padding = 0 if self.with_explicit_padding else padding
		self.conv = build_conv_layer(
			conv_cfg,
			in_channels,
			out_channels,
			kernel_size,
			stride=stride,
			padding=conv_padding,
			dilation=dilation,
			groups=groups,
			bias=self.with_bias,
		)

		# Mirror common conv attributes for compatibility with MMCV callers.
		self.in_channels = self.conv.in_channels
		self.out_channels = self.conv.out_channels
		self.kernel_size = self.conv.kernel_size
		self.stride = self.conv.stride
		self.padding = padding
		self.dilation = self.conv.dilation
		self.transposed = self.conv.transposed
		self.output_padding = self.conv.output_padding
		self.groups = self.conv.groups

		if self.with_spectral_norm:
			self.conv = nn.utils.spectral_norm(self.conv)

		# Build normalization inline and keep the MMCV-style attribute access.
		if self.with_norm:
			norm_channels = (
				out_channels if order.index('norm') > order.index('conv') else in_channels
			)
			self.norm_name, norm = build_norm_layer(norm_cfg, norm_channels)
			self.add_module(self.norm_name, norm)
			if self.with_bias and isinstance(norm, (_BatchNorm, _InstanceNorm)):
				warnings.warn('Unnecessary conv bias before batch/instance norm', stacklevel=2)
		else:
			self.norm_name = None

		# Build activation inline for the small set used in this repo.
		if self.with_activation:
			act_cfg_ = dict(act_cfg)
			act_type = str(act_cfg_.pop('type'))
			if act_type not in {'Tanh', 'PReLU', 'Sigmoid', 'HSigmoid', 'Swish'}:
				act_cfg_.setdefault('inplace', inplace)

			if act_type == 'ReLU':
				self.activate = nn.ReLU(**act_cfg_)
			elif act_type == 'LeakyReLU':
				self.activate = nn.LeakyReLU(**act_cfg_)
			elif act_type == 'GELU':
				self.activate = nn.GELU()
			elif act_type == 'PReLU':
				self.activate = nn.PReLU(**act_cfg_)
			elif act_type == 'Sigmoid':
				self.activate = nn.Sigmoid()
			elif act_type == 'Tanh':
				self.activate = nn.Tanh()
			elif act_type in {'SiLU', 'Swish'}:
				self.activate = nn.SiLU(**act_cfg_)
			elif act_type == 'HSigmoid':
				self.activate = nn.Hardsigmoid(**act_cfg_)
			else:
				raise KeyError(f'Unrecognized activation type {act_type}')

		self.init_weights()

	@property
	def norm(self) -> nn.Module | None:
		return getattr(self, self.norm_name) if self.norm_name else None

	def init_weights(self) -> None:
		# Keep the default MMCV convention: Kaiming init for conv, constant 1 for norm.
		if not hasattr(self.conv, 'init_weights'):
			if (
				self.with_activation
				and self.act_cfg is not None
				and self.act_cfg['type'] == 'LeakyReLU'
			):
				nonlinearity = 'leaky_relu'
				a = self.act_cfg.get('negative_slope', 0.01)
			else:
				nonlinearity = 'relu'
				a = 0
			nn.init.kaiming_normal_(
				self.conv.weight, a=a, mode='fan_out', nonlinearity=nonlinearity
			)
			if self.conv.bias is not None:
				nn.init.constant_(self.conv.bias, 0)
		if self.with_norm and self.norm is not None:
			if hasattr(self.norm, 'weight') and self.norm.weight is not None:
				nn.init.constant_(self.norm.weight, 1)
			if hasattr(self.norm, 'bias') and self.norm.bias is not None:
				nn.init.constant_(self.norm.bias, 0)

	def forward(self, x, activate: bool = True, norm: bool = True):
		for layer in self.order:
			if layer == 'conv':
				if self.with_explicit_padding:
					x = self.padding_layer(x)
				x = self.conv(x)
			elif layer == 'norm' and norm and self.with_norm:
				x = self.norm(x)
			elif layer == 'act' and activate and self.with_activation:
				x = self.activate(x)
		return x
