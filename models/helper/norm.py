from __future__ import annotations

import inspect
from typing import Any

import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.instancenorm import _InstanceNorm

__all__ = ['build_norm_layer', 'infer_norm_abbr']


def infer_norm_abbr(norm_cls: type[nn.Module]) -> str:
	if hasattr(norm_cls, '_abbr_'):
		return norm_cls._abbr_
	if issubclass(norm_cls, _InstanceNorm):
		return 'in'
	if issubclass(norm_cls, _BatchNorm):
		return 'bn'
	if issubclass(norm_cls, nn.GroupNorm):
		return 'gn'
	if issubclass(norm_cls, nn.LayerNorm):
		return 'ln'

	class_name = norm_cls.__name__.lower()
	if 'batch' in class_name:
		return 'bn'
	if 'instance' in class_name:
		return 'in'
	if 'group' in class_name:
		return 'gn'
	if 'layer' in class_name:
		return 'ln'
	return 'norm'


def build_norm_layer(
	norm_cfg: dict[str, Any] | None,
	num_features: int,
	postfix: int | str = '',
) -> tuple[str, nn.Module]:
	if norm_cfg is None:
		raise ValueError('norm_cfg is required')

	cfg = dict(norm_cfg)
	norm_type = str(cfg.pop('type'))
	requires_grad = bool(cfg.pop('requires_grad', True))
	eps = float(cfg.pop('eps', 1e-5))
	momentum = float(cfg.pop('momentum', 0.1))

	norm_map: dict[str, type[nn.Module]] = {
		'BN': nn.BatchNorm2d,
		'BN1d': nn.BatchNorm1d,
		'BN2d': nn.BatchNorm2d,
		'BN3d': nn.BatchNorm3d,
		'BatchNorm1d': nn.BatchNorm1d,
		'BatchNorm2d': nn.BatchNorm2d,
		'BatchNorm3d': nn.BatchNorm3d,
		'IN': nn.InstanceNorm2d,
		'IN1d': nn.InstanceNorm1d,
		'IN2d': nn.InstanceNorm2d,
		'IN3d': nn.InstanceNorm3d,
		'InstanceNorm1d': nn.InstanceNorm1d,
		'InstanceNorm2d': nn.InstanceNorm2d,
		'InstanceNorm3d': nn.InstanceNorm3d,
		'GN': nn.GroupNorm,
		'LN': nn.LayerNorm,
	}
	if norm_type not in norm_map:
		raise KeyError(f'Unrecognized norm type {norm_type}')

	norm_cls = norm_map[norm_type]
	name = f'{infer_norm_abbr(norm_cls)}{postfix}'

	if norm_type == 'GN':
		num_groups = cfg.pop('num_groups', None)
		if num_groups is None:
			raise KeyError('num_groups must be specified for GN')
		layer = norm_cls(num_groups=num_groups, num_channels=num_features, eps=eps, **cfg)
	elif norm_type == 'LN':
		layer = norm_cls(num_features, eps=eps, **cfg)
	else:
		sig = inspect.signature(norm_cls)
		kwargs = dict(cfg)
		if 'eps' in sig.parameters:
			kwargs['eps'] = eps
		if 'momentum' in sig.parameters:
			kwargs['momentum'] = momentum
		layer = norm_cls(num_features, **kwargs)

	for param in layer.parameters():
		param.requires_grad = requires_grad
	return name, layer
