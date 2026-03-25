from __future__ import annotations

from torch import nn

from models.helper import build_norm_layer
from . import spconv

__all__ = ['SparseBasicBlock', 'SparseBottleneck', 'make_sparse_convmodule']


def _build_sparse_conv(
	conv_type, in_channels, out_channels, kernel_size, stride, padding, bias, indice_key
):
	conv_map = {
		'SubMConv3d': spconv.SubMConv3d,
		'SparseConv3d': spconv.SparseConv3d,
		'SparseInverseConv3d': spconv.SparseInverseConv3d,
		'SubMConv2d': spconv.SubMConv2d,
		'SparseConv2d': spconv.SparseConv2d,
		'SparseInverseConv2d': spconv.SparseInverseConv2d,
	}
	if conv_type not in conv_map:
		raise KeyError(f'Unsupported sparse conv type: {conv_type}')

	conv_cls = conv_map[conv_type]
	kwargs = dict(bias=bias, indice_key=indice_key)
	if conv_type not in {'SparseInverseConv3d', 'SparseInverseConv2d'}:
		kwargs.update(stride=stride, padding=padding)
	return conv_cls(in_channels, out_channels, kernel_size, **kwargs)


class SparseBasicBlock(spconv.SparseModule):
	expansion = 1

	def __init__(self, inplanes, planes, stride=1, downsample=None, conv_cfg=None, norm_cfg=None):
		super().__init__()
		conv_type = 'SubMConv3d' if conv_cfg is None else conv_cfg.get('type', 'SubMConv3d')
		indice_key = None if conv_cfg is None else conv_cfg.get('indice_key')

		self.conv1 = _build_sparse_conv(
			conv_type,
			inplanes,
			planes,
			kernel_size=3,
			stride=stride,
			padding=1,
			bias=False,
			indice_key=indice_key,
		)
		self.bn1 = build_norm_layer(norm_cfg, planes)[1]
		self.relu = nn.ReLU(inplace=True)
		self.conv2 = _build_sparse_conv(
			conv_type,
			planes,
			planes,
			kernel_size=3,
			stride=1,
			padding=1,
			bias=False,
			indice_key=indice_key,
		)
		self.bn2 = build_norm_layer(norm_cfg, planes)[1]
		self.downsample = downsample

	@property
	def norm1(self):
		return self.bn1

	@property
	def norm2(self):
		return self.bn2

	def forward(self, x):
		identity = x.features

		out = self.conv1(x)
		out.features = self.bn1(out.features)
		out.features = self.relu(out.features)

		out = self.conv2(out)
		out.features = self.bn2(out.features)

		if self.downsample is not None:
			downsampled = self.downsample(x)
			identity = downsampled.features if hasattr(downsampled, 'features') else downsampled

		out.features = out.features + identity
		out.features = self.relu(out.features)
		return out


class SparseBottleneck(spconv.SparseModule):
	expansion = 4

	def __init__(self, inplanes, planes, stride=1, downsample=None, conv_cfg=None, norm_cfg=None):
		super().__init__()
		conv_type = 'SubMConv3d' if conv_cfg is None else conv_cfg.get('type', 'SubMConv3d')
		indice_key = None if conv_cfg is None else conv_cfg.get('indice_key')
		outplanes = planes * self.expansion

		self.conv1 = _build_sparse_conv(
			conv_type,
			inplanes,
			planes,
			kernel_size=1,
			stride=1,
			padding=0,
			bias=False,
			indice_key=indice_key,
		)
		self.bn1 = build_norm_layer(norm_cfg, planes)[1]
		self.conv2 = _build_sparse_conv(
			conv_type,
			planes,
			planes,
			kernel_size=3,
			stride=stride,
			padding=1,
			bias=False,
			indice_key=indice_key,
		)
		self.bn2 = build_norm_layer(norm_cfg, planes)[1]
		self.conv3 = _build_sparse_conv(
			conv_type,
			planes,
			outplanes,
			kernel_size=1,
			stride=1,
			padding=0,
			bias=False,
			indice_key=indice_key,
		)
		self.bn3 = build_norm_layer(norm_cfg, outplanes)[1]
		self.relu = nn.ReLU(inplace=True)
		self.downsample = downsample

	@property
	def norm1(self):
		return self.bn1

	@property
	def norm2(self):
		return self.bn2

	@property
	def norm3(self):
		return self.bn3

	def forward(self, x):
		identity = x.features

		out = self.conv1(x)
		out.features = self.bn1(out.features)
		out.features = self.relu(out.features)

		out = self.conv2(out)
		out.features = self.bn2(out.features)
		out.features = self.relu(out.features)

		out = self.conv3(out)
		out.features = self.bn3(out.features)

		if self.downsample is not None:
			downsampled = self.downsample(x)
			identity = downsampled.features if hasattr(downsampled, 'features') else downsampled

		out.features = out.features + identity
		out.features = self.relu(out.features)
		return out


def make_sparse_convmodule(
	in_channels,
	out_channels,
	kernel_size,
	indice_key,
	stride=1,
	padding=0,
	conv_type='SubMConv3d',
	norm_cfg=None,
	order=('conv', 'norm', 'act'),
):
	if not isinstance(order, tuple):
		order = tuple(order)
	assert len(order) <= 3
	assert set(order) == {'conv', 'norm', 'act'}

	layers = []
	for layer in order:
		if layer == 'conv':
			layers.append(
				_build_sparse_conv(
					conv_type,
					in_channels,
					out_channels,
					kernel_size=kernel_size,
					stride=stride,
					padding=padding,
					bias=False,
					indice_key=indice_key,
				)
			)
		elif layer == 'norm':
			layers.append(build_norm_layer(norm_cfg, out_channels)[1])
		elif layer == 'act':
			layers.append(nn.ReLU(inplace=True))

	return spconv.SparseSequential(*layers)
