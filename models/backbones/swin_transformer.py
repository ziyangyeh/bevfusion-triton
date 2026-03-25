from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urlparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from models.helper import build_conv_layer, build_norm_layer

__all__ = ['SwinTransformer']


def convert_swin_official_weights(
	state_dict: dict[str, torch.Tensor],
) -> OrderedDict[str, torch.Tensor]:
	converted = OrderedDict()

	def correct_unfold_reduction_order(weight: torch.Tensor) -> torch.Tensor:
		out_channel, in_channel = weight.shape
		weight = weight.reshape(out_channel, 4, in_channel // 4)
		weight = weight[:, [0, 2, 1, 3], :].transpose(1, 2).reshape(out_channel, in_channel)
		return weight

	def correct_unfold_norm_order(weight: torch.Tensor) -> torch.Tensor:
		in_channel = weight.shape[0]
		weight = weight.reshape(4, in_channel // 4)
		weight = weight[[0, 2, 1, 3], :].transpose(0, 1).reshape(in_channel)
		return weight

	for key, value in state_dict.items():
		if (
			key.startswith('head')
			or key.startswith('norm.')
			or key.endswith('relative_position_index')
			or key.endswith('attn_mask')
		):
			continue
		new_key = key
		new_value = value

		if new_key.startswith('patch_embed'):
			new_key = new_key.replace('patch_embed.proj', 'patch_embed.projection')
		elif new_key.startswith('layers'):
			new_key = new_key.replace('layers', 'stages', 1)
			if '.attn.' in new_key:
				new_key = new_key.replace('.attn.', '.attn.w_msa.')
			elif '.mlp.fc1.' in new_key:
				new_key = new_key.replace('.mlp.fc1.', '.ffn.layers.0.0.')
			elif '.mlp.fc2.' in new_key:
				new_key = new_key.replace('.mlp.fc2.', '.ffn.layers.1.')
			elif '.downsample.reduction.' in new_key:
				new_value = correct_unfold_reduction_order(value)
			elif '.downsample.norm.' in new_key:
				new_value = correct_unfold_norm_order(value)

		converted[new_key] = new_value
	return converted


def _ensure_pretrained_checkpoint(pretrained: str) -> Path:
	parsed = urlparse(pretrained)
	if parsed.scheme not in {'http', 'https'}:
		return Path(pretrained)

	checkpoint_dir = Path.cwd() / 'pretrained'
	checkpoint_dir.mkdir(parents=True, exist_ok=True)
	filename = Path(parsed.path).name
	checkpoint_path = checkpoint_dir / filename

	if checkpoint_path.exists():
		return checkpoint_path

	torch.hub.download_url_to_file(pretrained, str(checkpoint_path), progress=True)
	return checkpoint_path


def to_2tuple(value: int | tuple[int, int]) -> tuple[int, int]:
	if isinstance(value, tuple):
		return value
	return (value, value)


class AdaptivePadding(nn.Module):
	"""Pad inputs so patch embedding / merging can cover the whole feature map."""

	def __init__(
		self,
		kernel_size: int | tuple[int, int] = 1,
		stride: int | tuple[int, int] = 1,
		dilation: int | tuple[int, int] = 1,
		padding: str = 'corner',
	) -> None:
		super().__init__()
		if padding not in {'same', 'corner'}:
			raise ValueError(f'Unsupported adaptive padding mode: {padding}')
		self.kernel_size = to_2tuple(kernel_size)
		self.stride = to_2tuple(stride)
		self.dilation = to_2tuple(dilation)
		self.padding = padding

	def get_pad_shape(self, input_shape: tuple[int, int]) -> tuple[int, int]:
		input_h, input_w = input_shape
		kernel_h, kernel_w = self.kernel_size
		stride_h, stride_w = self.stride
		output_h = (input_h + stride_h - 1) // stride_h
		output_w = (input_w + stride_w - 1) // stride_w
		pad_h = max((output_h - 1) * stride_h + (kernel_h - 1) * self.dilation[0] + 1 - input_h, 0)
		pad_w = max((output_w - 1) * stride_w + (kernel_w - 1) * self.dilation[1] + 1 - input_w, 0)
		return pad_h, pad_w

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		pad_h, pad_w = self.get_pad_shape((x.shape[-2], x.shape[-1]))
		if pad_h == 0 and pad_w == 0:
			return x
		if self.padding == 'corner':
			return F.pad(x, [0, pad_w, 0, pad_h])
		return F.pad(
			x,
			[pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2],
		)


class PatchEmbed(nn.Module):
	"""Conv patch embedding used by the MMDet Swin backbone."""

	def __init__(
		self,
		in_channels: int = 3,
		embed_dims: int = 96,
		kernel_size: int | tuple[int, int] = 4,
		stride: int | tuple[int, int] = 4,
		padding: int | tuple[int, int] | str = 'corner',
		norm_cfg: dict[str, Any] | None = None,
	) -> None:
		super().__init__()
		kernel_size = to_2tuple(kernel_size)
		stride = to_2tuple(stride)
		if isinstance(padding, str):
			self.adap_padding = AdaptivePadding(
				kernel_size=kernel_size, stride=stride, padding=padding
			)
			conv_padding = 0
		else:
			self.adap_padding = None
			conv_padding = padding
		self.projection = build_conv_layer(
			{'type': 'Conv2d'},
			in_channels=in_channels,
			out_channels=embed_dims,
			kernel_size=kernel_size,
			stride=stride,
			padding=conv_padding,
			bias=True,
		)
		self.norm = build_norm_layer(norm_cfg, embed_dims)[1] if norm_cfg is not None else None

	def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
		if self.adap_padding is not None:
			x = self.adap_padding(x)
		x = self.projection(x)
		out_size = (x.shape[2], x.shape[3])
		x = x.flatten(2).transpose(1, 2).contiguous()
		if self.norm is not None:
			x = self.norm(x)
		return x, out_size


class PatchMerging(nn.Module):
	"""Unfold-based patch merging from the MMDet Swin implementation."""

	def __init__(
		self,
		in_channels: int,
		out_channels: int,
		kernel_size: int | tuple[int, int] = 2,
		stride: int | tuple[int, int] | None = None,
		padding: int | tuple[int, int] | str = 'corner',
		norm_cfg: dict[str, Any] | None = dict(type='LN'),
	) -> None:
		super().__init__()
		kernel_size = to_2tuple(kernel_size)
		stride = kernel_size if stride is None else to_2tuple(stride)
		self.in_channels = in_channels
		self.out_channels = out_channels

		if isinstance(padding, str):
			self.adap_padding = AdaptivePadding(
				kernel_size=kernel_size, stride=stride, padding=padding
			)
			unfold_padding = 0
		else:
			self.adap_padding = None
			unfold_padding = padding

		self.sampler = nn.Unfold(kernel_size=kernel_size, stride=stride, padding=unfold_padding)
		sample_dim = kernel_size[0] * kernel_size[1] * in_channels
		self.norm = build_norm_layer(norm_cfg, sample_dim)[1] if norm_cfg is not None else None
		self.reduction = nn.Linear(sample_dim, out_channels, bias=False)

	def forward(
		self, x: torch.Tensor, input_size: tuple[int, int]
	) -> tuple[torch.Tensor, tuple[int, int]]:
		batch_size, length, channels = x.shape
		height, width = input_size
		if length != height * width:
			raise ValueError('Input sequence length does not match spatial shape.')

		x = x.view(batch_size, height, width, channels).permute(0, 3, 1, 2).contiguous()
		if self.adap_padding is not None:
			x = self.adap_padding(x)
			height, width = x.shape[-2:]

		x = self.sampler(x).transpose(1, 2).contiguous()
		if self.norm is not None:
			x = self.norm(x)
		x = self.reduction(x)

		padding = to_2tuple(self.sampler.padding)
		dilation = to_2tuple(self.sampler.dilation)
		kernel_size = to_2tuple(self.sampler.kernel_size)
		stride = to_2tuple(self.sampler.stride)
		out_h = (height + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) // stride[0] + 1
		out_w = (width + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) // stride[1] + 1
		return x, (out_h, out_w)


class WindowMSA(nn.Module):
	def __init__(
		self,
		embed_dims: int,
		num_heads: int,
		window_size: tuple[int, int],
		qkv_bias: bool = True,
		qk_scale: float | None = None,
		attn_drop_rate: float = 0.0,
		proj_drop_rate: float = 0.0,
	) -> None:
		super().__init__()
		self.embed_dims = embed_dims
		self.window_size = window_size
		self.num_heads = num_heads
		head_dim = embed_dims // num_heads
		self.scale = qk_scale or head_dim**-0.5

		self.relative_position_bias_table = nn.Parameter(
			torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
		)

		coords_h = torch.arange(window_size[0])
		coords_w = torch.arange(window_size[1])
		coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))
		coords_flatten = coords.flatten(1)
		relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
		relative_coords = relative_coords.permute(1, 2, 0).contiguous()
		relative_coords[:, :, 0] += window_size[0] - 1
		relative_coords[:, :, 1] += window_size[1] - 1
		relative_coords[:, :, 0] *= 2 * window_size[1] - 1
		self.register_buffer('relative_position_index', relative_coords.sum(-1), persistent=True)

		self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=qkv_bias)
		self.attn_drop = nn.Dropout(attn_drop_rate)
		self.proj = nn.Linear(embed_dims, embed_dims)
		self.proj_drop = nn.Dropout(proj_drop_rate)
		self.softmax = nn.Softmax(dim=-1)

	def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
		batch_windows, num_tokens, channels = x.shape
		qkv = self.qkv(x).reshape(
			batch_windows, num_tokens, 3, self.num_heads, channels // self.num_heads
		)
		qkv = qkv.permute(2, 0, 3, 1, 4)
		q, k, v = qkv[0], qkv[1], qkv[2]

		q = q * self.scale
		attn = q @ k.transpose(-2, -1)
		rel_pos_bias = self.relative_position_bias_table[self.relative_position_index.reshape(-1)]
		rel_pos_bias = rel_pos_bias.view(num_tokens, num_tokens, -1).permute(2, 0, 1).contiguous()
		attn = attn + rel_pos_bias.unsqueeze(0)

		if mask is not None:
			num_windows = mask.shape[0]
			attn = attn.view(
				batch_windows // num_windows, num_windows, self.num_heads, num_tokens, num_tokens
			)
			attn = attn + mask.unsqueeze(1).unsqueeze(0)
			attn = attn.view(-1, self.num_heads, num_tokens, num_tokens)

		attn = self.softmax(attn)
		attn = self.attn_drop(attn)
		x = (attn @ v).transpose(1, 2).reshape(batch_windows, num_tokens, channels)
		x = self.proj(x)
		x = self.proj_drop(x)
		return x


class ShiftWindowMSA(nn.Module):
	def __init__(
		self,
		embed_dims: int,
		num_heads: int,
		window_size: int,
		shift_size: int = 0,
		qkv_bias: bool = True,
		qk_scale: float | None = None,
		attn_drop_rate: float = 0.0,
		proj_drop_rate: float = 0.0,
		drop_path_rate: float = 0.0,
	) -> None:
		super().__init__()
		self.window_size = window_size
		self.shift_size = shift_size
		self.w_msa = WindowMSA(
			embed_dims=embed_dims,
			num_heads=num_heads,
			window_size=(window_size, window_size),
			qkv_bias=qkv_bias,
			qk_scale=qk_scale,
			attn_drop_rate=attn_drop_rate,
			proj_drop_rate=proj_drop_rate,
		)
		self.drop = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

	def window_partition(self, x: torch.Tensor) -> torch.Tensor:
		batch_size, height, width, channels = x.shape
		window_size = self.window_size
		x = x.view(
			batch_size,
			height // window_size,
			window_size,
			width // window_size,
			window_size,
			channels,
		)
		windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
		return windows.view(-1, window_size, window_size, channels)

	def window_reverse(self, windows: torch.Tensor, height: int, width: int) -> torch.Tensor:
		window_size = self.window_size
		batch_size = int(windows.shape[0] / (height * width / window_size / window_size))
		x = windows.view(
			batch_size, height // window_size, width // window_size, window_size, window_size, -1
		)
		x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
		return x.view(batch_size, height, width, -1)

	def forward(self, query: torch.Tensor, hw_shape: tuple[int, int]) -> torch.Tensor:
		batch_size, length, channels = query.shape
		height, width = hw_shape
		if length != height * width:
			raise ValueError('Input feature has wrong size.')
		query = query.view(batch_size, height, width, channels)

		pad_r = (self.window_size - width % self.window_size) % self.window_size
		pad_b = (self.window_size - height % self.window_size) % self.window_size
		query = F.pad(query, (0, 0, 0, pad_r, 0, pad_b))
		height_pad, width_pad = query.shape[1], query.shape[2]

		if self.shift_size > 0:
			shifted_query = torch.roll(
				query, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)
			)
			img_mask = torch.zeros((1, height_pad, width_pad, 1), device=query.device)
			h_slices = (
				slice(0, -self.window_size),
				slice(-self.window_size, -self.shift_size),
				slice(-self.shift_size, None),
			)
			w_slices = (
				slice(0, -self.window_size),
				slice(-self.window_size, -self.shift_size),
				slice(-self.shift_size, None),
			)
			count = 0
			for h_slice in h_slices:
				for w_slice in w_slices:
					img_mask[:, h_slice, w_slice, :] = count
					count += 1
			mask_windows = self.window_partition(img_mask).view(
				-1, self.window_size * self.window_size
			)
			attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
			attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0))
			attn_mask = attn_mask.masked_fill(attn_mask == 0, float(0.0))
		else:
			shifted_query = query
			attn_mask = None

		query_windows = self.window_partition(shifted_query)
		query_windows = query_windows.view(-1, self.window_size * self.window_size, channels)
		attn_windows = self.w_msa(query_windows, mask=attn_mask)
		attn_windows = attn_windows.view(-1, self.window_size, self.window_size, channels)

		shifted_x = self.window_reverse(attn_windows, height_pad, width_pad)
		if self.shift_size > 0:
			x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
		else:
			x = shifted_x

		if pad_r > 0 or pad_b > 0:
			x = x[:, :height, :width, :].contiguous()
		x = x.view(batch_size, height * width, channels)
		return self.drop(x)


class SwinFFN(nn.Module):
	def __init__(
		self,
		embed_dims: int,
		feedforward_channels: int,
		drop_rate: float = 0.0,
		drop_path_rate: float = 0.0,
		act_cfg: dict[str, Any] | None = None,
	) -> None:
		super().__init__()
		act_type = (act_cfg or {}).get('type', 'GELU')
		if act_type != 'GELU':
			raise NotImplementedError(f'Only GELU is supported in local Swin FFN, got {act_type}.')
		# Keep the layer naming close to the MMDet checkpoint layout:
		# ffn.layers.0.0 -> first linear, ffn.layers.1 -> second linear.
		self.layers = nn.Sequential(
			nn.Sequential(
				nn.Linear(embed_dims, feedforward_channels),
				nn.GELU(),
				nn.Dropout(drop_rate),
			),
			nn.Linear(feedforward_channels, embed_dims),
			nn.Dropout(drop_rate),
		)
		self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

	def forward(self, x: torch.Tensor, identity: torch.Tensor) -> torch.Tensor:
		x = self.layers[0](x)
		x = self.layers[1](x)
		x = self.layers[2](x)
		return identity + self.drop_path(x)


class SwinBlock(nn.Module):
	def __init__(
		self,
		embed_dims: int,
		num_heads: int,
		feedforward_channels: int,
		window_size: int = 7,
		shift: bool = False,
		qkv_bias: bool = True,
		qk_scale: float | None = None,
		drop_rate: float = 0.0,
		attn_drop_rate: float = 0.0,
		drop_path_rate: float = 0.0,
		act_cfg: dict[str, Any] | None = None,
		norm_cfg: dict[str, Any] | None = None,
		with_cp: bool = False,
	) -> None:
		super().__init__()
		self.with_cp = with_cp
		self.norm1 = build_norm_layer(norm_cfg, embed_dims)[1]
		self.attn = ShiftWindowMSA(
			embed_dims=embed_dims,
			num_heads=num_heads,
			window_size=window_size,
			shift_size=window_size // 2 if shift else 0,
			qkv_bias=qkv_bias,
			qk_scale=qk_scale,
			attn_drop_rate=attn_drop_rate,
			proj_drop_rate=drop_rate,
			drop_path_rate=drop_path_rate,
		)
		self.norm2 = build_norm_layer(norm_cfg, embed_dims)[1]
		self.ffn = SwinFFN(
			embed_dims=embed_dims,
			feedforward_channels=feedforward_channels,
			drop_rate=drop_rate,
			drop_path_rate=drop_path_rate,
			act_cfg=act_cfg,
		)

	def forward(self, x: torch.Tensor, hw_shape: tuple[int, int]) -> torch.Tensor:
		def _inner_forward(x: torch.Tensor) -> torch.Tensor:
			identity = x
			x = self.norm1(x)
			x = self.attn(x, hw_shape)
			x = x + identity
			identity = x
			x = self.norm2(x)
			x = self.ffn(x, identity)
			return x

		if self.with_cp and x.requires_grad:
			return checkpoint.checkpoint(_inner_forward, x)
		return _inner_forward(x)


class SwinBlockSequence(nn.Module):
	def __init__(
		self,
		embed_dims: int,
		num_heads: int,
		feedforward_channels: int,
		depth: int,
		window_size: int = 7,
		qkv_bias: bool = True,
		qk_scale: float | None = None,
		drop_rate: float = 0.0,
		attn_drop_rate: float = 0.0,
		drop_path_rate: float | Sequence[float] = 0.0,
		downsample: nn.Module | None = None,
		act_cfg: dict[str, Any] | None = None,
		norm_cfg: dict[str, Any] | None = None,
		with_cp: bool = False,
	) -> None:
		super().__init__()
		if isinstance(drop_path_rate, Sequence):
			drop_path_rates = list(drop_path_rate)
		else:
			drop_path_rates = [drop_path_rate] * depth

		self.blocks = nn.ModuleList(
			[
				SwinBlock(
					embed_dims=embed_dims,
					num_heads=num_heads,
					feedforward_channels=feedforward_channels,
					window_size=window_size,
					shift=(block_idx % 2 == 1),
					qkv_bias=qkv_bias,
					qk_scale=qk_scale,
					drop_rate=drop_rate,
					attn_drop_rate=attn_drop_rate,
					drop_path_rate=drop_path_rates[block_idx],
					act_cfg=act_cfg,
					norm_cfg=norm_cfg,
					with_cp=with_cp,
				)
				for block_idx in range(depth)
			]
		)
		self.downsample = downsample

	def forward(
		self,
		x: torch.Tensor,
		hw_shape: tuple[int, int],
	) -> tuple[torch.Tensor, tuple[int, int], torch.Tensor, tuple[int, int]]:
		for block in self.blocks:
			x = block(x, hw_shape)

		if self.downsample is None:
			return x, hw_shape, x, hw_shape
		x_down, down_hw_shape = self.downsample(x, hw_shape)
		return x_down, down_hw_shape, x, hw_shape


class DropPath(nn.Module):
	def __init__(self, drop_prob: float = 0.0) -> None:
		super().__init__()
		self.drop_prob = float(drop_prob)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		if self.drop_prob == 0.0 or not self.training:
			return x
		keep_prob = 1 - self.drop_prob
		shape = (x.shape[0],) + (1,) * (x.ndim - 1)
		random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
		random_tensor.floor_()
		return x.div(keep_prob) * random_tensor


class SwinTransformer(nn.Module):
	"""Local MMDet-style Swin backbone for detection / BEVFusion use."""

	def __init__(
		self,
		pretrain_img_size: int | tuple[int, int] = 224,
		in_channels: int = 3,
		embed_dims: int = 96,
		patch_size: int = 4,
		window_size: int = 7,
		mlp_ratio: int = 4,
		depths: tuple[int, ...] = (2, 2, 6, 2),
		num_heads: tuple[int, ...] = (3, 6, 12, 24),
		strides: tuple[int, ...] = (4, 2, 2, 2),
		out_indices: tuple[int, ...] = (0, 1, 2, 3),
		qkv_bias: bool = True,
		qk_scale: float | None = None,
		patch_norm: bool = True,
		drop_rate: float = 0.0,
		attn_drop_rate: float = 0.0,
		drop_path_rate: float = 0.1,
		use_abs_pos_embed: bool = False,
		act_cfg: dict[str, Any] | None = None,
		norm_cfg: dict[str, Any] | None = None,
		with_cp: bool = False,
		pretrained: str | None = None,
		convert_weights: bool = False,
		frozen_stages: int = -1,
		init_cfg: dict[str, Any] | None = None,
	) -> None:
		super().__init__()
		del init_cfg
		self.convert_weights = convert_weights
		self.frozen_stages = frozen_stages
		self.out_indices = tuple(out_indices)
		self.use_abs_pos_embed = use_abs_pos_embed
		norm_cfg = norm_cfg or {'type': 'LN'}
		act_cfg = act_cfg or {'type': 'GELU'}

		if isinstance(pretrain_img_size, int):
			pretrain_img_size = to_2tuple(pretrain_img_size)
		self.pretrain_img_size = pretrain_img_size

		self.patch_embed = PatchEmbed(
			in_channels=in_channels,
			embed_dims=embed_dims,
			kernel_size=patch_size,
			stride=strides[0],
			norm_cfg=norm_cfg if patch_norm else None,
		)

		if self.use_abs_pos_embed:
			patch_row = pretrain_img_size[0] // patch_size
			patch_col = pretrain_img_size[1] // patch_size
			self.absolute_pos_embed = nn.Parameter(
				torch.zeros(1, patch_row * patch_col, embed_dims)
			)
		else:
			self.absolute_pos_embed = None

		self.drop_after_pos = nn.Dropout(p=drop_rate)
		total_depth = sum(depths)
		dpr = torch.linspace(0, drop_path_rate, total_depth).tolist()

		self.stages = nn.ModuleList()
		in_channels_stage = embed_dims
		start = 0
		for stage_idx, depth in enumerate(depths):
			if stage_idx < len(depths) - 1:
				downsample = PatchMerging(
					in_channels=in_channels_stage,
					out_channels=2 * in_channels_stage,
					stride=strides[stage_idx + 1],
					norm_cfg=norm_cfg if patch_norm else None,
				)
			else:
				downsample = None

			stage = SwinBlockSequence(
				embed_dims=in_channels_stage,
				num_heads=num_heads[stage_idx],
				feedforward_channels=mlp_ratio * in_channels_stage,
				depth=depth,
				window_size=window_size,
				qkv_bias=qkv_bias,
				qk_scale=qk_scale,
				drop_rate=drop_rate,
				attn_drop_rate=attn_drop_rate,
				drop_path_rate=dpr[start : start + depth],
				downsample=downsample,
				act_cfg=act_cfg,
				norm_cfg=norm_cfg,
				with_cp=with_cp,
			)
			self.stages.append(stage)
			start += depth
			if downsample is not None:
				in_channels_stage = downsample.out_channels

		self.num_features = [int(embed_dims * 2**i) for i in range(len(depths))]
		for out_idx in self.out_indices:
			self.add_module(
				f'norm{out_idx}', build_norm_layer(norm_cfg, self.num_features[out_idx])[1]
			)

		self._init_weights()
		if pretrained:
			self.init_weights(pretrained=pretrained)

	def _init_weights(self) -> None:
		if self.absolute_pos_embed is not None:
			nn.init.trunc_normal_(self.absolute_pos_embed, std=0.02)
		for module in self.modules():
			if isinstance(module, nn.Linear):
				nn.init.trunc_normal_(module.weight, std=0.02)
				if module.bias is not None:
					nn.init.constant_(module.bias, 0)
			elif isinstance(module, nn.LayerNorm):
				nn.init.constant_(module.weight, 1.0)
				nn.init.constant_(module.bias, 0)

	def init_weights(self, pretrained: str) -> None:
		checkpoint_path = _ensure_pretrained_checkpoint(pretrained)
		state_dict = torch.load(checkpoint_path, map_location='cpu')
		if 'state_dict' in state_dict:
			state_dict = state_dict['state_dict']
		elif 'model' in state_dict:
			state_dict = state_dict['model']
		state_dict = convert_swin_official_weights(state_dict)
		incompatible = self.load_state_dict(state_dict, strict=False)
		missing = list(incompatible.missing_keys)
		unexpected = list(incompatible.unexpected_keys)
		print(
			f'Loaded Swin pretrained weights from {checkpoint_path} '
			f'(missing={len(missing)}, unexpected={len(unexpected)})'
		)
		if missing:
			print('Missing keys:', missing[:10])
		if unexpected:
			print('Unexpected keys:', unexpected[:10])

	def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
		x, hw_shape = self.patch_embed(x)

		if self.absolute_pos_embed is not None:
			if self.absolute_pos_embed.shape[1] == x.shape[1]:
				x = x + self.absolute_pos_embed
		x = self.drop_after_pos(x)

		outs: list[torch.Tensor] = []
		for stage_idx, stage in enumerate(self.stages):
			x, hw_shape, out, out_hw_shape = stage(x, hw_shape)
			if stage_idx in self.out_indices:
				norm_layer = getattr(self, f'norm{stage_idx}')
				out = norm_layer(out)
				out = out.view(-1, out_hw_shape[0], out_hw_shape[1], self.num_features[stage_idx])
				out = out.permute(0, 3, 1, 2).contiguous()
				outs.append(out)

		return outs


if __name__ == '__main__':
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model = (
		SwinTransformer(
			pretrain_img_size=(256, 704),
			embed_dims=96,
			depths=(2, 2, 6, 2),
			num_heads=(3, 6, 12, 24),
			out_indices=(1, 2, 3),
		)
		.to(device)
		.eval()
	)

	x = torch.randn(2, 3, 256, 704, device=device)
	with torch.no_grad():
		outputs = model(x)

	print([out.shape for out in outputs])
