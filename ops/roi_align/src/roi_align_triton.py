from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair

from ops.common_helper.src.common_triton_helper import (
	bilinear_interpolate_gradient_kernel,
	bilinear_interpolate_kernel,
)

_POOL_MODE_TO_ID = {'max': 0, 'avg': 1}
_BLOCK_SIZE = 128


def _pre_calc_for_bilinear_interpolate(
	height: int,
	width: int,
	pooled_h: int,
	pooled_w: int,
	roi_start_h: torch.Tensor,
	roi_start_w: torch.Tensor,
	bin_size_h: torch.Tensor,
	bin_size_w: torch.Tensor,
	roi_bin_grid_h: int,
	roi_bin_grid_w: int,
	dtype: torch.dtype,
	device: torch.device,
):
	pre_calc: list[
		tuple[int, int, int, int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
	] = []
	for ph in range(pooled_h):
		for pw in range(pooled_w):
			for iy in range(roi_bin_grid_h):
				yy = (
					roi_start_h + ph * bin_size_h + (iy + 0.5) * bin_size_h / max(roi_bin_grid_h, 1)
				)
				for ix in range(roi_bin_grid_w):
					xx = (
						roi_start_w
						+ pw * bin_size_w
						+ (ix + 0.5) * bin_size_w / max(roi_bin_grid_w, 1)
					)
					y = yy
					x = xx
					if bool((y < -1.0) or (y > height) or (x < -1.0) or (x > width)):
						zero = torch.zeros((), device=device, dtype=dtype)
						pre_calc.append((0, 0, 0, 0, zero, zero, zero, zero))
						continue

					y = torch.clamp(y, min=0.0)
					x = torch.clamp(x, min=0.0)

					y_low = int(y.item())
					x_low = int(x.item())

					if y_low >= height - 1:
						y_low = height - 1
						y_high = y_low
						y = y.new_tensor(float(y_low))
					else:
						y_high = y_low + 1

					if x_low >= width - 1:
						x_low = width - 1
						x_high = x_low
						x = x.new_tensor(float(x_low))
					else:
						x_high = x_low + 1

					ly = y - y_low
					lx = x - x_low
					hy = 1.0 - ly
					hx = 1.0 - lx
					pre_calc.append(
						(
							y_low * width + x_low,
							y_low * width + x_high,
							y_high * width + x_low,
							y_high * width + x_high,
							hy * hx,
							hy * lx,
							ly * hx,
							ly * lx,
						)
					)
	return pre_calc


def _roi_align_cpu_fallback(
	input: torch.Tensor,
	rois: torch.Tensor,
	output_size: tuple[int, int],
	spatial_scale: float = 1.0,
	sampling_ratio: int = 0,
	pool_mode: str = 'avg',
	aligned: bool = True,
) -> torch.Tensor:
	pooled_h, pooled_w = _pair(output_size)
	assert input.dim() == 4, 'input must have shape [N, C, H, W]'
	assert rois.dim() == 2 and rois.size(1) == 5, 'rois must have shape [K, 5]'
	assert pooled_h > 0 and pooled_w > 0, 'output_size must be positive'
	assert sampling_ratio >= 0, 'sampling_ratio must be non-negative'
	assert pool_mode in _POOL_MODE_TO_ID, f'pool_mode must be one of {tuple(_POOL_MODE_TO_ID)}'
	assert input.dtype in (torch.float16, torch.bfloat16, torch.float32), (
		'input must be float16, bfloat16, or float32'
	)
	assert rois.dtype in (torch.float16, torch.bfloat16, torch.float32), (
		'rois must be float16, bfloat16, or float32'
	)

	num_rois = rois.size(0)
	channels = input.size(1)
	height = input.size(2)
	width = input.size(3)
	output = input.new_empty((num_rois, channels, pooled_h, pooled_w))

	for n in range(num_rois):
		batch_idx = int(rois[n, 0].item())
		feature_map = input[batch_idx]
		feature_map_flat = feature_map.reshape(channels, -1)

		offset = 0.5 if aligned else 0.0
		roi_start_w = rois[n, 1] * spatial_scale - offset
		roi_start_h = rois[n, 2] * spatial_scale - offset
		roi_end_w = rois[n, 3] * spatial_scale - offset
		roi_end_h = rois[n, 4] * spatial_scale - offset

		roi_width = roi_end_w - roi_start_w
		roi_height = roi_end_h - roi_start_h
		if aligned:
			assert roi_width >= 0 and roi_height >= 0, 'ROIs in ROIAlign cannot have negative size'
		else:
			roi_width = torch.clamp_min(roi_width, 1.0)
			roi_height = torch.clamp_min(roi_height, 1.0)

		bin_size_h = roi_height / pooled_h
		bin_size_w = roi_width / pooled_w
		if sampling_ratio > 0:
			roi_bin_grid_h = sampling_ratio
			roi_bin_grid_w = sampling_ratio
		else:
			roi_bin_grid_h = int(torch.ceil(roi_height / pooled_h).item())
			roi_bin_grid_w = int(torch.ceil(roi_width / pooled_w).item())

		count = max(roi_bin_grid_h * roi_bin_grid_w, 1)
		pre_calc = _pre_calc_for_bilinear_interpolate(
			height,
			width,
			pooled_h,
			pooled_w,
			roi_start_h,
			roi_start_w,
			bin_size_h,
			bin_size_w,
			roi_bin_grid_h,
			roi_bin_grid_w,
			feature_map.dtype,
			feature_map.device,
		)

		pre_calc_index = 0
		for ph in range(pooled_h):
			for pw in range(pooled_w):
				if pool_mode == 'avg':
					pooled_val = feature_map.new_zeros(channels)
				else:
					pooled_val = feature_map.new_full((channels,), float('-inf'))

				for _iy in range(roi_bin_grid_h):
					for _ix in range(roi_bin_grid_w):
						pos1, pos2, pos3, pos4, w1, w2, w3, w4 = pre_calc[pre_calc_index]
						sample = (
							w1 * feature_map_flat[:, pos1]
							+ w2 * feature_map_flat[:, pos2]
							+ w3 * feature_map_flat[:, pos3]
							+ w4 * feature_map_flat[:, pos4]
						)
						if pool_mode == 'avg':
							pooled_val = pooled_val + sample
						else:
							pooled_val = torch.maximum(pooled_val, sample)
						pre_calc_index += 1

				if pool_mode == 'avg':
					output[n, :, ph, pw] = pooled_val / count
				else:
					output[n, :, ph, pw] = pooled_val

	return output


@triton.jit
def _roi_align_forward(
	input_ptr,
	batch_indices_ptr,
	roi_start_h_ptr,
	roi_start_w_ptr,
	bin_size_h_ptr,
	bin_size_w_ptr,
	grid_h_ptr,
	grid_w_ptr,
	output_ptr,
	argmax_y_ptr,
	argmax_x_ptr,
	pooled_h,
	pooled_w,
	channels,
	height,
	width,
	numel,
	pool_mode: tl.constexpr,
	MAX_GRID_H: tl.constexpr,
	MAX_GRID_W: tl.constexpr,
	BLOCK_SIZE: tl.constexpr,
):
	pid = tl.program_id(axis=0)
	offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
	mask = offsets < numel

	pw = offsets % pooled_w
	ph = (offsets // pooled_w) % pooled_h
	c = (offsets // (pooled_w * pooled_h)) % channels
	n = offsets // (pooled_w * pooled_h * channels)

	batch_idx = tl.load(batch_indices_ptr + n, mask=mask, other=0)
	roi_start_h = tl.load(roi_start_h_ptr + n, mask=mask, other=0.0)
	roi_start_w = tl.load(roi_start_w_ptr + n, mask=mask, other=0.0)
	bin_size_h = tl.load(bin_size_h_ptr + n, mask=mask, other=0.0)
	bin_size_w = tl.load(bin_size_w_ptr + n, mask=mask, other=0.0)
	grid_h = tl.load(grid_h_ptr + n, mask=mask, other=0)
	grid_w = tl.load(grid_w_ptr + n, mask=mask, other=0)
	one = tl.full([BLOCK_SIZE], 1, dtype=roi_start_h.dtype)
	half = tl.full([BLOCK_SIZE], 0.5, dtype=roi_start_h.dtype)
	grid_h_f = tl.maximum(grid_h.to(roi_start_h.dtype), one)
	grid_w_f = tl.maximum(grid_w.to(roi_start_h.dtype), one)

	plane_ptr = input_ptr + ((batch_idx * channels + c) * height * width)

	if pool_mode == 0:
		pooled = tl.full([BLOCK_SIZE], float('-inf'), dtype=roi_start_h.dtype)
		argmax_y = tl.full([BLOCK_SIZE], -1.0, dtype=roi_start_h.dtype)
		argmax_x = tl.full([BLOCK_SIZE], -1.0, dtype=roi_start_h.dtype)
	else:
		pooled = tl.zeros([BLOCK_SIZE], dtype=roi_start_h.dtype)

	for iy in range(MAX_GRID_H):
		iy_mask = mask & (iy < grid_h)
		iy_half = tl.full([BLOCK_SIZE], iy, dtype=roi_start_h.dtype) + half
		sample_y = (
			roi_start_h + ph.to(roi_start_h.dtype) * bin_size_h + iy_half * bin_size_h / grid_h_f
		).to(roi_start_h.dtype)
		for ix in range(MAX_GRID_W):
			sample_mask = iy_mask & (ix < grid_w)
			ix_half = tl.full([BLOCK_SIZE], ix, dtype=roi_start_h.dtype) + half
			sample_x = (
				roi_start_w
				+ pw.to(roi_start_h.dtype) * bin_size_w
				+ ix_half * bin_size_w / grid_w_f
			).to(roi_start_h.dtype)
			val = bilinear_interpolate_kernel(plane_ptr, sample_y, sample_x, height, width).to(
				roi_start_h.dtype
			)
			val = tl.where(sample_mask, val, 0.0)

			if pool_mode == 0:
				better = sample_mask & (val > pooled)
				pooled = tl.where(better, val, pooled)
				argmax_y = tl.where(better, sample_y, argmax_y)
				argmax_x = tl.where(better, sample_x, argmax_x)
			else:
				pooled += val

	if pool_mode == 1:
		count = tl.maximum((grid_h * grid_w).to(roi_start_h.dtype), one)
		pooled = pooled / count
		tl.store(output_ptr + offsets, pooled, mask=mask)
	else:
		tl.store(output_ptr + offsets, pooled, mask=mask)
		tl.store(argmax_y_ptr + offsets, argmax_y, mask=mask)
		tl.store(argmax_x_ptr + offsets, argmax_x, mask=mask)


@triton.jit
def _roi_align_backward(
	grad_output_ptr,
	batch_indices_ptr,
	roi_start_h_ptr,
	roi_start_w_ptr,
	bin_size_h_ptr,
	bin_size_w_ptr,
	grid_h_ptr,
	grid_w_ptr,
	argmax_y_ptr,
	argmax_x_ptr,
	grad_input_ptr,
	pooled_h,
	pooled_w,
	channels,
	height,
	width,
	numel,
	pool_mode: tl.constexpr,
	MAX_GRID_H: tl.constexpr,
	MAX_GRID_W: tl.constexpr,
	BLOCK_SIZE: tl.constexpr,
):
	pid = tl.program_id(axis=0)
	offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
	mask = offsets < numel

	pw = offsets % pooled_w
	ph = (offsets // pooled_w) % pooled_h
	c = (offsets // (pooled_w * pooled_h)) % channels
	n = offsets // (pooled_w * pooled_h * channels)

	batch_idx = tl.load(batch_indices_ptr + n, mask=mask, other=0)
	roi_start_h = tl.load(roi_start_h_ptr + n, mask=mask, other=0.0)
	roi_start_w = tl.load(roi_start_w_ptr + n, mask=mask, other=0.0)
	bin_size_h = tl.load(bin_size_h_ptr + n, mask=mask, other=0.0)
	bin_size_w = tl.load(bin_size_w_ptr + n, mask=mask, other=0.0)
	grid_h = tl.load(grid_h_ptr + n, mask=mask, other=0)
	grid_w = tl.load(grid_w_ptr + n, mask=mask, other=0)
	grad = tl.load(grad_output_ptr + offsets, mask=mask, other=0.0)
	one = tl.full([BLOCK_SIZE], 1, dtype=roi_start_h.dtype)
	half = tl.full([BLOCK_SIZE], 0.5, dtype=roi_start_h.dtype)
	neg_one = tl.full([BLOCK_SIZE], -1, dtype=roi_start_h.dtype)
	grid_h_f = tl.maximum(grid_h.to(roi_start_h.dtype), one)
	grid_w_f = tl.maximum(grid_w.to(roi_start_h.dtype), one)
	plane_ptr = grad_input_ptr + ((batch_idx * channels + c) * height * width)

	if pool_mode == 0:
		y = tl.load(argmax_y_ptr + offsets, mask=mask, other=-1.0)
		x = tl.load(argmax_x_ptr + offsets, mask=mask, other=-1.0)
		w1, w2, w3, w4, y_low, x_low, y_high, x_high = bilinear_interpolate_gradient_kernel(
			y, x, height, width
		)

		valid = mask & (y != neg_one)
		tl.atomic_add(
			plane_ptr + y_low * width + x_low, grad * w1, mask=valid & (x_low >= 0) & (y_low >= 0)
		)
		tl.atomic_add(
			plane_ptr + y_low * width + x_high, grad * w2, mask=valid & (x_high >= 0) & (y_low >= 0)
		)
		tl.atomic_add(
			plane_ptr + y_high * width + x_low, grad * w3, mask=valid & (x_low >= 0) & (y_high >= 0)
		)
		tl.atomic_add(
			plane_ptr + y_high * width + x_high,
			grad * w4,
			mask=valid & (x_high >= 0) & (y_high >= 0),
		)
	else:
		count = tl.maximum((grid_h * grid_w).to(roi_start_h.dtype), one)
		grad = grad / count

		for iy in range(MAX_GRID_H):
			iy_mask = mask & (iy < grid_h)
			iy_half = tl.full([BLOCK_SIZE], iy, dtype=roi_start_h.dtype) + half
			sample_y = (
				roi_start_h
				+ ph.to(roi_start_h.dtype) * bin_size_h
				+ iy_half * bin_size_h / grid_h_f
			)
			for ix in range(MAX_GRID_W):
				sample_mask = iy_mask & (ix < grid_w)
				ix_half = tl.full([BLOCK_SIZE], ix, dtype=roi_start_h.dtype) + half
				sample_x = (
					roi_start_w
					+ pw.to(roi_start_h.dtype) * bin_size_w
					+ ix_half * bin_size_w / grid_w_f
				)
				w1, w2, w3, w4, y_low, x_low, y_high, x_high = bilinear_interpolate_gradient_kernel(
					sample_y, sample_x, height, width
				)

				contrib1 = grad * w1
				contrib2 = grad * w2
				contrib3 = grad * w3
				contrib4 = grad * w4

				tl.atomic_add(
					plane_ptr + y_low * width + x_low,
					contrib1,
					mask=sample_mask & (x_low >= 0) & (y_low >= 0),
				)
				tl.atomic_add(
					plane_ptr + y_low * width + x_high,
					contrib2,
					mask=sample_mask & (x_high >= 0) & (y_low >= 0),
				)
				tl.atomic_add(
					plane_ptr + y_high * width + x_low,
					contrib3,
					mask=sample_mask & (x_low >= 0) & (y_high >= 0),
				)
				tl.atomic_add(
					plane_ptr + y_high * width + x_high,
					contrib4,
					mask=sample_mask & (x_high >= 0) & (y_high >= 0),
				)


class RoIAlignFunction(Function):
	@staticmethod
	def forward(
		ctx,
		input: torch.Tensor,
		rois: torch.Tensor,
		output_size: tuple[int, int],
		spatial_scale: float = 1.0,
		sampling_ratio: int = 0,
		pool_mode: str = 'avg',
		aligned: bool = True,
	) -> torch.Tensor:
		output_size = _pair(output_size)
		assert input.dim() == 4, 'input must have shape [N, C, H, W]'
		assert rois.dim() == 2 and rois.size(1) == 5, 'rois must have shape [K, 5]'
		assert output_size[0] > 0 and output_size[1] > 0, 'output_size must be positive'
		assert sampling_ratio >= 0, 'sampling_ratio must be non-negative'
		assert pool_mode in _POOL_MODE_TO_ID, f'pool_mode must be one of {tuple(_POOL_MODE_TO_ID)}'
		assert input.dtype in (torch.float16, torch.bfloat16, torch.float32), (
			'input must be float16, bfloat16, or float32'
		)
		assert rois.dtype in (torch.float16, torch.bfloat16, torch.float32), (
			'rois must be float16, bfloat16, or float32'
		)

		rois = rois.to(dtype=input.dtype).contiguous()

		if not input.is_cuda:
			assert not rois.is_cuda, 'rois must be CPU tensor when input is CPU'
			ctx.output_size = output_size
			ctx.pool_mode = pool_mode
			ctx.input_shape = tuple(input.shape)
			ctx.spatial_scale = float(spatial_scale)
			ctx.sampling_ratio = int(sampling_ratio)
			ctx.aligned = bool(aligned)
			ctx.is_cuda = False
			ctx.save_for_backward(input, rois)
			return _roi_align_cpu_fallback(
				input,
				rois,
				output_size,
				spatial_scale,
				sampling_ratio,
				pool_mode,
				aligned,
			)

		assert rois.is_cuda, 'rois must be CUDA tensor when input is CUDA'
		ctx.is_cuda = True

		input = input.contiguous()

		offset = 0.5 if aligned else 0.0
		batch_indices = rois[:, 0].to(torch.int32)
		roi_start_w = rois[:, 1] * spatial_scale - offset
		roi_start_h = rois[:, 2] * spatial_scale - offset
		roi_end_w = rois[:, 3] * spatial_scale - offset
		roi_end_h = rois[:, 4] * spatial_scale - offset

		roi_width = roi_end_w - roi_start_w
		roi_height = roi_end_h - roi_start_h
		if aligned:
			assert bool(torch.all(roi_width >= 0)), 'ROIs in ROIAlign cannot have negative width'
			assert bool(torch.all(roi_height >= 0)), 'ROIs in ROIAlign cannot have negative height'
		else:
			roi_width = torch.clamp_min(roi_width, 1.0)
			roi_height = torch.clamp_min(roi_height, 1.0)

		bin_size_h = roi_height / output_size[0]
		bin_size_w = roi_width / output_size[1]
		if sampling_ratio > 0:
			grid_h = torch.full_like(batch_indices, sampling_ratio, dtype=torch.int32)
			grid_w = torch.full_like(batch_indices, sampling_ratio, dtype=torch.int32)
		else:
			grid_h = torch.ceil(roi_height / output_size[0]).to(torch.int32)
			grid_w = torch.ceil(roi_width / output_size[1]).to(torch.int32)

		num_rois = rois.size(0)
		channels = input.size(1)
		pooled_h, pooled_w = output_size
		output = torch.empty(
			(num_rois, channels, pooled_h, pooled_w), device=input.device, dtype=input.dtype
		)

		if pool_mode == 'max':
			argmax_y = torch.empty_like(output)
			argmax_x = torch.empty_like(output)
		else:
			argmax_y = input.new_empty(0)
			argmax_x = input.new_empty(0)

		numel = output.numel()
		max_grid_h = int(grid_h.max().item()) if grid_h.numel() > 0 else 1
		max_grid_w = int(grid_w.max().item()) if grid_w.numel() > 0 else 1

		if numel > 0:
			grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
			_roi_align_forward[grid](
				input,
				batch_indices,
				roi_start_h,
				roi_start_w,
				bin_size_h,
				bin_size_w,
				grid_h,
				grid_w,
				output,
				argmax_y,
				argmax_x,
				pooled_h,
				pooled_w,
				channels,
				input.size(2),
				input.size(3),
				numel,
				pool_mode=_POOL_MODE_TO_ID[pool_mode],
				MAX_GRID_H=max(max_grid_h, 1),
				MAX_GRID_W=max(max_grid_w, 1),
				BLOCK_SIZE=_BLOCK_SIZE,
			)

		ctx.output_size = output_size
		ctx.pool_mode = pool_mode
		ctx.input_shape = tuple(input.shape)
		ctx.max_grid_h = max(max_grid_h, 1)
		ctx.max_grid_w = max(max_grid_w, 1)
		ctx.save_for_backward(
			batch_indices,
			roi_start_h,
			roi_start_w,
			bin_size_h,
			bin_size_w,
			grid_h,
			grid_w,
			argmax_y,
			argmax_x,
		)
		return output

	@staticmethod
	@once_differentiable
	def backward(ctx, grad_output: torch.Tensor) -> tuple:
		if not ctx.is_cuda:
			input, rois = ctx.saved_tensors
			with torch.enable_grad():
				replay_input = input.detach().requires_grad_(True)
				replay_out = _roi_align_cpu_fallback(
					replay_input,
					rois,
					ctx.output_size,
					ctx.spatial_scale,
					ctx.sampling_ratio,
					ctx.pool_mode,
					ctx.aligned,
				)
			grad_input = torch.autograd.grad(
				replay_out,
				replay_input,
				grad_output,
				retain_graph=False,
				create_graph=False,
				allow_unused=False,
			)[0]
			return grad_input, None, None, None, None, None, None

		(
			batch_indices,
			roi_start_h,
			roi_start_w,
			bin_size_h,
			bin_size_w,
			grid_h,
			grid_w,
			argmax_y,
			argmax_x,
		) = ctx.saved_tensors
		grad_output = grad_output.contiguous()
		grad_input = grad_output.new_zeros(ctx.input_shape)

		numel = grad_output.numel()
		if numel > 0:
			pooled_h, pooled_w = ctx.output_size
			channels = grad_input.size(1)
			grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)

			if ctx.pool_mode == 'avg':
				pool_mode = _POOL_MODE_TO_ID['avg']
			else:
				pool_mode = _POOL_MODE_TO_ID['max']
			_roi_align_backward[grid](
				grad_output,
				batch_indices,
				roi_start_h,
				roi_start_w,
				bin_size_h,
				bin_size_w,
				grid_h,
				grid_w,
				argmax_y,
				argmax_x,
				grad_input,
				pooled_h,
				pooled_w,
				channels,
				grad_input.size(2),
				grad_input.size(3),
				numel,
				pool_mode=pool_mode,
				MAX_GRID_H=ctx.max_grid_h,
				MAX_GRID_W=ctx.max_grid_w,
				BLOCK_SIZE=_BLOCK_SIZE,
			)

		return grad_input, None, None, None, None, None, None


roi_align = RoIAlignFunction.apply


class RoIAlign(nn.Module):
	def __init__(
		self,
		output_size,
		spatial_scale: float = 1.0,
		sampling_ratio: int = 0,
		pool_mode: str = 'avg',
		aligned: bool = True,
	):
		super().__init__()
		self.output_size = _pair(output_size)
		self.spatial_scale = float(spatial_scale)
		self.sampling_ratio = int(sampling_ratio)
		self.pool_mode = pool_mode
		self.aligned = bool(aligned)

	def forward(self, input: torch.Tensor, rois: torch.Tensor) -> torch.Tensor:
		return roi_align(
			input,
			rois,
			self.output_size,
			self.spatial_scale,
			self.sampling_ratio,
			self.pool_mode,
			self.aligned,
		)

	def extra_repr(self) -> str:
		return (
			f'output_size={self.output_size}, spatial_scale={self.spatial_scale}, '
			f'sampling_ratio={self.sampling_ratio}, pool_mode={self.pool_mode}, aligned={self.aligned}'
		)


__all__ = ['RoIAlign', 'roi_align']
