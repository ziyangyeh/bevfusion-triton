from __future__ import annotations

import math

import torch
import triton
import triton.language as tl
from torch import nn
from torch.autograd import Function


_BLOCK_SIZE = 256


@triton.jit
def _build_voxel_index_tensors_kernel(
	rois_ptr,
	pts_ptr,
	pts_mask_ptr,
	local_xyz_ptr,
	num_rois,
	num_pts,
	out_x,
	out_y,
	out_z,
):
	pid = tl.program_id(axis=0)
	roi_idx = pid // num_pts
	pt_idx = pid % num_pts
	if roi_idx >= num_rois:
		return

	roi_base = roi_idx * 7
	pt_base = pt_idx * 3
	cx = tl.load(rois_ptr + roi_base + 0)
	cy = tl.load(rois_ptr + roi_base + 1)
	cz_bottom = tl.load(rois_ptr + roi_base + 2)
	w = tl.load(rois_ptr + roi_base + 3)
	l = tl.load(rois_ptr + roi_base + 4)
	h = tl.load(rois_ptr + roi_base + 5)
	rz = tl.load(rois_ptr + roi_base + 6)

	px = tl.load(pts_ptr + pt_base + 0)
	py = tl.load(pts_ptr + pt_base + 1)
	pz = tl.load(pts_ptr + pt_base + 2)

	cz = cz_bottom + h * 0.5
	dx = px - cx
	dy = py - cy
	rot_angle = rz + 1.5707963267948966
	cosa = tl.cos(rot_angle)
	sina = tl.sin(rot_angle)
	local_x = dx * cosa + dy * (-sina)
	local_y = dx * sina + dy * cosa
	local_z = pz - cz_bottom

	local_base = (roi_idx * num_pts + pt_idx) * 3
	tl.store(local_xyz_ptr + local_base + 0, local_x)
	tl.store(local_xyz_ptr + local_base + 1, local_y)
	tl.store(local_xyz_ptr + local_base + 2, local_z)

	in_z = tl.abs(pz - cz) <= h * 0.5
	in_xy = (local_x > -l * 0.5) & (local_x < l * 0.5) & (local_y > -w * 0.5) & (local_y < w * 0.5)
	inside = in_z & in_xy

	mask_val = tl.full((), -1, tl.int32)
	if inside:
		x_res = l / out_x
		y_res = w / out_y
		z_res = h / out_z
		x_idx = tl.minimum(tl.maximum(((local_x + l * 0.5) / x_res).to(tl.int32), 0), out_x - 1)
		y_idx = tl.minimum(tl.maximum(((local_y + w * 0.5) / y_res).to(tl.int32), 0), out_y - 1)
		z_idx = tl.minimum(tl.maximum((local_z / z_res).to(tl.int32), 0), out_z - 1)
		mask_val = (x_idx << 16) + (y_idx << 8) + z_idx

	tl.store(pts_mask_ptr + roi_idx * num_pts + pt_idx, mask_val)


@triton.jit
def _collect_points_per_voxel_kernel(
	pts_mask_ptr,
	pts_idx_of_voxels_ptr,
	num_rois,
	num_pts,
	out_x,
	out_y,
	out_z,
	max_pts_per_voxel,
):
	pid = tl.program_id(axis=0)
	roi_idx = pid // num_pts
	pt_idx = pid % num_pts
	if roi_idx >= num_rois:
		return

	encoded = tl.load(pts_mask_ptr + roi_idx * num_pts + pt_idx).to(tl.int32)
	if encoded == -1:
		return

	x_idx = (encoded >> 16) & 0xFF
	y_idx = (encoded >> 8) & 0xFF
	z_idx = encoded & 0xFF
	voxel_base = (((roi_idx * out_x + x_idx) * out_y + y_idx) * out_z + z_idx) * max_pts_per_voxel
	count_ptr = pts_idx_of_voxels_ptr + voxel_base
	old_count = tl.atomic_add(count_ptr, 1)
	max_num_pts = max_pts_per_voxel - 1
	if old_count < max_num_pts:
		tl.store(pts_idx_of_voxels_ptr + voxel_base + old_count + 1, pt_idx)


@triton.jit
def _clamp_voxel_point_count_kernel(
	pts_idx_of_voxels_ptr,
	num_voxels,
	max_pts_per_voxel,
):
	pid = tl.program_id(axis=0)
	if pid >= num_voxels:
		return
	base = pid * max_pts_per_voxel
	count = tl.load(pts_idx_of_voxels_ptr + base).to(tl.int32)
	max_num_pts = max_pts_per_voxel - 1
	clamped = tl.minimum(count, max_num_pts)
	tl.store(pts_idx_of_voxels_ptr + base, clamped)


@triton.jit
def _pool_features_kernel(
	pts_feature_ptr,
	pts_idx_of_voxels_ptr,
	pooled_ptr,
	argmax_ptr,
	num_voxels,
	num_channels,
	max_pts_per_voxel,
	mode: tl.constexpr,
):
	pid = tl.program_id(axis=0)
	total = num_voxels * num_channels
	if pid >= total:
		return
	voxel_idx = pid // num_channels
	channel_idx = pid % num_channels
	voxel_base = voxel_idx * max_pts_per_voxel
	count = tl.load(pts_idx_of_voxels_ptr + voxel_base).to(tl.int32)
	if count <= 0:
		tl.store(pooled_ptr + pid, 0.0)
		if mode == 0:
			tl.store(argmax_ptr + pid, -1)
		return
	if mode == 0:
		best_val = tl.full((), -float('inf'), tl.float32)
		best_idx = tl.full((), -1, tl.int32)
		i = 0
		while i < max_pts_per_voxel - 1:
			if i < count:
				pt_idx = tl.load(pts_idx_of_voxels_ptr + voxel_base + i + 1).to(tl.int32)
				val = tl.load(pts_feature_ptr + pt_idx * num_channels + channel_idx)
				update = val > best_val
				best_val = tl.where(update, val, best_val)
				best_idx = tl.where(update, pt_idx, best_idx)
			i += 1
		tl.store(pooled_ptr + pid, best_val)
		tl.store(argmax_ptr + pid, best_idx)
	else:
		acc = tl.full((), 0.0, tl.float32)
		i = 0
		while i < max_pts_per_voxel - 1:
			if i < count:
				pt_idx = tl.load(pts_idx_of_voxels_ptr + voxel_base + i + 1).to(tl.int32)
				val = tl.load(pts_feature_ptr + pt_idx * num_channels + channel_idx)
				acc += val
			i += 1
		tl.store(pooled_ptr + pid, acc / count.to(tl.float32))


@triton.jit
def _max_pool_backward_kernel(
	grad_out_ptr,
	argmax_ptr,
	grad_in_ptr,
	total,
	num_channels,
):
	pid = tl.program_id(axis=0)
	if pid >= total:
		return
	pt_idx = tl.load(argmax_ptr + pid).to(tl.int32)
	if pt_idx < 0:
		return
	channel_idx = pid % num_channels
	grad = tl.load(grad_out_ptr + pid)
	tl.atomic_add(grad_in_ptr + pt_idx * num_channels + channel_idx, grad)


@triton.jit
def _avg_pool_backward_kernel(
	grad_out_ptr,
	pts_idx_of_voxels_ptr,
	grad_in_ptr,
	num_voxels,
	num_channels,
	max_pts_per_voxel,
):
	pid = tl.program_id(axis=0)
	total = num_voxels * num_channels
	if pid >= total:
		return
	voxel_idx = pid // num_channels
	channel_idx = pid % num_channels
	voxel_base = voxel_idx * max_pts_per_voxel
	count = tl.load(pts_idx_of_voxels_ptr + voxel_base).to(tl.int32)
	if count <= 0:
		return
	grad_share = tl.load(grad_out_ptr + pid) / count.to(tl.float32)
	i = 0
	while i < max_pts_per_voxel - 1:
		if i < count:
			pt_idx = tl.load(pts_idx_of_voxels_ptr + voxel_base + i + 1).to(tl.int32)
			tl.atomic_add(grad_in_ptr + pt_idx * num_channels + channel_idx, grad_share)
		i += 1


def _pair_out_size(out_size) -> tuple[int, int, int]:
	if isinstance(out_size, int):
		return out_size, out_size, out_size
	if len(out_size) != 3:
		raise ValueError('out_size must be an int or a tuple of length 3.')
	if not all(isinstance(v, int) for v in out_size):
		raise TypeError('out_size must contain integers.')
	return int(out_size[0]), int(out_size[1]), int(out_size[2])


def _check_forward_inputs(rois: torch.Tensor, pts: torch.Tensor, pts_feature: torch.Tensor) -> None:
	if rois.dim() != 2 or rois.size(-1) != 7:
		raise ValueError('rois must have shape [N, 7].')
	if pts.dim() != 2 or pts.size(-1) != 3:
		raise ValueError('pts must have shape [npoints, 3].')
	if pts_feature.dim() != 2:
		raise ValueError('pts_feature must have shape [npoints, C].')
	if pts.size(0) != pts_feature.size(0):
		raise ValueError('pts and pts_feature must have the same number of points.')
	if (
		rois.dtype != torch.float32
		or pts.dtype != torch.float32
		or pts_feature.dtype != torch.float32
	):
		raise TypeError('rois, pts, and pts_feature must have dtype torch.float32.')
	if rois.device != pts.device or pts.device != pts_feature.device:
		raise ValueError('rois, pts, and pts_feature must be on the same device.')


def _points_in_single_roi(
	pts: torch.Tensor,
	roi: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
	cx, cy, cz_bottom, w, l, h, rz = roi
	cz = cz_bottom + h * 0.5
	dx = pts[:, 0] - cx
	dy = pts[:, 1] - cy
	rot_angle = rz + math.pi * 0.5
	cosa = math.cos(float(rot_angle))
	sina = math.sin(float(rot_angle))
	local_x = dx * cosa + dy * (-sina)
	local_y = dx * sina + dy * cosa
	mask = (
		((pts[:, 2] - cz).abs() <= h * 0.5)
		& (local_x > -l * 0.5)
		& (local_x < l * 0.5)
		& (local_y > -w * 0.5)
		& (local_y < w * 0.5)
	)
	local_z = pts[:, 2] - cz_bottom
	return mask, local_x, local_y, local_z


def _build_voxel_index_tensors_cpu(
	rois: torch.Tensor,
	pts: torch.Tensor,
	out_size: tuple[int, int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
	out_x, out_y, out_z = out_size
	pts_mask = torch.full((rois.size(0), pts.size(0)), -1, device=pts.device, dtype=torch.int32)
	local_xyz = torch.zeros((rois.size(0), pts.size(0), 3), device=pts.device, dtype=torch.float32)

	for roi_idx in range(rois.size(0)):
		mask, local_x, local_y, local_z = _points_in_single_roi(pts, rois[roi_idx])
		if not mask.any():
			continue
		l = rois[roi_idx, 4]
		w = rois[roi_idx, 3]
		h = rois[roi_idx, 5]
		x_res = l / out_x
		y_res = w / out_y
		z_res = h / out_z
		x_idx = ((local_x + l * 0.5) / x_res).to(torch.int64).clamp_(0, out_x - 1)
		y_idx = ((local_y + w * 0.5) / y_res).to(torch.int64).clamp_(0, out_y - 1)
		z_idx = (local_z / z_res).to(torch.int64).clamp_(0, out_z - 1)
		encoded = ((x_idx << 16) + (y_idx << 8) + z_idx).to(torch.int32)
		pts_mask[roi_idx, mask] = encoded[mask]
		local_xyz[roi_idx, :, 0] = local_x
		local_xyz[roi_idx, :, 1] = local_y
		local_xyz[roi_idx, :, 2] = local_z
	return pts_mask, local_xyz


def _build_voxel_index_tensors_triton(
	rois: torch.Tensor,
	pts: torch.Tensor,
	out_size: tuple[int, int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
	out_x, out_y, out_z = out_size
	pts_mask = torch.full((rois.size(0), pts.size(0)), -1, device=pts.device, dtype=torch.int32)
	local_xyz = torch.empty((rois.size(0), pts.size(0), 3), device=pts.device, dtype=torch.float32)
	total = int(rois.size(0) * pts.size(0))
	if total > 0:
		_build_voxel_index_tensors_kernel[(total,)](
			rois.contiguous(),
			pts.contiguous(),
			pts_mask,
			local_xyz,
			int(rois.size(0)),
			int(pts.size(0)),
			out_x,
			out_y,
			out_z,
		)
	return pts_mask, local_xyz


def _build_voxel_index_tensors(
	rois: torch.Tensor,
	pts: torch.Tensor,
	out_size: tuple[int, int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
	if rois.is_cuda:
		return _build_voxel_index_tensors_triton(rois, pts, out_size)
	return _build_voxel_index_tensors_cpu(rois, pts, out_size)


def _collect_points_per_voxel_cpu(
	pts_mask: torch.Tensor,
	out_size: tuple[int, int, int],
	max_pts_per_voxel: int,
) -> torch.Tensor:
	num_rois, pts_num = pts_mask.shape
	out_x, out_y, out_z = out_size
	pts_idx_of_voxels = torch.zeros(
		(num_rois, out_x, out_y, out_z, max_pts_per_voxel),
		device=pts_mask.device,
		dtype=torch.int32,
	)
	max_num_pts = max_pts_per_voxel - 1
	for roi_idx in range(num_rois):
		for pt_idx in range(pts_num):
			encoded = int(pts_mask[roi_idx, pt_idx].item())
			if encoded == -1:
				continue
			x_idx = (encoded >> 16) & 0xFF
			y_idx = (encoded >> 8) & 0xFF
			z_idx = encoded & 0xFF
			count = int(pts_idx_of_voxels[roi_idx, x_idx, y_idx, z_idx, 0].item())
			if count < max_num_pts:
				pts_idx_of_voxels[roi_idx, x_idx, y_idx, z_idx, count + 1] = pt_idx
				pts_idx_of_voxels[roi_idx, x_idx, y_idx, z_idx, 0] += 1
	return pts_idx_of_voxels


def _collect_points_per_voxel_triton(
	pts_mask: torch.Tensor,
	out_size: tuple[int, int, int],
	max_pts_per_voxel: int,
) -> torch.Tensor:
	num_rois, pts_num = pts_mask.shape
	out_x, out_y, out_z = out_size
	pts_idx_of_voxels = torch.zeros(
		(num_rois, out_x, out_y, out_z, max_pts_per_voxel),
		device=pts_mask.device,
		dtype=torch.int32,
	)
	total = int(num_rois * pts_num)
	if total > 0:
		_collect_points_per_voxel_kernel[(total,)](
			pts_mask.contiguous(),
			pts_idx_of_voxels,
			num_rois,
			pts_num,
			out_x,
			out_y,
			out_z,
			max_pts_per_voxel,
		)
		num_voxels = int(num_rois * out_x * out_y * out_z)
		_clamp_voxel_point_count_kernel[(num_voxels,)](
			pts_idx_of_voxels,
			num_voxels,
			max_pts_per_voxel,
		)
	return pts_idx_of_voxels


def _collect_points_per_voxel(
	pts_mask: torch.Tensor,
	out_size: tuple[int, int, int],
	max_pts_per_voxel: int,
) -> torch.Tensor:
	if pts_mask.is_cuda:
		return _collect_points_per_voxel_triton(pts_mask, out_size, max_pts_per_voxel)
	return _collect_points_per_voxel_cpu(pts_mask, out_size, max_pts_per_voxel)


def _pool_features(
	pts_feature: torch.Tensor,
	pts_idx_of_voxels: torch.Tensor,
	mode: int,
) -> tuple[torch.Tensor, torch.Tensor]:
	if pts_feature.is_cuda:
		num_rois, out_x, out_y, out_z, max_pts_per_voxel = pts_idx_of_voxels.shape
		channels = pts_feature.size(1)
		pooled = pts_feature.new_zeros((num_rois, out_x, out_y, out_z, channels))
		argmax = torch.full(
			(num_rois, out_x, out_y, out_z, channels),
			-1,
			device=pts_feature.device,
			dtype=torch.int32,
		)
		num_voxels = int(num_rois * out_x * out_y * out_z)
		total = int(num_voxels * channels)
		if total > 0:
			_pool_features_kernel[(total,)](
				pts_feature.contiguous(),
				pts_idx_of_voxels.contiguous(),
				pooled,
				argmax,
				num_voxels,
				channels,
				max_pts_per_voxel,
				mode=mode,
			)
		return pooled, argmax

	num_rois, out_x, out_y, out_z, max_pts_per_voxel = pts_idx_of_voxels.shape
	channels = pts_feature.size(1)
	pooled = pts_feature.new_zeros((num_rois, out_x, out_y, out_z, channels))
	argmax = torch.full(
		(num_rois, out_x, out_y, out_z, channels),
		-1,
		device=pts_feature.device,
		dtype=torch.int32,
	)

	for roi_idx in range(num_rois):
		for x_idx in range(out_x):
			for y_idx in range(out_y):
				for z_idx in range(out_z):
					total_pts = int(pts_idx_of_voxels[roi_idx, x_idx, y_idx, z_idx, 0].item())
					if total_pts <= 0:
						continue
					pts_idx = pts_idx_of_voxels[roi_idx, x_idx, y_idx, z_idx, 1 : total_pts + 1].to(
						torch.long
					)
					voxel_feats = pts_feature[pts_idx]
					if mode == 0:
						values, indices = voxel_feats.max(dim=0)
						pooled[roi_idx, x_idx, y_idx, z_idx] = values
						argmax[roi_idx, x_idx, y_idx, z_idx] = pts_idx[indices].to(torch.int32)
					else:
						pooled[roi_idx, x_idx, y_idx, z_idx] = voxel_feats.mean(dim=0)
	return pooled, argmax


def _pool_features_backward_cpu(
	grad_out: torch.Tensor,
	pts_idx_of_voxels: torch.Tensor,
	argmax: torch.Tensor,
	mode: int,
	num_pts: int,
	num_channels: int,
) -> torch.Tensor:
	grad_in = grad_out.new_zeros((num_pts, num_channels))
	num_rois, out_x, out_y, out_z, _ = pts_idx_of_voxels.shape

	for roi_idx in range(num_rois):
		for x_idx in range(out_x):
			for y_idx in range(out_y):
				for z_idx in range(out_z):
					if mode == 0:
						argmax_idx = argmax[roi_idx, x_idx, y_idx, z_idx]
						valid = argmax_idx >= 0
						if valid.any():
							flat_idx = argmax_idx[valid].to(torch.long)
							grad_in[flat_idx, valid] += grad_out[
								roi_idx, x_idx, y_idx, z_idx, valid
							]
					else:
						total_pts = int(pts_idx_of_voxels[roi_idx, x_idx, y_idx, z_idx, 0].item())
						if total_pts <= 0:
							continue
						pts_idx = pts_idx_of_voxels[
							roi_idx, x_idx, y_idx, z_idx, 1 : total_pts + 1
						].to(torch.long)
						grad_share = grad_out[roi_idx, x_idx, y_idx, z_idx] / total_pts
						grad_in[pts_idx] += grad_share
	return grad_in


def _pool_features_backward_triton(
	grad_out: torch.Tensor,
	pts_idx_of_voxels: torch.Tensor,
	argmax: torch.Tensor,
	mode: int,
	num_pts: int,
	num_channels: int,
) -> torch.Tensor:
	grad_in = grad_out.new_zeros((num_pts, num_channels))
	num_rois, out_x, out_y, out_z, max_pts_per_voxel = pts_idx_of_voxels.shape
	num_voxels = int(num_rois * out_x * out_y * out_z)
	total = int(num_voxels * num_channels)
	if total <= 0:
		return grad_in
	if mode == 0:
		_max_pool_backward_kernel[(total,)](
			grad_out.contiguous(),
			argmax.contiguous(),
			grad_in,
			total,
			num_channels,
		)
	else:
		_avg_pool_backward_kernel[(total,)](
			grad_out.contiguous(),
			pts_idx_of_voxels.contiguous(),
			grad_in,
			num_voxels,
			num_channels,
			max_pts_per_voxel,
		)
	return grad_in


def _pool_features_backward(
	grad_out: torch.Tensor,
	pts_idx_of_voxels: torch.Tensor,
	argmax: torch.Tensor,
	mode: int,
	num_pts: int,
	num_channels: int,
) -> torch.Tensor:
	if grad_out.is_cuda:
		return _pool_features_backward_triton(
			grad_out, pts_idx_of_voxels, argmax, mode, num_pts, num_channels
		)
	return _pool_features_backward_cpu(
		grad_out, pts_idx_of_voxels, argmax, mode, num_pts, num_channels
	)


class RoIAwarePool3dFunction(Function):
	@staticmethod
	def forward(ctx, rois, pts, pts_feature, out_size, max_pts_per_voxel, mode):
		_check_forward_inputs(rois, pts, pts_feature)
		out_size_3d = _pair_out_size(out_size)
		pts_mask, _ = _build_voxel_index_tensors(rois, pts, out_size_3d)
		pts_idx_of_voxels = _collect_points_per_voxel(pts_mask, out_size_3d, max_pts_per_voxel)
		pooled, argmax = _pool_features(pts_feature, pts_idx_of_voxels, mode)
		ctx.roiaware_pool3d_for_backward = (
			pts_idx_of_voxels,
			argmax,
			mode,
			pts.size(0),
			pts_feature.size(1),
		)
		return pooled

	@staticmethod
	def backward(ctx, grad_out):
		pts_idx_of_voxels, argmax, mode, num_pts, num_channels = ctx.roiaware_pool3d_for_backward
		grad_in = _pool_features_backward(
			grad_out, pts_idx_of_voxels, argmax, mode, num_pts, num_channels
		)
		return None, None, grad_in, None, None, None


class RoIAwarePool3d(nn.Module):
	def __init__(self, out_size, max_pts_per_voxel=128, mode='max'):
		super().__init__()
		self.out_size = out_size
		self.max_pts_per_voxel = max_pts_per_voxel
		if mode not in ['max', 'avg']:
			raise ValueError("mode must be 'max' or 'avg'")
		self.mode = 0 if mode == 'max' else 1

	def forward(self, rois, pts, pts_feature):
		return RoIAwarePool3dFunction.apply(
			rois,
			pts,
			pts_feature,
			self.out_size,
			self.max_pts_per_voxel,
			self.mode,
		)
