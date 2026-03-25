from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


def _check_points_boxes_batch(points: torch.Tensor, boxes: torch.Tensor) -> None:
	if points.dim() != 3 or points.size(-1) != 3:
		raise ValueError('points must have shape [B, M, 3].')
	if boxes.dim() != 3 or boxes.size(-1) != 7:
		raise ValueError('boxes must have shape [B, T, 7].')
	if points.size(0) != boxes.size(0):
		raise ValueError('points and boxes must have the same batch size.')
	if points.dtype != torch.float32 or boxes.dtype != torch.float32:
		raise TypeError('points and boxes must have dtype torch.float32.')


def _check_points_boxes_cpu(points: torch.Tensor, boxes: torch.Tensor) -> None:
	if points.dim() != 2 or points.size(-1) != 3:
		raise ValueError('points must have shape [M, 3].')
	if boxes.dim() != 2 or boxes.size(-1) != 7:
		raise ValueError('boxes must have shape [T, 7].')
	if points.dtype != torch.float32 or boxes.dtype != torch.float32:
		raise TypeError('points and boxes must have dtype torch.float32.')


@triton.jit
def _load_box_component(base_ptr, box_idx, stride_box, comp_idx):
	return tl.load(base_ptr + box_idx * stride_box + comp_idx)


@triton.jit
def _check_point_in_box3d(px, py, pz, boxes_ptr, box_idx, stride_box):
	cx = _load_box_component(boxes_ptr, box_idx, stride_box, 0)
	cy = _load_box_component(boxes_ptr, box_idx, stride_box, 1)
	cz_bottom = _load_box_component(boxes_ptr, box_idx, stride_box, 2)
	w = _load_box_component(boxes_ptr, box_idx, stride_box, 3)
	l = _load_box_component(boxes_ptr, box_idx, stride_box, 4)
	h = _load_box_component(boxes_ptr, box_idx, stride_box, 5)
	rz = _load_box_component(boxes_ptr, box_idx, stride_box, 6)

	cz = cz_bottom + h * 0.5
	in_z = tl.abs(pz - cz) <= h * 0.5

	shift_x = px - cx
	shift_y = py - cy
	rot_angle = rz + math.pi * 0.5
	cosa = tl.cos(rot_angle)
	sina = tl.sin(rot_angle)
	local_x = shift_x * cosa + shift_y * (-sina)
	local_y = shift_x * sina + shift_y * cosa
	in_xy = (local_x > -l * 0.5) & (local_x < l * 0.5) & (local_y > -w * 0.5) & (local_y < w * 0.5)
	return in_z & in_xy


@triton.jit
def _points_in_boxes_gpu_kernel(
	boxes_ptr,
	points_ptr,
	out_ptr,
	boxes_num,
	pts_num,
	stride_boxes_batch,
	stride_boxes_box,
	stride_points_batch,
	stride_points_point,
	stride_out_batch,
	MAX_BOXES: tl.constexpr,
):
	pid = tl.program_id(axis=0)
	bs_idx = pid // pts_num
	pt_idx = pid % pts_num

	boxes_batch_ptr = boxes_ptr + bs_idx * stride_boxes_batch
	points_batch_ptr = points_ptr + bs_idx * stride_points_batch
	out_batch_ptr = out_ptr + bs_idx * stride_out_batch

	px = tl.load(points_batch_ptr + pt_idx * stride_points_point + 0)
	py = tl.load(points_batch_ptr + pt_idx * stride_points_point + 1)
	pz = tl.load(points_batch_ptr + pt_idx * stride_points_point + 2)

	found = tl.full((), -1, tl.int32)
	for box_idx in range(MAX_BOXES):
		if box_idx < boxes_num:
			inside = _check_point_in_box3d(px, py, pz, boxes_batch_ptr, box_idx, stride_boxes_box)
			found = tl.where((found == -1) & inside, box_idx, found)
	tl.store(out_batch_ptr + pt_idx, found)


@triton.jit
def _points_in_boxes_batch_kernel(
	boxes_ptr,
	points_ptr,
	out_ptr,
	boxes_num,
	pts_num,
	stride_boxes_batch,
	stride_boxes_box,
	stride_points_batch,
	stride_points_point,
	stride_out_batch,
	stride_out_point,
	MAX_BOXES: tl.constexpr,
):
	pid = tl.program_id(axis=0)
	bs_idx = pid // pts_num
	pt_idx = pid % pts_num

	boxes_batch_ptr = boxes_ptr + bs_idx * stride_boxes_batch
	points_batch_ptr = points_ptr + bs_idx * stride_points_batch
	out_point_ptr = out_ptr + bs_idx * stride_out_batch + pt_idx * stride_out_point

	px = tl.load(points_batch_ptr + pt_idx * stride_points_point + 0)
	py = tl.load(points_batch_ptr + pt_idx * stride_points_point + 1)
	pz = tl.load(points_batch_ptr + pt_idx * stride_points_point + 2)

	for box_idx in range(MAX_BOXES):
		if box_idx < boxes_num:
			inside = _check_point_in_box3d(px, py, pz, boxes_batch_ptr, box_idx, stride_boxes_box)
			tl.store(out_point_ptr + box_idx, inside.to(tl.int32))


def points_in_boxes_gpu(points: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
	_check_points_boxes_batch(points, boxes)
	if not points.is_cuda or not boxes.is_cuda:
		raise NotImplementedError('points_in_boxes_gpu is only implemented for CUDA tensors.')

	batch_size, pts_num, _ = points.shape
	boxes_num = boxes.size(1)
	out = torch.full((batch_size, pts_num), -1, device=points.device, dtype=torch.int32)
	if out.numel() == 0:
		return out

	grid = (batch_size * pts_num,)
	_points_in_boxes_gpu_kernel[grid](
		boxes.contiguous(),
		points.contiguous(),
		out,
		boxes_num,
		pts_num,
		boxes.stride(0),
		boxes.stride(1),
		points.stride(0),
		points.stride(1),
		out.stride(0),
		MAX_BOXES=triton.next_power_of_2(max(boxes_num, 1)),
	)
	return out


def points_in_boxes_batch(points: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
	_check_points_boxes_batch(points, boxes)
	if not points.is_cuda or not boxes.is_cuda:
		raise NotImplementedError('points_in_boxes_batch is only implemented for CUDA tensors.')

	batch_size, pts_num, _ = points.shape
	boxes_num = boxes.size(1)
	out = torch.zeros((batch_size, pts_num, boxes_num), device=points.device, dtype=torch.int32)
	if out.numel() == 0:
		return out

	grid = (batch_size * pts_num,)
	_points_in_boxes_batch_kernel[grid](
		boxes.contiguous(),
		points.contiguous(),
		out,
		boxes_num,
		pts_num,
		boxes.stride(0),
		boxes.stride(1),
		points.stride(0),
		points.stride(1),
		out.stride(0),
		out.stride(1),
		MAX_BOXES=triton.next_power_of_2(max(boxes_num, 1)),
	)
	return out


def points_in_boxes_cpu(points: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
	_check_points_boxes_cpu(points, boxes)
	point_indices = torch.zeros(
		(boxes.size(0), points.size(0)), device=points.device, dtype=torch.int32
	)
	if points.numel() == 0 or boxes.numel() == 0:
		return point_indices

	for box_idx in range(boxes.size(0)):
		cx, cy, cz_bottom, w, l, h, rz = boxes[box_idx]
		cz = cz_bottom + h * 0.5
		dx = points[:, 0] - cx
		dy = points[:, 1] - cy
		rot_angle = rz + math.pi * 0.5
		cosa = math.cos(float(rot_angle))
		sina = math.sin(float(rot_angle))
		local_x = dx * cosa + dy * (-sina)
		local_y = dx * sina + dy * cosa
		mask = (
			((points[:, 2] - cz).abs() <= h * 0.5)
			& (local_x > -l * 0.5)
			& (local_x < l * 0.5)
			& (local_y > -w * 0.5)
			& (local_y < w * 0.5)
		)
		point_indices[box_idx, mask] = 1
	return point_indices
