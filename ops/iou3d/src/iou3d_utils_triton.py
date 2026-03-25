from __future__ import annotations

import torch
import triton
import triton.language as tl

_BOX_DIMS = 5
_NMS_BLOCK_SIZE = 256


def _check_boxes(boxes: torch.Tensor, *, name: str) -> None:
	if not boxes.is_cuda:
		raise NotImplementedError(f'{name} is only implemented for CUDA tensors.')
	if boxes.dtype != torch.float32:
		raise TypeError(f'{name} must have dtype torch.float32.')
	if boxes.dim() != 2 or boxes.size(-1) != _BOX_DIMS:
		raise ValueError(f'{name} must have shape [N, 5].')
	if not boxes.is_contiguous():
		raise ValueError(f'{name} must be contiguous.')


def _check_scores(scores: torch.Tensor) -> None:
	if not scores.is_cuda:
		raise NotImplementedError('scores is only implemented for CUDA tensors.')
	if scores.dtype != torch.float32:
		raise TypeError('scores must have dtype torch.float32.')
	if scores.dim() != 1:
		raise ValueError('scores must have shape [N].')
	if not scores.is_contiguous():
		raise ValueError('scores must be contiguous.')


@triton.jit
def _select(vec, idx, size: tl.constexpr):
	offsets = tl.arange(0, size)
	return tl.sum(tl.where(offsets == idx, vec, 0.0), axis=0)


@triton.jit
def _write(vec, idx, value, size: tl.constexpr):
	offsets = tl.arange(0, size)
	return tl.where(offsets == idx, value, vec)


@triton.jit
def _rotate_points(xs, ys, center_x, center_y, angle_cos, angle_sin, size: tl.constexpr):
	shift_x = xs - center_x
	shift_y = ys - center_y
	rot_x = shift_x * angle_cos + shift_y * angle_sin + center_x
	rot_y = -shift_x * angle_sin + shift_y * angle_cos + center_y
	return rot_x, rot_y


@triton.jit
def _clip_inside(px, py, ax, ay, bx, by):
	return (bx - ax) * (py - ay) - (by - ay) * (px - ax) >= 0.0


@triton.jit
def _segment_intersection(sx, sy, ex, ey, ax, ay, bx, by):
	dc_x = ax - bx
	dc_y = ay - by
	dp_x = sx - ex
	dp_y = sy - ey
	n1 = ax * by - ay * bx
	n2 = sx * ey - sy * ex
	den = dc_x * dp_y - dc_y * dp_x
	den = tl.where(tl.abs(den) < 1e-8, 1e-8, den)
	ix = (n1 * dp_x - dc_x * n2) / den
	iy = (n1 * dp_y - dc_y * n2) / den
	return ix, iy


@triton.jit
def _polygon_area(xs, ys, count):
	area = 0.0
	for i in range(8):
		if i < count:
			j = tl.where(i + 1 < count, i + 1, 0)
			x0 = _select(xs, i, 8)
			y0 = _select(ys, i, 8)
			x1 = _select(xs, j, 8)
			y1 = _select(ys, j, 8)
			area += x0 * y1 - y0 * x1
	return tl.abs(area) * 0.5


@triton.jit
def _box_overlap(box_a, box_b):
	a_x1 = _select(box_a, 0, 8)
	a_y1 = _select(box_a, 1, 8)
	a_x2 = _select(box_a, 2, 8)
	a_y2 = _select(box_a, 3, 8)
	a_angle = _select(box_a, 4, 8)

	b_x1 = _select(box_b, 0, 8)
	b_y1 = _select(box_b, 1, 8)
	b_x2 = _select(box_b, 2, 8)
	b_y2 = _select(box_b, 3, 8)
	b_angle = _select(box_b, 4, 8)

	subject_x = tl.zeros([8], dtype=tl.float32)
	subject_y = tl.zeros([8], dtype=tl.float32)
	clip_x = tl.zeros([4], dtype=tl.float32)
	clip_y = tl.zeros([4], dtype=tl.float32)

	subject_x = _write(subject_x, 0, a_x1, 8)
	subject_y = _write(subject_y, 0, a_y1, 8)
	subject_x = _write(subject_x, 1, a_x2, 8)
	subject_y = _write(subject_y, 1, a_y1, 8)
	subject_x = _write(subject_x, 2, a_x2, 8)
	subject_y = _write(subject_y, 2, a_y2, 8)
	subject_x = _write(subject_x, 3, a_x1, 8)
	subject_y = _write(subject_y, 3, a_y2, 8)

	clip_x = _write(clip_x, 0, b_x1, 4)
	clip_y = _write(clip_y, 0, b_y1, 4)
	clip_x = _write(clip_x, 1, b_x2, 4)
	clip_y = _write(clip_y, 1, b_y1, 4)
	clip_x = _write(clip_x, 2, b_x2, 4)
	clip_y = _write(clip_y, 2, b_y2, 4)
	clip_x = _write(clip_x, 3, b_x1, 4)
	clip_y = _write(clip_y, 3, b_y2, 4)

	a_center_x = 0.5 * (a_x1 + a_x2)
	a_center_y = 0.5 * (a_y1 + a_y2)
	b_center_x = 0.5 * (b_x1 + b_x2)
	b_center_y = 0.5 * (b_y1 + b_y2)

	subject_x, subject_y = _rotate_points(
		subject_x, subject_y, a_center_x, a_center_y, tl.cos(a_angle), tl.sin(a_angle), 8
	)
	clip_x, clip_y = _rotate_points(
		clip_x, clip_y, b_center_x, b_center_y, tl.cos(b_angle), tl.sin(b_angle), 4
	)

	count = 4
	for edge_idx in range(4):
		ax = _select(clip_x, edge_idx, 4)
		ay = _select(clip_y, edge_idx, 4)
		next_edge_idx = 0 if edge_idx == 3 else edge_idx + 1
		bx = _select(clip_x, next_edge_idx, 4)
		by = _select(clip_y, next_edge_idx, 4)

		out_x = tl.zeros([8], dtype=tl.float32)
		out_y = tl.zeros([8], dtype=tl.float32)
		out_count = 0

		for i in range(8):
			if i < count:
				j = tl.where(i + 1 < count, i + 1, 0)
				sx = _select(subject_x, i, 8)
				sy = _select(subject_y, i, 8)
				ex = _select(subject_x, j, 8)
				ey = _select(subject_y, j, 8)

				s_inside = _clip_inside(sx, sy, ax, ay, bx, by)
				e_inside = _clip_inside(ex, ey, ax, ay, bx, by)

				if s_inside and e_inside:
					out_x = _write(out_x, out_count, ex, 8)
					out_y = _write(out_y, out_count, ey, 8)
					out_count += 1
				elif s_inside and not e_inside:
					ix, iy = _segment_intersection(sx, sy, ex, ey, ax, ay, bx, by)
					out_x = _write(out_x, out_count, ix, 8)
					out_y = _write(out_y, out_count, iy, 8)
					out_count += 1
				elif (not s_inside) and e_inside:
					ix, iy = _segment_intersection(sx, sy, ex, ey, ax, ay, bx, by)
					out_x = _write(out_x, out_count, ix, 8)
					out_y = _write(out_y, out_count, iy, 8)
					out_count += 1
					out_x = _write(out_x, out_count, ex, 8)
					out_y = _write(out_y, out_count, ey, 8)
					out_count += 1

		subject_x = out_x
		subject_y = out_y
		count = out_count

	return tl.where(count > 0, _polygon_area(subject_x, subject_y, count), 0.0)


@triton.jit
def _rotated_iou(box_a, box_b):
	overlap = _box_overlap(box_a, box_b)
	sa = (_select(box_a, 2, 8) - _select(box_a, 0, 8)) * (
		_select(box_a, 3, 8) - _select(box_a, 1, 8)
	)
	sb = (_select(box_b, 2, 8) - _select(box_b, 0, 8)) * (
		_select(box_b, 3, 8) - _select(box_b, 1, 8)
	)
	return overlap / tl.maximum(sa + sb - overlap, 1e-8)


@triton.jit
def _normal_iou(box_a, box_b):
	left = tl.maximum(_select(box_a, 0, 8), _select(box_b, 0, 8))
	right = tl.minimum(_select(box_a, 2, 8), _select(box_b, 2, 8))
	top = tl.maximum(_select(box_a, 1, 8), _select(box_b, 1, 8))
	bottom = tl.minimum(_select(box_a, 3, 8), _select(box_b, 3, 8))
	width = tl.maximum(right - left, 0.0)
	height = tl.maximum(bottom - top, 0.0)
	inter = width * height
	sa = (_select(box_a, 2, 8) - _select(box_a, 0, 8)) * (
		_select(box_a, 3, 8) - _select(box_a, 1, 8)
	)
	sb = (_select(box_b, 2, 8) - _select(box_b, 0, 8)) * (
		_select(box_b, 3, 8) - _select(box_b, 1, 8)
	)
	return inter / tl.maximum(sa + sb - inter, 1e-8)


@triton.jit
def _pairwise_iou_kernel(
	boxes_a_ptr,
	boxes_b_ptr,
	out_ptr,
	num_a,
	num_b,
	stride_a0,
	stride_b0,
	stride_out0,
	stride_out1,
	mode: tl.constexpr,
):
	pid = tl.program_id(axis=0)
	row = pid // num_b
	col = pid % num_b
	mask = (row < num_a) & (col < num_b)
	box_cols = tl.arange(0, 8)

	box_a = tl.load(
		boxes_a_ptr + row * stride_a0 + box_cols,
		mask=mask & (box_cols < 5),
		other=0.0,
	)
	box_b = tl.load(
		boxes_b_ptr + col * stride_b0 + box_cols,
		mask=mask & (box_cols < 5),
		other=0.0,
	)
	if mode == 1:
		value = _normal_iou(box_a, box_b)
	elif mode == 2:
		value = _box_overlap(box_a, box_b)
	else:
		value = _rotated_iou(box_a, box_b)
	tl.store(out_ptr + row * stride_out0 + col * stride_out1, value, mask=mask)


@triton.jit
def _nms_suppress_kernel(
	iou_ptr,
	keep_mask_ptr,
	row_idx,
	num_boxes,
	stride_iou0,
	stride_iou1,
	thresh,
	BLOCK_SIZE: tl.constexpr,
):
	pid = tl.program_id(axis=0)
	cols = row_idx + 1 + pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
	mask = cols < num_boxes
	values = tl.load(iou_ptr + row_idx * stride_iou0 + cols * stride_iou1, mask=mask, other=0.0)
	cur = tl.load(keep_mask_ptr + cols, mask=mask, other=0).to(tl.int1)
	suppressed = values > thresh
	new_mask = cur & ~suppressed
	tl.store(keep_mask_ptr + cols, new_mask.to(tl.int8), mask=mask)


def _pairwise_metric(boxes_a: torch.Tensor, boxes_b: torch.Tensor, *, mode: int) -> torch.Tensor:
	_check_boxes(boxes_a, name='boxes_a')
	_check_boxes(boxes_b, name='boxes_b')
	out = torch.empty(
		(boxes_a.size(0), boxes_b.size(0)), device=boxes_a.device, dtype=torch.float32
	)
	if out.numel() == 0:
		return out
	grid = (boxes_a.size(0) * boxes_b.size(0),)
	_pairwise_iou_kernel[grid](
		boxes_a,
		boxes_b,
		out,
		boxes_a.size(0),
		boxes_b.size(0),
		boxes_a.stride(0),
		boxes_b.stride(0),
		out.stride(0),
		out.stride(1),
		mode=mode,
	)
	return out


def _boxes_overlap_bev(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
	return _pairwise_metric(boxes_a.contiguous(), boxes_b.contiguous(), mode=2)


def boxes_overlap_bev_gpu(
	boxes_a: torch.Tensor,
	boxes_b: torch.Tensor,
	ans_overlap: torch.Tensor,
) -> int:
	_check_boxes(boxes_a, name='boxes_a')
	_check_boxes(boxes_b, name='boxes_b')
	if not ans_overlap.is_cuda:
		raise NotImplementedError('ans_overlap is only implemented for CUDA tensors.')
	if ans_overlap.dtype != torch.float32:
		raise TypeError('ans_overlap must have dtype torch.float32.')
	if ans_overlap.dim() != 2 or ans_overlap.shape != (boxes_a.size(0), boxes_b.size(0)):
		raise ValueError('ans_overlap must have shape [boxes_a.size(0), boxes_b.size(0)].')
	if not ans_overlap.is_contiguous():
		raise ValueError('ans_overlap must be contiguous.')
	ans_overlap.copy_(_boxes_overlap_bev(boxes_a, boxes_b))
	return 1


def boxes_iou_bev(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
	return _pairwise_metric(boxes_a.contiguous(), boxes_b.contiguous(), mode=0)


def _nms_from_iou(
	boxes: torch.Tensor, scores: torch.Tensor, thresh: float, *, normal: bool
) -> torch.Tensor:
	_check_boxes(boxes, name='boxes')
	_check_scores(scores)
	if boxes.size(0) != scores.size(0):
		raise ValueError('boxes and scores must have the same leading dimension.')
	order = scores.sort(descending=True).indices
	boxes_sorted = boxes[order].contiguous()
	iou = _pairwise_metric(boxes_sorted, boxes_sorted, mode=1 if normal else 0)
	keep_mask = torch.ones(boxes_sorted.size(0), device=boxes.device, dtype=torch.int8)
	keep_indices: list[int] = []
	for i in range(boxes_sorted.size(0)):
		if not bool(keep_mask[i].item()):
			continue
		keep_indices.append(i)
		if i + 1 < boxes_sorted.size(0):
			num_remaining = boxes_sorted.size(0) - (i + 1)
			grid = (triton.cdiv(num_remaining, _NMS_BLOCK_SIZE),)
			_nms_suppress_kernel[grid](
				iou,
				keep_mask,
				i,
				boxes_sorted.size(0),
				iou.stride(0),
				iou.stride(1),
				thresh,
				BLOCK_SIZE=_NMS_BLOCK_SIZE,
			)
	keep = torch.tensor(keep_indices, device=boxes.device, dtype=torch.long)
	return order[keep]


def nms_gpu(
	boxes: torch.Tensor,
	scores: torch.Tensor,
	thresh: float,
	pre_maxsize: int | None = None,
	post_max_size: int | None = None,
) -> torch.Tensor:
	_check_boxes(boxes, name='boxes')
	_check_scores(scores)
	if boxes.size(0) != scores.size(0):
		raise ValueError('boxes and scores must have the same leading dimension.')
	if pre_maxsize is not None:
		presort = scores.sort(descending=True).indices[:pre_maxsize]
		boxes = boxes[presort].contiguous()
		scores = scores[presort].contiguous()
		keep = _nms_from_iou(boxes, scores, thresh, normal=False)
		keep = presort[keep]
	else:
		keep = _nms_from_iou(boxes.contiguous(), scores.contiguous(), thresh, normal=False)
	if post_max_size is not None:
		keep = keep[:post_max_size]
	return keep


def nms_normal_gpu(
	boxes: torch.Tensor,
	scores: torch.Tensor,
	thresh: float,
) -> torch.Tensor:
	return _nms_from_iou(boxes.contiguous(), scores.contiguous(), thresh, normal=True)


# TODO(triton-cpu-fallback):
# Add CPU fallback implementations for the above functions,
# and modify the above functions to call the CPU fallback when the input tensors are not CUDA tensors.
# The CPU fallback implementations can be simple and not optimized, as they are only intended for correctness testing against the CUDA implementations.
