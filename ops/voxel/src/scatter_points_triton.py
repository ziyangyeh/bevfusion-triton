from __future__ import annotations

import torch
import triton
import triton.language as tl

from .coord_key_triton import decode_keys_to_coors_kernel, encode_coors_to_keys_kernel


_REDUCTION_TO_ID = {'none': 0, 'mean': 1, 'sum': 2}
_REDUCE_MODE = set(_REDUCTION_TO_ID) | {'max'}
_MAX_REDUCTION_ID = 3
_CHANNEL_BLOCK = 128


@triton.jit
def _clean_coors_kernel(
	coors_ptr,
	out_ptr,
	num_rows,
	num_dims,
	BLOCK_D: tl.constexpr,
):
	row = tl.program_id(axis=0)
	if row >= num_rows:
		return
	cols = tl.arange(0, BLOCK_D)
	mask = cols < num_dims
	base = row * num_dims
	values = tl.load(coors_ptr + base + cols, mask=mask, other=0)
	has_invalid = tl.sum(tl.where(mask & (values < 0), 1, 0), axis=0) > 0
	out_values = tl.where(has_invalid, -1, values)
	tl.store(out_ptr + base + cols, out_values, mask=mask)


@triton.jit
def _feats_reduce_kernel(
	feats_ptr,
	coors_map_ptr,
	reduced_feats_ptr,
	num_feats,
	num_feat_blocks,
	num_rows,
	reduce_type: tl.constexpr,
	accumulate_fp32: tl.constexpr,
	BLOCK_SIZE: tl.constexpr,
):
	pid = tl.program_id(axis=0)
	if pid >= num_rows * num_feat_blocks:
		return
	x = pid // num_feat_blocks
	feat_block = pid % num_feat_blocks
	cols = feat_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
	mask = cols < num_feats
	dst = tl.load(coors_map_ptr + x).to(tl.int32)
	if dst != -1:
		value = tl.load(feats_ptr + x * num_feats + cols, mask=mask)
		if accumulate_fp32:
			value = value.to(tl.float32)
		out_ptr = reduced_feats_ptr + dst * num_feats + cols
		if reduce_type == 3:
			tl.atomic_max(out_ptr, value, mask=mask)
		else:
			tl.atomic_add(out_ptr, value, mask=mask)


@triton.jit
def _add_reduce_traceback_grad_kernel(
	grad_feats_ptr,
	grad_reduced_feats_ptr,
	coors_map_ptr,
	reduce_count_ptr,
	num_feats,
	num_feat_blocks,
	num_rows,
	reduce_type: tl.constexpr,
	BLOCK_SIZE: tl.constexpr,
):
	pid = tl.program_id(axis=0)
	if pid >= num_rows * num_feat_blocks:
		return
	x = pid // num_feat_blocks
	feat_block = pid % num_feat_blocks
	cols = feat_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
	mask = cols < num_feats
	dst = tl.load(coors_map_ptr + x).to(tl.int32)
	if dst != -1:
		value = tl.load(grad_reduced_feats_ptr + dst * num_feats + cols, mask=mask)
		if reduce_type == 1:
			count = tl.load(reduce_count_ptr + dst).to(tl.float32)
			value = value / count.to(value.dtype)
		tl.store(grad_feats_ptr + x * num_feats + cols, value, mask=mask)


@triton.jit
def _max_reduce_traceback_scatter_idx_kernel(
	feats_ptr,
	reduced_feats_ptr,
	reduce_from_ptr,
	coors_map_ptr,
	num_feats,
	num_feat_blocks,
	num_rows,
	num_input,
	BLOCK_SIZE: tl.constexpr,
):
	pid = tl.program_id(axis=0)
	if pid >= num_rows * num_feat_blocks:
		return
	x = pid // num_feat_blocks
	feat_block = pid % num_feat_blocks
	cols = feat_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
	mask = cols < num_feats
	dst = tl.load(coors_map_ptr + x).to(tl.int32)
	if dst != -1:
		reduced_offset = dst * num_feats + cols
		feat = tl.load(feats_ptr + x * num_feats + cols, mask=mask)
		reduced = tl.load(reduced_feats_ptr + reduced_offset, mask=mask)
		update_mask = mask & (feat == reduced)
		value = tl.full((BLOCK_SIZE,), num_input, dtype=tl.int32)
		value = tl.where(update_mask, x.to(tl.int32), value)
		tl.atomic_min(reduce_from_ptr + reduced_offset, value, mask=mask)


@triton.jit
def _max_reduce_scatter_grad_kernel(
	grad_feats_ptr,
	grad_reduced_feats_ptr,
	reduce_from_ptr,
	num_feats,
	num_feat_blocks,
	num_rows,
	num_input,
	BLOCK_SIZE: tl.constexpr,
):
	pid = tl.program_id(axis=0)
	if pid >= num_rows * num_feat_blocks:
		return
	x = pid // num_feat_blocks
	feat_block = pid % num_feat_blocks
	cols = feat_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
	mask = cols < num_feats
	src = tl.load(reduce_from_ptr + x * num_feats + cols, mask=mask, other=num_input).to(tl.int32)
	value = tl.load(grad_reduced_feats_ptr + x * num_feats + cols, mask=mask)
	store_mask = mask & (src < num_input)
	tl.store(grad_feats_ptr + src * num_feats + cols, value, mask=store_mask)


def _clean_coors(coors: torch.Tensor) -> torch.Tensor:
	if coors.device.type == 'cpu':
		mask = coors.lt(0).any(dim=-1, keepdim=True)
		return coors.masked_fill(mask, -1)
	if coors.numel() == 0:
		return coors.clone()
	clean = torch.empty_like(coors)
	_clean_coors_kernel[(int(coors.size(0)),)](
		coors.contiguous(),
		clean,
		int(coors.size(0)),
		int(coors.size(1)),
		BLOCK_D=4,
	)
	return clean


def _unique_with_inverse_and_counts(
	coors: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
	clean = _clean_coors(coors)
	if coors.device.type == 'cpu':
		out_coors, coors_map, reduce_count = torch.unique(
			clean,
			sorted=True,
			return_inverse=True,
			return_counts=True,
			dim=0,
		)
		if out_coors.numel() > 0 and bool(out_coors[0, 0].lt(0).item()):
			out_coors = out_coors[1:]
			reduce_count = reduce_count[1:]
			coors_map = coors_map - 1
		return out_coors, coors_map.to(torch.int32), reduce_count.to(torch.int32)

	valid_mask = clean[:, 0] >= 0
	if bool(valid_mask.any().item()):
		valid_coors = clean[valid_mask]
		size_vec = valid_coors.to(torch.int64).amax(dim=0) + 1
	else:
		size_vec = torch.ones((clean.size(1),), device=clean.device, dtype=torch.int64)
	keys = torch.empty((clean.size(0),), device=clean.device, dtype=torch.int64)
	encode_coors_to_keys_kernel[(int(clean.size(0)),)](
		clean.contiguous(),
		keys,
		int(clean.size(0)),
		int(clean.size(1)),
		int(size_vec[0].item()) if size_vec.numel() >= 1 else 1,
		int(size_vec[1].item()) if size_vec.numel() >= 2 else 1,
		int(size_vec[2].item()) if size_vec.numel() >= 3 else 1,
		int(size_vec[3].item()) if size_vec.numel() >= 4 else 1,
	)
	unique_keys, coors_map, reduce_count = torch.unique(
		keys,
		sorted=True,
		return_inverse=True,
		return_counts=True,
	)
	if unique_keys.numel() > 0 and bool(unique_keys[0].lt(0).item()):
		unique_keys = unique_keys[1:]
		reduce_count = reduce_count[1:]
		coors_map = coors_map - 1
	out_coors = torch.empty(
		(unique_keys.size(0), clean.size(1)),
		device=clean.device,
		dtype=torch.int32,
	)
	if unique_keys.numel() > 0:
		decode_keys_to_coors_kernel[(int(unique_keys.size(0)),)](
			unique_keys.contiguous(),
			out_coors,
			int(unique_keys.size(0)),
			int(clean.size(1)),
			int(size_vec[0].item()) if size_vec.numel() >= 1 else 1,
			int(size_vec[1].item()) if size_vec.numel() >= 2 else 1,
			int(size_vec[2].item()) if size_vec.numel() >= 3 else 1,
			int(size_vec[3].item()) if size_vec.numel() >= 4 else 1,
		)
	return out_coors, coors_map.to(torch.int32), reduce_count.to(torch.int32)


def _dynamic_point_to_voxel_forward_cpu(
	feats: torch.Tensor,
	coors: torch.Tensor,
	reduce_type: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
	out_coors, coors_map, reduce_count = _unique_with_inverse_and_counts(coors)
	num_reduced = int(out_coors.size(0))
	if reduce_type == 'max':
		reduced_feats = feats.new_full((num_reduced, feats.size(1)), float('-inf'))
		for row in range(int(feats.size(0))):
			dst = int(coors_map[row].item())
			if dst >= 0:
				reduced_feats[dst] = torch.maximum(reduced_feats[dst], feats[row])
	else:
		reduced_feats = feats.new_zeros((num_reduced, feats.size(1)))
		valid_mask = coors_map >= 0
		if valid_mask.any():
			reduced_feats.index_add_(0, coors_map[valid_mask].to(torch.long), feats[valid_mask])
		if reduce_type == 'mean' and num_reduced > 0:
			reduced_feats = reduced_feats / reduce_count.to(reduced_feats.dtype).unsqueeze(-1)
	return reduced_feats, out_coors, coors_map, reduce_count


def dynamic_point_to_voxel_forward(
	feats: torch.Tensor,
	coors: torch.Tensor,
	reduce_type: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
	if reduce_type not in _REDUCE_MODE:
		raise ValueError(f'Unsupported reduce_type: {reduce_type}')
	if feats.dim() != 2:
		raise AssertionError('feats must have shape [N, C]')
	if coors.dim() != 2:
		raise AssertionError('coors must have shape [N, ndim]')
	if feats.size(0) != coors.size(0):
		raise AssertionError('feats and coors must have the same first dimension')

	num_input = int(feats.size(0))
	if num_input == 0:
		return (
			feats.clone().detach(),
			coors.clone().detach(),
			coors.new_empty((0,), dtype=torch.int32),
			coors.new_empty((0,), dtype=torch.int32),
		)
	if not feats.is_cuda:
		return _dynamic_point_to_voxel_forward_cpu(feats, coors, reduce_type)

	feats = feats.contiguous()
	coors = coors.contiguous()
	out_coors, coors_map, reduce_count = _unique_with_inverse_and_counts(coors)
	accumulate_fp32 = reduce_type == 'max' and feats.dtype in {torch.float16, torch.bfloat16}
	reduced_dtype = torch.float32 if accumulate_fp32 else feats.dtype
	reduced_feats = torch.empty(
		(out_coors.size(0), feats.size(1)),
		device=feats.device,
		dtype=reduced_dtype,
	)
	if reduce_type == 'max':
		reduced_feats.fill_(float('-inf'))
		reduce_id = _MAX_REDUCTION_ID
	else:
		reduced_feats.zero_()
		reduce_id = _REDUCTION_TO_ID[reduce_type]

	if num_input > 0 and reduced_feats.numel() > 0:
		num_feats = int(feats.size(1))
		num_feat_blocks = triton.cdiv(num_feats, _CHANNEL_BLOCK)
		total_blocks = num_input * num_feat_blocks
		_feats_reduce_kernel[(total_blocks,)](
			feats,
			coors_map,
			reduced_feats,
			num_feats,
			num_feat_blocks,
			num_input,
			reduce_type=reduce_id,
			accumulate_fp32=accumulate_fp32,
			BLOCK_SIZE=_CHANNEL_BLOCK,
		)
		if reduce_type == 'mean':
			reduced_feats /= reduce_count.to(reduced_feats.dtype).unsqueeze(-1)
	if accumulate_fp32:
		reduced_feats = reduced_feats.to(feats.dtype)
	return reduced_feats, out_coors, coors_map, reduce_count


def _dynamic_point_to_voxel_backward_cpu(
	grad_feats: torch.Tensor,
	grad_reduced_feats: torch.Tensor,
	feats: torch.Tensor,
	reduced_feats: torch.Tensor,
	coors_map: torch.Tensor,
	reduce_count: torch.Tensor,
	reduce_type: str,
) -> None:
	grad_feats.zero_()
	if reduce_type in {'sum', 'mean'}:
		valid_mask = coors_map >= 0
		if valid_mask.any():
			grad = grad_reduced_feats[coors_map[valid_mask].to(torch.long)]
			if reduce_type == 'mean':
				grad = grad / reduce_count[coors_map[valid_mask].to(torch.long)].to(
					grad.dtype
				).unsqueeze(-1)
			grad_feats[valid_mask] = grad
		return

	for voxel_idx in range(int(reduced_feats.size(0))):
		for feat_idx in range(int(feats.size(1))):
			for point_idx in range(int(feats.size(0))):
				if int(coors_map[point_idx].item()) != voxel_idx:
					continue
				if feats[point_idx, feat_idx] == reduced_feats[voxel_idx, feat_idx]:
					grad_feats[point_idx, feat_idx] = grad_reduced_feats[voxel_idx, feat_idx]
					break


def dynamic_point_to_voxel_backward(
	grad_feats: torch.Tensor,
	grad_reduced_feats: torch.Tensor,
	feats: torch.Tensor,
	reduced_feats: torch.Tensor,
	coors_map: torch.Tensor,
	reduce_count: torch.Tensor,
	reduce_type: str,
) -> None:
	if reduce_type not in _REDUCE_MODE:
		raise ValueError(f'Unsupported reduce_type: {reduce_type}')
	if not grad_feats.is_cuda:
		_dynamic_point_to_voxel_backward_cpu(
			grad_feats,
			grad_reduced_feats,
			feats,
			reduced_feats,
			coors_map,
			reduce_count,
			reduce_type,
		)
		return

	grad_feats.zero_()
	num_input = int(feats.size(0))
	num_reduced = int(reduced_feats.size(0))
	if num_input == 0 or num_reduced == 0:
		return

	if reduce_type in {'sum', 'mean'}:
		num_feats = int(feats.size(1))
		num_feat_blocks = triton.cdiv(num_feats, _CHANNEL_BLOCK)
		total_blocks = num_input * num_feat_blocks
		_add_reduce_traceback_grad_kernel[(total_blocks,)](
			grad_feats,
			grad_reduced_feats.contiguous(),
			coors_map.contiguous(),
			reduce_count.contiguous(),
			num_feats,
			num_feat_blocks,
			num_input,
			reduce_type=_REDUCTION_TO_ID[reduce_type],
			BLOCK_SIZE=_CHANNEL_BLOCK,
		)
		return

	reduce_from = torch.full(
		(num_reduced, feats.size(1)),
		num_input,
		device=feats.device,
		dtype=torch.int32,
	)
	num_feats = int(feats.size(1))
	num_feat_blocks = triton.cdiv(num_feats, _CHANNEL_BLOCK)
	input_blocks = num_input * num_feat_blocks
	_max_reduce_traceback_scatter_idx_kernel[(input_blocks,)](
		feats.contiguous(),
		reduced_feats.contiguous(),
		reduce_from,
		coors_map.contiguous(),
		num_feats,
		num_feat_blocks,
		num_input,
		num_input,
		BLOCK_SIZE=_CHANNEL_BLOCK,
	)
	reduced_blocks = num_reduced * num_feat_blocks
	_max_reduce_scatter_grad_kernel[(reduced_blocks,)](
		grad_feats,
		grad_reduced_feats.contiguous(),
		reduce_from,
		num_feats,
		num_feat_blocks,
		num_reduced,
		num_input,
		BLOCK_SIZE=_CHANNEL_BLOCK,
	)


__all__ = [
	'dynamic_point_to_voxel_backward',
	'dynamic_point_to_voxel_forward',
]
