from __future__ import annotations

import math
from itertools import product

import torch
import triton
import triton.language as tl

from .sparse_ops_triton import sparse_gather, sparse_scatter_add


_BLOCK_SIZE = 256
_INDEX_MAP_MODE = {
	'mark': 0,
	'write_identity': 1,
	'write': 2,
	'lookup': 3,
}


@triton.jit
def _flatten_indices_kernel(
	indices_ptr,
	out_ptr,
	num_rows,
	spatial_volume,
	shape0,
	shape1,
	shape2,
	shape3,
	ndim: tl.constexpr,
):
	row = tl.program_id(axis=0)
	if row >= num_rows:
		return
	base = row * (ndim + 1)
	batch = tl.load(indices_ptr + base + 0).to(tl.int64)
	c0 = tl.load(indices_ptr + base + 1).to(tl.int64)
	flat = c0
	if ndim >= 2:
		c1 = tl.load(indices_ptr + base + 2).to(tl.int64)
		flat = flat * shape1 + c1
	if ndim >= 3:
		c2 = tl.load(indices_ptr + base + 3).to(tl.int64)
		flat = flat * shape2 + c2
	if ndim >= 4:
		c3 = tl.load(indices_ptr + base + 4).to(tl.int64)
		flat = flat * shape3 + c3
	tl.store(out_ptr + row, batch * spatial_volume + flat)


@triton.jit
def _decode_outids_kernel(
	unique_indices_ptr,
	out_ptr,
	num_rows,
	spatial_volume,
	out_shape0,
	out_shape1,
	out_shape2,
	out_shape3,
	ndim: tl.constexpr,
):
	row = tl.program_id(axis=0)
	if row >= num_rows:
		return
	flat = tl.load(unique_indices_ptr + row).to(tl.int64)
	batch = flat // spatial_volume
	spatial = flat % spatial_volume
	base = row * (ndim + 1)
	tl.store(out_ptr + base + 0, batch.to(tl.int32))
	if ndim >= 4:
		c3 = spatial % out_shape3
		spatial = spatial // out_shape3
		tl.store(out_ptr + base + 4, c3.to(tl.int32))
	if ndim >= 3:
		c2 = spatial % out_shape2
		spatial = spatial // out_shape2
		tl.store(out_ptr + base + 3, c2.to(tl.int32))
	if ndim >= 2:
		c1 = spatial % out_shape1
		spatial = spatial // out_shape1
		tl.store(out_ptr + base + 2, c1.to(tl.int32))
	c0 = spatial % out_shape0
	tl.store(out_ptr + base + 1, c0.to(tl.int32))


@triton.jit
def _generate_rule_candidates_kernel(
	indices_ptr,
	flat_out_ptr,
	valid_ptr,
	num_act_in,
	kernel_volume,
	stride0,
	stride1,
	stride2,
	stride3,
	padding0,
	padding1,
	padding2,
	padding3,
	dilation0,
	dilation1,
	dilation2,
	dilation3,
	ksize0,
	ksize1,
	ksize2,
	ksize3,
	out_shape0,
	out_shape1,
	out_shape2,
	out_shape3,
	spatial_volume,
	ndim: tl.constexpr,
	transpose: tl.constexpr,
	BLOCK_SIZE: tl.constexpr,
):
	pid = tl.program_id(axis=0)
	offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
	nthreads = num_act_in * kernel_volume
	mask = offs < nthreads

	row = offs // kernel_volume
	k_linear = offs % kernel_volume
	base = row * (ndim + 1)
	batch = tl.load(indices_ptr + base + 0, mask=mask, other=0).to(tl.int64)

	rem = k_linear
	k3 = tl.zeros_like(rem)
	k2 = tl.zeros_like(rem)
	k1 = tl.zeros_like(rem)
	k0 = tl.zeros_like(rem)
	if ndim >= 4:
		k3 = rem % ksize3
		rem = rem // ksize3
	if ndim >= 3:
		k2 = rem % ksize2
		rem = rem // ksize2
	if ndim >= 2:
		k1 = rem % ksize1
		rem = rem // ksize1
	k0 = rem % ksize0

	valid = mask
	flat_spatial = tl.zeros_like(batch)

	in0 = tl.load(indices_ptr + base + 1, mask=mask, other=0).to(tl.int64)
	if transpose:
		out0 = in0 * stride0 - padding0 + k0 * dilation0
		valid = valid & (out0 >= 0) & (out0 < out_shape0)
	else:
		numer0 = in0 + padding0 - k0 * dilation0
		valid = valid & (numer0 % stride0 == 0)
		out0 = numer0 // stride0
		valid = valid & (out0 >= 0) & (out0 < out_shape0)
	flat_spatial = out0

	if ndim >= 2:
		in1 = tl.load(indices_ptr + base + 2, mask=mask, other=0).to(tl.int64)
		if transpose:
			out1 = in1 * stride1 - padding1 + k1 * dilation1
			valid = valid & (out1 >= 0) & (out1 < out_shape1)
		else:
			numer1 = in1 + padding1 - k1 * dilation1
			valid = valid & (numer1 % stride1 == 0)
			out1 = numer1 // stride1
			valid = valid & (out1 >= 0) & (out1 < out_shape1)
		flat_spatial = flat_spatial * out_shape1 + out1
	if ndim >= 3:
		in2 = tl.load(indices_ptr + base + 3, mask=mask, other=0).to(tl.int64)
		if transpose:
			out2 = in2 * stride2 - padding2 + k2 * dilation2
			valid = valid & (out2 >= 0) & (out2 < out_shape2)
		else:
			numer2 = in2 + padding2 - k2 * dilation2
			valid = valid & (numer2 % stride2 == 0)
			out2 = numer2 // stride2
			valid = valid & (out2 >= 0) & (out2 < out_shape2)
		flat_spatial = flat_spatial * out_shape2 + out2
	if ndim >= 4:
		in3 = tl.load(indices_ptr + base + 4, mask=mask, other=0).to(tl.int64)
		if transpose:
			out3 = in3 * stride3 - padding3 + k3 * dilation3
			valid = valid & (out3 >= 0) & (out3 < out_shape3)
		else:
			numer3 = in3 + padding3 - k3 * dilation3
			valid = valid & (numer3 % stride3 == 0)
			out3 = numer3 // stride3
			valid = valid & (out3 >= 0) & (out3 < out_shape3)
		flat_spatial = flat_spatial * out_shape3 + out3

	flat_out = batch * spatial_volume + flat_spatial
	tl.store(flat_out_ptr + offs, flat_out, mask=mask)
	tl.store(valid_ptr + offs, valid.to(tl.int8), mask=mask)


@triton.jit
def _pack_indice_pairs_kernel(
	sorted_offsets_ptr,
	sorted_in_ptr,
	sorted_out_ptr,
	slot_ptr,
	indice_pairs_ptr,
	num_valid,
	num_act_in,
):
	idx = tl.program_id(axis=0)
	if idx >= num_valid:
		return
	offset = tl.load(sorted_offsets_ptr + idx).to(tl.int64)
	slot = tl.load(slot_ptr + idx).to(tl.int64)
	in_idx = tl.load(sorted_in_ptr + idx)
	out_idx = tl.load(sorted_out_ptr + idx)
	base = offset * 2 * num_act_in + slot
	tl.store(indice_pairs_ptr + base, in_idx)
	tl.store(indice_pairs_ptr + base + num_act_in, out_idx)


@triton.jit
def _index_map_kernel(
	keys_ptr,
	values_ptr,
	index_map_ptr,
	out_ptr,
	num_keys,
	mode: tl.constexpr,
):
	idx = tl.program_id(axis=0)
	if idx >= num_keys:
		return
	key = tl.load(keys_ptr + idx).to(tl.int64)
	if mode == 0:
		tl.store(index_map_ptr + key, 1)
	elif mode == 1:
		tl.store(index_map_ptr + key, idx.to(tl.int32))
	elif mode == 2:
		value = tl.load(values_ptr + idx).to(tl.int32)
		tl.store(index_map_ptr + key, value)
	else:
		value = tl.load(index_map_ptr + key)
		tl.store(out_ptr + idx, value)


@triton.jit
def _gather_marked_indices_kernel(
	flags_ptr,
	pos_ptr,
	out_ptr,
	num_items,
):
	idx = tl.program_id(axis=0)
	if idx >= num_items:
		return
	flag = tl.load(flags_ptr + idx).to(tl.int32)
	if flag == 0:
		return
	pos = tl.load(pos_ptr + idx).to(tl.int64)
	tl.store(out_ptr + pos, idx)


@triton.jit
def _pack_indice_pairs_by_rule_kernel(
	valid_offsets_ptr,
	valid_in_ptr,
	valid_out_ptr,
	indice_pairs_ptr,
	indice_num_ptr,
	num_valid,
	num_act_in,
):
	rule_id = tl.program_id(axis=0)
	slot = 0
	idx = 0
	while idx < num_valid:
		offset = tl.load(valid_offsets_ptr + idx).to(tl.int32)
		if offset == rule_id:
			in_idx = tl.load(valid_in_ptr + idx)
			out_idx = tl.load(valid_out_ptr + idx)
			base = rule_id * 2 * num_act_in + slot
			tl.store(indice_pairs_ptr + base, in_idx)
			tl.store(indice_pairs_ptr + base + num_act_in, out_idx)
			slot += 1
		idx += 1
	tl.store(indice_num_ptr + rule_id, slot)


def _normalize_nd_param(value, ndim: int) -> list[int]:
	if isinstance(value, (list, tuple)):
		if len(value) != ndim:
			raise ValueError(f'Expected parameter of length {ndim}, got {len(value)}')
		return [int(v) for v in value]
	return [int(value)] * ndim


def _get_conv_output_size(input_size, kernel_size, stride, padding, dilation):
	output_size = []
	for i in range(len(input_size)):
		size = (input_size[i] + 2 * padding[i] - dilation[i] * (kernel_size[i] - 1) - 1) // stride[
			i
		] + 1
		if kernel_size[i] == -1:
			output_size.append(1)
		else:
			output_size.append(size)
	return output_size


def _get_deconv_output_size(input_size, kernel_size, stride, padding, dilation, output_padding):
	output_size = []
	for i in range(len(input_size)):
		if kernel_size[i] == -1:
			raise ValueError("deconv don't support kernel_size < 0")
		size = (input_size[i] - 1) * stride[i] - 2 * padding[i] + kernel_size[i] + output_padding[i]
		output_size.append(size)
	return output_size


def _row_array_idx(coords: list[int], shape: list[int]) -> int:
	index = 0
	for coord, dim in zip(coords, shape):
		index = index * dim + coord
	return index


def _row_array_idx_inv(index: int, shape: list[int]) -> list[int]:
	coords = [0] * len(shape)
	for i in range(len(shape) - 1, -1, -1):
		coords[i] = index % shape[i]
		index //= shape[i]
	return coords


def _flatten_indices(
	indices: torch.Tensor, spatial_shape: list[int], spatial_volume: int
) -> torch.Tensor:
	if indices.numel() == 0:
		return torch.empty((0,), device=indices.device, dtype=torch.int64)
	if indices.is_cuda:
		out = torch.empty((indices.size(0),), device=indices.device, dtype=torch.int64)
		_flatten_indices_kernel[(int(indices.size(0)),)](
			indices.contiguous(),
			out,
			int(indices.size(0)),
			spatial_volume,
			spatial_shape[0],
			spatial_shape[1] if len(spatial_shape) >= 2 else 1,
			spatial_shape[2] if len(spatial_shape) >= 3 else 1,
			spatial_shape[3] if len(spatial_shape) >= 4 else 1,
			ndim=len(spatial_shape),
		)
		return out
	spatial = indices[:, 1:].to(torch.int64)
	flat_spatial = spatial[:, 0]
	for dim_idx, dim in enumerate(spatial_shape[1:], start=1):
		flat_spatial = flat_spatial * dim + spatial[:, dim_idx]
	return indices[:, 0].to(torch.int64) * spatial_volume + flat_spatial


def _decode_outids_from_unique(
	unique_indices: torch.Tensor,
	out_spatial_shape: list[int],
	spatial_volume: int,
	dtype: torch.dtype,
) -> torch.Tensor:
	if unique_indices.numel() == 0:
		return torch.empty(
			(0, len(out_spatial_shape) + 1), device=unique_indices.device, dtype=dtype
		)
	if unique_indices.is_cuda:
		out = torch.empty(
			(unique_indices.size(0), len(out_spatial_shape) + 1),
			device=unique_indices.device,
			dtype=torch.int32,
		)
		_decode_outids_kernel[(int(unique_indices.size(0)),)](
			unique_indices.contiguous(),
			out,
			int(unique_indices.size(0)),
			spatial_volume,
			out_spatial_shape[0],
			out_spatial_shape[1] if len(out_spatial_shape) >= 2 else 1,
			out_spatial_shape[2] if len(out_spatial_shape) >= 3 else 1,
			out_spatial_shape[3] if len(out_spatial_shape) >= 4 else 1,
			ndim=len(out_spatial_shape),
		)
		return out.to(dtype)
	coords = []
	spatial_index = unique_indices % spatial_volume
	for dim in reversed(out_spatial_shape):
		coords.append((spatial_index % dim).to(dtype))
		spatial_index = spatial_index // dim
	coords.reverse()
	batch = (unique_indices // spatial_volume).to(dtype).unsqueeze(1)
	return torch.cat([batch, *[coord.unsqueeze(1) for coord in coords]], dim=1)


def _borrow_or_alloc_index_map(
	grid: torch.Tensor | None,
	total_slots: int,
	device: torch.device,
) -> tuple[torch.Tensor, bool]:
	if grid is not None:
		flat = grid.view(-1)
		if int(flat.numel()) < total_slots:
			raise ValueError('grid scratch tensor is smaller than required flat map size')
		index_map = flat[:total_slots]
		index_map.fill_(-1)
		return index_map, True
	index_map = torch.full((total_slots,), -1, device=device, dtype=torch.int32)
	return index_map, False


def _index_map_op(
	op: str,
	keys: torch.Tensor,
	index_map: torch.Tensor,
	values: torch.Tensor | None = None,
) -> torch.Tensor | None:
	if op == 'lookup':
		out = torch.empty((int(keys.numel()),), device=keys.device, dtype=torch.int32)
		if keys.numel() == 0:
			return out
		_index_map_kernel[(int(keys.numel()),)](
			keys.contiguous(),
			torch.empty((0,), device=keys.device, dtype=torch.int32),
			index_map,
			out,
			int(keys.numel()),
			mode=_INDEX_MAP_MODE['lookup'],
		)
		return out
	if keys.numel() == 0:
		return None
	if op == 'write':
		assert values is not None
		_index_map_kernel[(int(keys.numel()),)](
			keys.contiguous(),
			values.contiguous(),
			index_map,
			torch.empty((0,), device=keys.device, dtype=torch.int32),
			int(keys.numel()),
			mode=_INDEX_MAP_MODE['write'],
		)
		return None
	if op == 'write_identity':
		_index_map_kernel[(int(keys.numel()),)](
			keys.contiguous(),
			torch.empty((0,), device=keys.device, dtype=torch.int32),
			index_map,
			torch.empty((0,), device=keys.device, dtype=torch.int32),
			int(keys.numel()),
			mode=_INDEX_MAP_MODE['write_identity'],
		)
		return None
	if op == 'mark':
		_index_map_kernel[(int(keys.numel()),)](
			keys.contiguous(),
			torch.empty((0,), device=keys.device, dtype=torch.int32),
			index_map,
			torch.empty((0,), device=keys.device, dtype=torch.int32),
			int(keys.numel()),
			mode=_INDEX_MAP_MODE['mark'],
		)
		return None
	raise ValueError(f'Unsupported index map op: {op}')


def _compact_marked_indices(flags: torch.Tensor) -> torch.Tensor:
	if flags.numel() == 0:
		return torch.empty((0,), device=flags.device, dtype=torch.int64)
	flag_i32 = flags.to(torch.int32)
	pos = torch.cumsum(flag_i32, dim=0) - flag_i32
	total = int(flag_i32.sum().item())
	if total == 0:
		return torch.empty((0,), device=flags.device, dtype=torch.int64)
	out = torch.empty((total,), device=flags.device, dtype=torch.int64)
	_gather_marked_indices_kernel[(int(flags.numel()),)](
		flag_i32.contiguous(),
		pos.contiguous(),
		out,
		int(flags.numel()),
	)
	return out


def _pack_indice_pairs(
	indice_pairs: torch.Tensor,
	indice_num: torch.Tensor,
	valid_offsets: torch.Tensor,
	valid_in: torch.Tensor,
	valid_out: torch.Tensor,
	kernel_volume: int,
) -> None:
	if valid_offsets.numel() == 0:
		return
	if valid_offsets.is_cuda:
		num_valid = int(valid_offsets.numel())
		_pack_indice_pairs_by_rule_kernel[(kernel_volume,)](
			valid_offsets.contiguous(),
			valid_in.contiguous(),
			valid_out.contiguous(),
			indice_pairs,
			indice_num,
			num_valid,
			int(indice_pairs.size(2)),
		)
	else:
		order = torch.argsort(valid_offsets, stable=True)
		sorted_offsets = valid_offsets[order].to(torch.int64)
		sorted_in = valid_in[order]
		sorted_out = valid_out[order]
		num_valid = int(sorted_offsets.numel())
		slot = torch.arange(num_valid, device=sorted_offsets.device, dtype=torch.int64)
		group_start = torch.ones((num_valid,), device=sorted_offsets.device, dtype=torch.bool)
		group_start[1:] = sorted_offsets[1:] != sorted_offsets[:-1]
		start_idx = torch.where(group_start, slot, torch.zeros_like(slot))
		start_idx = torch.cummax(start_idx, dim=0).values
		slot = slot - start_idx
		indice_pairs[sorted_offsets, 0, slot] = sorted_in
		indice_pairs[sorted_offsets, 1, slot] = sorted_out
		counts = torch.bincount(sorted_offsets, minlength=kernel_volume)
		indice_num.copy_(counts.to(indice_num.dtype))


def _valid_out_pos(
	input_pos: list[int],
	kernel_size: list[int],
	stride: list[int],
	padding: list[int],
	dilation: list[int],
	out_spatial_shape: list[int],
) -> list[tuple[list[int], int]]:
	ndim = len(input_pos)
	lowers = []
	uppers = []
	for i in range(ndim):
		lower = (
			input_pos[i] - (kernel_size[i] - 1) * dilation[i] - 1 + stride[i] + padding[i]
		) // stride[i]
		upper = (input_pos[i] + padding[i]) // stride[i]
		lowers.append(lower)
		uppers.append(upper)

	counter_sizes = []
	for i in range(ndim):
		counter_sizes.append((uppers[i] - lowers[i]) // dilation[i] + 1)

	valid_points: list[tuple[list[int], int]] = []
	for counter in product(*[range(size) for size in counter_sizes]):
		coords = [0] * ndim
		offset = 0
		mult = 1
		valid = True
		for j in range(ndim - 1, -1, -1):
			val = uppers[j] - counter[j] * dilation[j]
			coords[j] = val
			if val < 0 or val > out_spatial_shape[j] - 1:
				valid = False
			offset += mult * ((input_pos[j] - val * stride[j] + padding[j]) // dilation[j])
			mult *= kernel_size[j]
		if valid:
			valid_points.append((coords, offset))
	return valid_points


def _valid_out_pos_transpose(
	input_pos: list[int],
	kernel_size: list[int],
	stride: list[int],
	padding: list[int],
	dilation: list[int],
	out_spatial_shape: list[int],
) -> list[tuple[list[int], int]]:
	ndim = len(input_pos)
	lowers = []
	uppers = []
	for i in range(ndim):
		lower = input_pos[i] * stride[i] - padding[i]
		upper = lower + (kernel_size[i] - 1) * dilation[i]
		lowers.append(lower)
		uppers.append(upper)

	counter_sizes = []
	for i in range(ndim):
		counter_sizes.append((uppers[i] - lowers[i]) // dilation[i] + 1)

	valid_points: list[tuple[list[int], int]] = []
	for counter in product(*[range(size) for size in counter_sizes]):
		coords = [0] * ndim
		offset = 0
		mult = 1
		valid = True
		for j in range(ndim - 1, -1, -1):
			val = uppers[j] - counter[j] * dilation[j]
			coords[j] = val
			if val < 0 or val > out_spatial_shape[j] - 1:
				valid = False
			offset += mult * ((val - lowers[j]) // dilation[j])
			mult *= kernel_size[j]
		if valid:
			valid_points.append((coords, offset))
	return valid_points


def _prepare_indice_pair_args(
	indices: torch.Tensor,
	batch_size: int,
	spatial_shape,
	ksize=3,
	stride=1,
	padding=0,
	dilation=1,
	out_padding=0,
	subm: bool = False,
	transpose: bool = False,
):
	if batch_size <= 0:
		raise ValueError('batch_size must be positive')
	if indices.numel() > 0:
		batch_col = indices[:, 0]
		if int(batch_col.min().item()) < 0:
			raise ValueError('indices batch column must be non-negative')
		if int(batch_col.max().item()) >= batch_size:
			raise ValueError('indices batch column exceeds batch_size')

	ndim = indices.shape[1] - 1
	ksize = _normalize_nd_param(ksize, ndim)
	stride = _normalize_nd_param(stride, ndim)
	padding = _normalize_nd_param(padding, ndim)
	dilation = _normalize_nd_param(dilation, ndim)
	out_padding = _normalize_nd_param(out_padding, ndim)
	spatial_shape = [int(v) for v in spatial_shape]

	for d, s in zip(dilation, stride):
		assert any([s == 1, d == 1]), "don't support this."

	if not subm:
		if transpose:
			out_spatial_shape = _get_deconv_output_size(
				spatial_shape, ksize, stride, padding, dilation, out_padding
			)
		else:
			out_spatial_shape = _get_conv_output_size(
				spatial_shape, ksize, stride, padding, dilation
			)
	else:
		out_spatial_shape = spatial_shape

	return ndim, ksize, stride, padding, dilation, out_padding, out_spatial_shape


def _get_indice_pairs_impl(
	indices: torch.Tensor,
	batch_size: int,
	out_spatial_shape: list[int],
	ksize: list[int],
	stride: list[int],
	padding: list[int],
	dilation: list[int],
	*,
	subm: bool,
	transpose: bool,
	cuda_ordering: bool,
	grid: torch.Tensor | None,
):
	ndim = indices.shape[1] - 1

	kernel_volume = math.prod(ksize)
	num_act_in = int(indices.shape[0])
	device = indices.device
	indice_pairs = torch.full(
		(kernel_volume, 2, num_act_in),
		-1,
		device=device,
		dtype=indices.dtype,
	)
	indice_num = torch.zeros((kernel_volume,), device=device, dtype=indices.dtype)
	if num_act_in == 0:
		if grid is not None:
			grid.fill_(-1)
		return indices.new_empty((0, ndim + 1)), indice_pairs, indice_num

	indices_cpu = indices.detach().to('cpu', dtype=torch.int64)
	outids_list: list[list[int]] = []
	grids_out: dict[int, int] = {}
	spatial_volume = math.prod(out_spatial_shape)
	pending_pairs: list[tuple[int, int, int]] = []

	if subm:
		for j in range(num_act_in):
			batch_idx = int(indices_cpu[j, 0].item())
			coord = [int(v) for v in indices_cpu[j, 1:].tolist()]
			index = _row_array_idx(coord, out_spatial_shape) + spatial_volume * batch_idx
			grids_out[index] = j
			outids_list.append([batch_idx, *coord])

	for j in range(num_act_in):
		batch_idx = int(indices_cpu[j, 0].item())
		input_pos = [int(v) for v in indices_cpu[j, 1:].tolist()]
		if transpose:
			valid_points = _valid_out_pos_transpose(
				input_pos, ksize, stride, padding, dilation, out_spatial_shape
			)
		else:
			valid_points = _valid_out_pos(
				input_pos, ksize, stride, padding, dilation, out_spatial_shape
			)

		for out_coord, offset in valid_points:
			index = _row_array_idx(out_coord, out_spatial_shape) + spatial_volume * batch_idx
			if subm:
				out_idx = grids_out.get(index, -1)
				if out_idx == -1:
					continue
				slot = int(indice_num[offset].item())
				indice_pairs[offset, 0, slot] = j
				indice_pairs[offset, 1, slot] = out_idx
				indice_num[offset] += 1
			else:
				pending_pairs.append((offset, j, index))

	if not subm:
		if cuda_ordering:
			unique_indices = sorted({index for _, _, index in pending_pairs})
		else:
			unique_indices = []
			seen = set()
			for _, _, index in pending_pairs:
				if index not in seen:
					seen.add(index)
					unique_indices.append(index)
		for out_idx, flat_index in enumerate(unique_indices):
			grids_out[flat_index] = out_idx
			batch_idx = flat_index // spatial_volume
			spatial_index = flat_index % spatial_volume
			outids_list.append([batch_idx, *_row_array_idx_inv(spatial_index, out_spatial_shape)])
		for offset, in_idx, flat_index in pending_pairs:
			slot = int(indice_num[offset].item())
			indice_pairs[offset, 0, slot] = in_idx
			indice_pairs[offset, 1, slot] = grids_out[flat_index]
			indice_num[offset] += 1

	if grid is not None:
		grid.fill_(-1)
	outids = torch.tensor(outids_list, device=device, dtype=indices.dtype)
	return outids, indice_pairs, indice_num


def get_indice_pairs_cpu_fallback(
	indices: torch.Tensor,
	batch_size: int,
	spatial_shape,
	ksize=3,
	stride=1,
	padding=0,
	dilation=1,
	out_padding=0,
	subm: bool = False,
	transpose: bool = False,
	grid: torch.Tensor | None = None,
):
	_, ksize, stride, padding, dilation, _, out_spatial_shape = _prepare_indice_pair_args(
		indices,
		batch_size,
		spatial_shape,
		ksize=ksize,
		stride=stride,
		padding=padding,
		dilation=dilation,
		out_padding=out_padding,
		subm=subm,
		transpose=transpose,
	)
	return _get_indice_pairs_impl(
		indices,
		batch_size,
		out_spatial_shape,
		ksize,
		stride,
		padding,
		dilation,
		subm=subm,
		transpose=transpose,
		cuda_ordering=False,
		grid=grid,
	)


def get_indice_pairs_triton(
	indices: torch.Tensor,
	batch_size: int,
	spatial_shape,
	ksize=3,
	stride=1,
	padding=0,
	dilation=1,
	out_padding=0,
	subm: bool = False,
	transpose: bool = False,
	grid: torch.Tensor | None = None,
):
	ndim, ksize, stride, padding, dilation, _, out_spatial_shape = _prepare_indice_pair_args(
		indices,
		batch_size,
		spatial_shape,
		ksize=ksize,
		stride=stride,
		padding=padding,
		dilation=dilation,
		out_padding=out_padding,
		subm=subm,
		transpose=transpose,
	)
	kernel_volume = math.prod(ksize)
	num_act_in = int(indices.shape[0])
	device = indices.device
	indice_pairs = torch.full(
		(kernel_volume, 2, num_act_in),
		-1,
		device=device,
		dtype=indices.dtype,
	)
	indice_num = torch.zeros((kernel_volume,), device=device, dtype=indices.dtype)
	if num_act_in == 0:
		if grid is not None:
			grid.fill_(-1)
		return indices.new_empty((0, ndim + 1)), indice_pairs, indice_num

	spatial_volume = math.prod(out_spatial_shape)
	total_slots = batch_size * spatial_volume
	nthreads = num_act_in * kernel_volume
	flat_out = torch.empty((nthreads,), device=device, dtype=torch.int64)
	valid = torch.empty((nthreads,), device=device, dtype=torch.int8)
	launch_grid = lambda meta: (triton.cdiv(nthreads, meta['BLOCK_SIZE']),)
	_generate_rule_candidates_kernel[launch_grid](
		indices.contiguous(),
		flat_out,
		valid,
		num_act_in,
		kernel_volume,
		stride[0],
		stride[1] if ndim >= 2 else 1,
		stride[2] if ndim >= 3 else 1,
		stride[3] if ndim >= 4 else 1,
		padding[0],
		padding[1] if ndim >= 2 else 0,
		padding[2] if ndim >= 3 else 0,
		padding[3] if ndim >= 4 else 0,
		dilation[0],
		dilation[1] if ndim >= 2 else 1,
		dilation[2] if ndim >= 3 else 1,
		dilation[3] if ndim >= 4 else 1,
		ksize[0],
		ksize[1] if ndim >= 2 else 1,
		ksize[2] if ndim >= 3 else 1,
		ksize[3] if ndim >= 4 else 1,
		out_spatial_shape[0],
		out_spatial_shape[1] if ndim >= 2 else 1,
		out_spatial_shape[2] if ndim >= 3 else 1,
		out_spatial_shape[3] if ndim >= 4 else 1,
		spatial_volume,
		ndim=ndim,
		transpose=transpose,
		BLOCK_SIZE=_BLOCK_SIZE,
	)

	valid_mask = valid.bool()
	valid_candidate_idx = _compact_marked_indices(valid_mask)
	valid_flat = flat_out[valid_mask]
	valid_offsets = valid_candidate_idx % kernel_volume
	valid_in = (valid_candidate_idx // kernel_volume).to(torch.int32)

	if subm:
		input_flat = _flatten_indices(indices, out_spatial_shape, spatial_volume)
		index_map, borrowed = _borrow_or_alloc_index_map(grid, total_slots, device)
		_index_map_op('write_identity', input_flat.to(torch.int64), index_map)
		out_idx = _index_map_op('lookup', valid_flat.to(torch.int64), index_map)
		match = out_idx >= 0
		valid_offsets = valid_offsets[match]
		valid_in = valid_in[match]
		out_idx = out_idx[match]
		_pack_indice_pairs(
			indice_pairs,
			indice_num,
			valid_offsets,
			valid_in,
			out_idx,
			kernel_volume,
		)
		if borrowed:
			index_map.fill_(-1)
		return indices, indice_pairs, indice_num

	index_map, borrowed = _borrow_or_alloc_index_map(grid, total_slots, device)
	_index_map_op('mark', valid_flat.to(torch.int64), index_map)
	unique_indices = _compact_marked_indices(index_map >= 0)
	outids = _decode_outids_from_unique(
		unique_indices, out_spatial_shape, spatial_volume, indices.dtype
	)
	_index_map_op('write_identity', unique_indices.to(torch.int64), index_map)
	out_idx = _index_map_op('lookup', valid_flat.to(torch.int64), index_map)
	_pack_indice_pairs(
		indice_pairs,
		indice_num,
		valid_offsets,
		valid_in,
		out_idx,
		kernel_volume,
	)
	if borrowed:
		index_map.fill_(-1)
	return outids, indice_pairs, indice_num


def get_indice_pairs(
	indices: torch.Tensor,
	batch_size: int,
	spatial_shape,
	ksize=3,
	stride=1,
	padding=0,
	dilation=1,
	out_padding=0,
	subm: bool = False,
	transpose: bool = False,
	grid: torch.Tensor | None = None,
):
	if indices.is_cuda:
		return get_indice_pairs_triton(
			indices,
			batch_size,
			spatial_shape,
			ksize=ksize,
			stride=stride,
			padding=padding,
			dilation=dilation,
			out_padding=out_padding,
			subm=subm,
			transpose=transpose,
			grid=grid,
		)
	return get_indice_pairs_cpu_fallback(
		indices,
		batch_size,
		spatial_shape,
		ksize=ksize,
		stride=stride,
		padding=padding,
		dilation=dilation,
		out_padding=out_padding,
		subm=subm,
		transpose=transpose,
		grid=grid,
	)


def _to_cpu_pair_counts(indice_pair_num: torch.Tensor) -> torch.Tensor:
	"""Return rule counts on CPU to avoid repeated CUDA-host sync during iteration."""
	if indice_pair_num.device.type == 'cpu':
		return indice_pair_num
	return indice_pair_num.detach().to('cpu')


def _iter_indice_pairs(indice_pairs: torch.Tensor, indice_pair_num: torch.Tensor, inverse: bool):
	"""Iterate active kernel rules without per-rule CUDA sync via .item()."""
	in_slot = 1 if inverse else 0
	out_slot = 0 if inverse else 1

	pair_num_cpu = _to_cpu_pair_counts(indice_pair_num)
	active_rule_ids = torch.nonzero(pair_num_cpu > 0, as_tuple=False).flatten().tolist()
	for rule_id in active_rule_ids:
		nnz = int(pair_num_cpu[rule_id])
		in_idx = indice_pairs[rule_id, in_slot, :nnz].contiguous()
		out_idx = indice_pairs[rule_id, out_slot, :nnz].contiguous()
		yield rule_id, in_idx, out_idx


def indice_conv(
	features: torch.Tensor,
	filters: torch.Tensor,
	indice_pairs: torch.Tensor,
	indice_pair_num: torch.Tensor,
	num_activate_out: int,
	inverse: bool = False,
	subm: bool = False,
) -> torch.Tensor:
	del subm

	in_channels = features.size(1)
	out_channels = filters.size(-1)
	filters_k = filters.view(-1, in_channels, out_channels).contiguous()

	out_features = torch.zeros(
		(num_activate_out, out_channels),
		device=features.device,
		dtype=features.dtype,
	)

	for rule_id, in_idx, out_idx in _iter_indice_pairs(indice_pairs, indice_pair_num, inverse):
		gathered_features = sparse_gather(features, in_idx)
		partial_out = gathered_features @ filters_k[rule_id]
		sparse_scatter_add(out_features, partial_out, out_idx)

	return out_features


def fused_indice_conv(
	features: torch.Tensor,
	filters: torch.Tensor,
	bias: torch.Tensor,
	indice_pairs: torch.Tensor,
	indice_pair_num: torch.Tensor,
	num_activate_out: int,
	inverse: bool,
	subm: bool,
) -> torch.Tensor:
	out_features = indice_conv(
		features,
		filters,
		indice_pairs,
		indice_pair_num,
		num_activate_out,
		inverse=inverse,
		subm=subm,
	)
	# Match legacy C++ behavior: subm fused path does not add bias.
	if subm:
		return out_features
	return out_features + bias


def indice_conv_backward(
	features: torch.Tensor,
	filters: torch.Tensor,
	out_bp: torch.Tensor,
	indice_pairs: torch.Tensor,
	indice_pair_num: torch.Tensor,
	inverse: bool = False,
	subm: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
	del subm

	in_channels = features.size(1)
	out_channels = filters.size(-1)
	filters_k = filters.view(-1, in_channels, out_channels).contiguous()
	filters_grad_k = torch.zeros_like(filters_k)
	input_grad = torch.zeros_like(features)

	for rule_id, in_idx, out_idx in _iter_indice_pairs(indice_pairs, indice_pair_num, inverse):
		in_features = sparse_gather(features, in_idx)
		out_grad = sparse_gather(out_bp, out_idx)

		filters_grad_k[rule_id] += in_features.transpose(0, 1) @ out_grad

		in_grad = out_grad @ filters_k[rule_id].transpose(0, 1)
		sparse_scatter_add(input_grad, in_grad, in_idx)

	return input_grad, filters_grad_k.view_as(filters)


def indice_maxpool(
	features: torch.Tensor,
	indice_pairs: torch.Tensor,
	indice_pair_num: torch.Tensor,
	num_activate_out: int,
) -> torch.Tensor:
	channels = features.size(1)
	out_features = torch.zeros(
		(num_activate_out, channels),
		dtype=features.dtype,
		device=features.device,
	)

	for _, in_idx, out_idx in _iter_indice_pairs(indice_pairs, indice_pair_num, inverse=False):
		in_features = sparse_gather(features, in_idx)
		scatter_idx = out_idx.to(torch.long).unsqueeze(1).expand(-1, channels)
		out_features.scatter_reduce_(0, scatter_idx, in_features, reduce='amax', include_self=True)

	return out_features


def indice_maxpool_backward(
	features: torch.Tensor,
	out_features: torch.Tensor,
	out_bp: torch.Tensor,
	indice_pairs: torch.Tensor,
	indice_pair_num: torch.Tensor,
) -> torch.Tensor:
	input_grad = torch.zeros_like(features)

	for _, in_idx, out_idx in _iter_indice_pairs(indice_pairs, indice_pair_num, inverse=False):
		in_features = sparse_gather(features, in_idx)
		pooled_features = sparse_gather(out_features, out_idx)
		out_grad = sparse_gather(out_bp, out_idx)

		# Keep identical behavior with the CUDA kernel: all equal maxima receive grad.
		in_grad = torch.where(in_features == pooled_features, out_grad, torch.zeros_like(out_grad))
		sparse_scatter_add(input_grad, in_grad, in_idx)

	return input_grad


__all__ = [
	'indice_conv',
	'fused_indice_conv',
	'indice_conv_backward',
	'indice_maxpool',
	'indice_maxpool_backward',
]

# TODO(spconv-cu130-align):
# Re-introduce parity tests against the pip package once semantic mapping between
# this repo's extracted legacy kernels and newer spconv-cu130 operators is defined.
