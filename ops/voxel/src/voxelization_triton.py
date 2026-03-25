from __future__ import annotations

from typing import Sequence

import torch
import triton
import triton.language as tl

from .coord_key_triton import decode_keys_to_coors_kernel, encode_coors_to_keys_kernel


_KEY_MAP_MODE = {
	'mark': 0,
	'write_identity': 1,
	'lookup': 2,
}
_GROUP_STATS_MODE = {
	'count_only': False,
	'count_and_first': True,
}


def _voxel_meta(
	voxel_size: Sequence[float],
	coors_range: Sequence[float],
	ndim: int = 3,
) -> tuple[list[float], list[float], list[int]]:
	voxel_size = [float(v) for v in voxel_size]
	coors_range = [float(v) for v in coors_range]
	if ndim != 3:
		raise NotImplementedError('Only 3D voxelization is supported')
	if len(voxel_size) != ndim:
		raise ValueError(f'voxel_size must have length {ndim}')
	if len(coors_range) != ndim * 2:
		raise ValueError(f'coors_range must have length {ndim * 2}')
	grid_size = [
		int(round((coors_range[3 + i] - coors_range[i]) / voxel_size[i])) for i in range(3)
	]
	return voxel_size, coors_range, grid_size


@triton.jit
def _dynamic_voxelize_kernel(
	points_ptr,
	coors_ptr,
	num_points,
	num_features,
	voxel_x,
	voxel_y,
	voxel_z,
	range_x_min,
	range_y_min,
	range_z_min,
	grid_x,
	grid_y,
	grid_z,
):
	index = tl.program_id(axis=0)
	if index >= num_points:
		return

	base = index * num_features
	x = tl.load(points_ptr + base + 0).to(tl.float32)
	y = tl.load(points_ptr + base + 1).to(tl.float32)
	z = tl.load(points_ptr + base + 2).to(tl.float32)

	coor_x = tl.floor((x - range_x_min) / voxel_x).to(tl.int32)
	out_base = index * 3
	if (coor_x < 0) or (coor_x >= grid_x):
		tl.store(coors_ptr + out_base + 0, -1)
		return

	coor_y = tl.floor((y - range_y_min) / voxel_y).to(tl.int32)
	if (coor_y < 0) or (coor_y >= grid_y):
		tl.store(coors_ptr + out_base + 0, -1)
		tl.store(coors_ptr + out_base + 1, -1)
		return

	coor_z = tl.floor((z - range_z_min) / voxel_z).to(tl.int32)
	if (coor_z < 0) or (coor_z >= grid_z):
		tl.store(coors_ptr + out_base + 0, -1)
		tl.store(coors_ptr + out_base + 1, -1)
		tl.store(coors_ptr + out_base + 2, -1)
		return

	tl.store(coors_ptr + out_base + 0, coor_x)
	tl.store(coors_ptr + out_base + 1, coor_y)
	tl.store(coors_ptr + out_base + 2, coor_z)


@triton.jit
def _point_to_voxelidx_kernel(
	coor_ptr,
	point_to_voxelidx_ptr,
	point_to_pointidx_ptr,
	max_points,
	num_points,
	ndim: tl.constexpr,
):
	index = tl.program_id(axis=0)
	if index >= num_points:
		return

	base = index * ndim
	coor_x = tl.load(coor_ptr + base + 0)
	if coor_x != -1:
		coor_y = tl.load(coor_ptr + base + 1)
		coor_z = tl.load(coor_ptr + base + 2)
		num = 0
		point_idx = index
		overflow = 0
		i = 0
		while i < index:
			prev_base = i * ndim
			prev_x = tl.load(coor_ptr + prev_base + 0)
			if prev_x != -1:
				prev_y = tl.load(coor_ptr + prev_base + 1)
				prev_z = tl.load(coor_ptr + prev_base + 2)
				if (prev_x == coor_x) and (prev_y == coor_y) and (prev_z == coor_z):
					num += 1
					if num == 1:
						point_idx = i
					if num >= max_points:
						overflow = 1
			i += 1
		if overflow == 0:
			if num == 0:
				tl.store(point_to_pointidx_ptr + index, index)
			else:
				tl.store(point_to_pointidx_ptr + index, point_idx)
			tl.store(point_to_voxelidx_ptr + index, num)


@triton.jit
def _determin_voxel_num_kernel(
	num_points_per_voxel_ptr,
	point_to_voxelidx_ptr,
	point_to_pointidx_ptr,
	coor_to_voxelidx_ptr,
	voxel_num_ptr,
	max_voxels,
	num_points,
):
	if tl.program_id(axis=0) != 0:
		return
	i = 0
	while i < num_points:
		point_pos = tl.load(point_to_voxelidx_ptr + i)
		if point_pos != -1:
			if point_pos == 0:
				voxelidx = tl.load(voxel_num_ptr + 0)
				if voxelidx < max_voxels:
					tl.store(voxel_num_ptr + 0, voxelidx + 1)
					tl.store(coor_to_voxelidx_ptr + i, voxelidx)
					tl.store(num_points_per_voxel_ptr + voxelidx, 1)
			else:
				point_idx = tl.load(point_to_pointidx_ptr + i)
				voxelidx = tl.load(coor_to_voxelidx_ptr + point_idx)
				if voxelidx != -1:
					tl.store(coor_to_voxelidx_ptr + i, voxelidx)
					count = tl.load(num_points_per_voxel_ptr + voxelidx)
					tl.store(num_points_per_voxel_ptr + voxelidx, count + 1)
		i += 1


@triton.jit
def _assign_point_to_voxel_kernel(
	points_ptr,
	point_to_voxelidx_ptr,
	coor_to_voxelidx_ptr,
	voxels_ptr,
	max_points,
	num_features,
	nthreads,
):
	tid = tl.program_id(axis=0)
	if tid >= nthreads:
		return
	index = tid // num_features
	num = tl.load(point_to_voxelidx_ptr + index)
	voxelidx = tl.load(coor_to_voxelidx_ptr + index)
	if (num > -1) and (voxelidx > -1):
		k = tid % num_features
		value = tl.load(points_ptr + tid)
		out_offset = (voxelidx * max_points + num) * num_features + k
		tl.store(voxels_ptr + out_offset, value)


@triton.jit
def _assign_voxel_coors_kernel(
	coor_ptr,
	point_to_voxelidx_ptr,
	coor_to_voxelidx_ptr,
	voxel_coors_ptr,
	num_points,
	ndim: tl.constexpr,
	nthreads,
):
	tid = tl.program_id(axis=0)
	if tid >= nthreads:
		return
	index = tid // ndim
	num = tl.load(point_to_voxelidx_ptr + index)
	voxelidx = tl.load(coor_to_voxelidx_ptr + index)
	if (num == 0) and (voxelidx > -1):
		k = tid % ndim
		value = tl.load(coor_ptr + tid)
		tl.store(voxel_coors_ptr + voxelidx * ndim + k, value)


@triton.jit
def _nondeterministic_get_assign_pos_kernel(
	coors_map_ptr,
	pts_id_ptr,
	reduce_count_ptr,
	nthreads,
):
	idx = tl.program_id(axis=0)
	if idx >= nthreads:
		return
	coors_idx = tl.load(coors_map_ptr + idx)
	if coors_idx > -1:
		coors_pts_pos = tl.atomic_add(reduce_count_ptr + coors_idx, 1)
		tl.store(pts_id_ptr + idx, coors_pts_pos)


@triton.jit
def _nondeterministic_assign_point_voxel_kernel(
	points_ptr,
	coors_map_ptr,
	pts_id_ptr,
	reduce_count_ptr,
	voxels_ptr,
	pts_count_ptr,
	max_voxels,
	max_points,
	num_features,
	nthreads,
):
	idx = tl.program_id(axis=0)
	if idx >= nthreads:
		return
	coors_idx = tl.load(coors_map_ptr + idx)
	coors_pts_pos = tl.load(pts_id_ptr + idx)
	if coors_idx > -1:
		coors_pos = coors_idx
		if (coors_pos < max_voxels) and (coors_pts_pos < max_points):
			points_offset = idx * num_features
			voxels_offset = (coors_pos * max_points + coors_pts_pos) * num_features
			k = 0
			while k < num_features:
				value = tl.load(points_ptr + points_offset + k)
				tl.store(voxels_ptr + voxels_offset + k, value)
				k += 1
			if coors_pts_pos == 0:
				count = tl.load(reduce_count_ptr + coors_idx)
				tl.store(pts_count_ptr + coors_pos, tl.minimum(count, max_points))


@triton.jit
def _group_stats_kernel(
	inverse_ptr,
	first_pos_ptr,
	counts_ptr,
	num_valid,
	track_first: tl.constexpr,
):
	pid = tl.program_id(axis=0)
	if pid >= num_valid:
		return
	group_id = tl.load(inverse_ptr + pid).to(tl.int32)
	if track_first:
		tl.atomic_min(first_pos_ptr + group_id, pid)
	tl.atomic_add(counts_ptr + group_id, 1)


@triton.jit
def _scatter_selected_points_kernel(
	points_ptr,
	inverse_ptr,
	remap_ptr,
	fill_count_ptr,
	voxels_ptr,
	num_valid,
	num_features,
	max_points,
):
	pid = tl.program_id(axis=0)
	if pid >= num_valid:
		return
	group_id = tl.load(inverse_ptr + pid).to(tl.int32)
	voxel_idx = tl.load(remap_ptr + group_id).to(tl.int32)
	if voxel_idx < 0:
		return
	slot = tl.atomic_add(fill_count_ptr + voxel_idx, 1)
	if slot >= max_points:
		return
	point_base = pid * num_features
	voxel_base = (voxel_idx * max_points + slot) * num_features
	k = 0
	while k < num_features:
		value = tl.load(points_ptr + point_base + k)
		tl.store(voxels_ptr + voxel_base + k, value)
		k += 1


@triton.jit
def _key_map_kernel(
	keys_ptr,
	key_map_ptr,
	out_ptr,
	num_keys,
	mode: tl.constexpr,
):
	pid = tl.program_id(axis=0)
	if pid >= num_keys:
		return
	key = tl.load(keys_ptr + pid).to(tl.int64)
	if mode == 0:
		tl.store(key_map_ptr + key, 1)
	elif mode == 1:
		tl.store(key_map_ptr + key, pid.to(tl.int32))
	else:
		value = tl.load(key_map_ptr + key)
		tl.store(out_ptr + pid, value)


def _dynamic_voxelize_cpu_fallback(
	points: torch.Tensor,
	coors: torch.Tensor,
	voxel_size: Sequence[float],
	coors_range: Sequence[float],
	NDim: int,
) -> None:
	if not points.device.type == 'cpu':
		raise AssertionError('points must be a CPU tensor')
	voxel_size, coors_range, grid_size = _voxel_meta(voxel_size, coors_range, NDim)
	num_points = int(points.size(0))
	coors.fill_(-1)
	for i in range(num_points):
		failed = False
		for j in range(NDim):
			c = int(torch.floor((points[i, j] - coors_range[j]) / voxel_size[j]).item())
			if c < 0 or c >= grid_size[j]:
				failed = True
				break
			coors[i, j] = c
		if failed:
			coors[i].fill_(-1)


def _unique_valid_coors_triton(
	coors: torch.Tensor,
	grid_size: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
	size_vec = [int(v) for v in grid_size]
	keys = torch.empty((coors.size(0),), device=coors.device, dtype=torch.int64)
	encode_coors_to_keys_kernel[(int(coors.size(0)),)](
		coors.contiguous(),
		keys,
		int(coors.size(0)),
		int(coors.size(1)),
		size_vec[0] if len(size_vec) >= 1 else 1,
		size_vec[1] if len(size_vec) >= 2 else 1,
		size_vec[2] if len(size_vec) >= 3 else 1,
		size_vec[3] if len(size_vec) >= 4 else 1,
	)
	inverse = torch.full((coors.size(0),), -1, device=coors.device, dtype=torch.int64)
	valid_mask = keys >= 0
	valid_keys = keys[valid_mask]
	total_slots = 1
	for size in size_vec[: int(coors.size(1))]:
		total_slots *= int(size)
	if valid_keys.numel() == 0:
		return coors.new_empty((0, coors.size(1))), inverse, keys.new_empty((0,))
	key_map = torch.full((total_slots,), -1, device=coors.device, dtype=torch.int32)
	_key_map_kernel[(int(valid_keys.numel()),)](
		valid_keys.contiguous(),
		key_map,
		torch.empty((0,), device=coors.device, dtype=torch.int32),
		int(valid_keys.numel()),
		mode=_KEY_MAP_MODE['mark'],
	)
	unique_keys = torch.nonzero(key_map >= 0, as_tuple=False).flatten().to(torch.int64)
	_key_map_kernel[(int(unique_keys.numel()),)](
		unique_keys.contiguous(),
		key_map,
		torch.empty((0,), device=coors.device, dtype=torch.int32),
		int(unique_keys.numel()),
		mode=_KEY_MAP_MODE['write_identity'],
	)
	inverse_valid = torch.empty((int(valid_keys.numel()),), device=coors.device, dtype=torch.int32)
	_key_map_kernel[(int(valid_keys.numel()),)](
		valid_keys.contiguous(),
		key_map,
		inverse_valid,
		int(valid_keys.numel()),
		mode=_KEY_MAP_MODE['lookup'],
	)
	inverse[valid_mask] = inverse_valid.to(torch.int64)
	counts = torch.zeros((int(unique_keys.numel()),), device=coors.device, dtype=torch.int32)
	_group_stats_kernel[(int(inverse_valid.numel()),)](
		inverse_valid.contiguous(),
		torch.empty((0,), device=coors.device, dtype=torch.int32),
		counts,
		int(inverse_valid.numel()),
		track_first=_GROUP_STATS_MODE['count_only'],
	)
	unique_coors = torch.empty(
		(unique_keys.size(0), coors.size(1)),
		device=coors.device,
		dtype=torch.int32,
	)
	if unique_keys.numel() > 0:
		decode_keys_to_coors_kernel[(int(unique_keys.size(0)),)](
			unique_keys.contiguous(),
			unique_coors,
			int(unique_keys.size(0)),
			int(coors.size(1)),
			size_vec[0] if len(size_vec) >= 1 else 1,
			size_vec[1] if len(size_vec) >= 2 else 1,
			size_vec[2] if len(size_vec) >= 3 else 1,
			size_vec[3] if len(size_vec) >= 4 else 1,
		)
	return unique_coors, inverse, counts


def dynamic_voxelize(
	points: torch.Tensor,
	coors: torch.Tensor,
	voxel_size: Sequence[float],
	coors_range: Sequence[float],
	NDim: int = 3,
) -> None:
	voxel_size, coors_range, grid_size = _voxel_meta(voxel_size, coors_range, NDim)
	if points.device.type == 'cpu':
		_dynamic_voxelize_cpu_fallback(points, coors, voxel_size, coors_range, NDim)
		return

	num_points = int(points.size(0))
	if num_points == 0:
		return

	points = points.contiguous()
	orig_coors = coors
	coors = coors.contiguous()
	_dynamic_voxelize_kernel[(num_points,)](
		points,
		coors,
		num_points,
		int(points.size(1)),
		voxel_size[0],
		voxel_size[1],
		voxel_size[2],
		coors_range[0],
		coors_range[1],
		coors_range[2],
		grid_size[0],
		grid_size[1],
		grid_size[2],
	)
	if coors.data_ptr() != orig_coors.data_ptr():
		orig_coors.copy_(coors)


def _hard_voxelize_cpu_fallback(
	points: torch.Tensor,
	voxels: torch.Tensor,
	coors: torch.Tensor,
	num_points_per_voxel: torch.Tensor,
	voxel_size: Sequence[float],
	coors_range: Sequence[float],
	max_points: int,
	max_voxels: int,
	NDim: int,
) -> int:
	if not points.device.type == 'cpu':
		raise AssertionError('points must be a CPU tensor')
	voxel_size, coors_range, _ = _voxel_meta(voxel_size, coors_range, NDim)

	temp_coors = torch.empty((points.size(0), NDim), dtype=torch.int32, device=points.device)
	_dynamic_voxelize_cpu_fallback(points, temp_coors, voxel_size, coors_range, NDim)

	voxels.zero_()
	num_points_per_voxel.zero_()
	coors.zero_()

	coor_to_voxelidx: dict[tuple[int, int, int], int] = {}
	voxel_num = 0
	for i in range(int(points.size(0))):
		coor_row = temp_coors[i]
		if int(coor_row[0].item()) == -1:
			continue
		key = (
			int(coor_row[0].item()),
			int(coor_row[1].item()),
			int(coor_row[2].item()),
		)
		voxelidx = coor_to_voxelidx.get(key, -1)
		if voxelidx == -1:
			voxelidx = voxel_num
			if max_voxels != -1 and voxel_num >= max_voxels:
				continue
			voxel_num += 1
			coor_to_voxelidx[key] = voxelidx
			for k in range(NDim):
				coors[voxelidx, k] = coor_row[k]
		num = int(num_points_per_voxel[voxelidx].item())
		if max_points == -1 or num < max_points:
			voxels[voxelidx, num] = points[i]
			num_points_per_voxel[voxelidx] += 1

	return voxel_num


def _hard_voxelize_gpu_nondeterministic(
	points: torch.Tensor,
	voxels: torch.Tensor,
	coors: torch.Tensor,
	num_points_per_voxel: torch.Tensor,
	voxel_size: Sequence[float],
	coors_range: Sequence[float],
	max_points: int,
	max_voxels: int,
	NDim: int,
) -> int:
	voxel_size, coors_range, grid_size = _voxel_meta(voxel_size, coors_range, NDim)
	num_points = int(points.size(0))
	if num_points == 0:
		voxels.zero_()
		coors.zero_()
		num_points_per_voxel.zero_()
		return 0

	temp_coors = torch.empty((num_points, 3), dtype=torch.int32, device=points.device)
	dynamic_voxelize(points, temp_coors, voxel_size, coors_range, 3)
	unique_coors, coors_map, reduce_count = _unique_valid_coors_triton(temp_coors, grid_size)

	num_coors = int(unique_coors.size(0))
	voxel_num = num_coors if max_voxels == -1 else min(num_coors, max_voxels)
	voxels.zero_()
	coors.zero_()
	num_points_per_voxel.zero_()
	if voxel_num == 0:
		return 0

	coors[:voxel_num] = unique_coors[:voxel_num].to(coors.dtype)
	valid_mask = coors_map >= 0
	if not valid_mask.any():
		return voxel_num

	reduce_count_runtime = coors_map.new_zeros(num_coors)
	pts_id = coors_map.new_zeros(num_points)
	_nondeterministic_get_assign_pos_kernel[(num_points,)](
		coors_map.contiguous(),
		pts_id,
		reduce_count_runtime,
		num_points,
	)
	_nondeterministic_assign_point_voxel_kernel[(num_points,)](
		points.contiguous(),
		coors_map.contiguous(),
		pts_id,
		reduce_count_runtime,
		voxels,
		num_points_per_voxel,
		voxel_num,
		max_points,
		int(points.size(1)),
		nthreads=num_points,
	)
	return voxel_num


def _hard_voxelize_gpu(
	points: torch.Tensor,
	voxels: torch.Tensor,
	coors: torch.Tensor,
	num_points_per_voxel: torch.Tensor,
	voxel_size: Sequence[float],
	coors_range: Sequence[float],
	max_points: int,
	max_voxels: int,
	NDim: int,
	deterministic: bool,
) -> int:
	voxel_size, coors_range, grid_size = _voxel_meta(voxel_size, coors_range, NDim)
	if not deterministic:
		return _hard_voxelize_gpu_nondeterministic(
			points,
			voxels,
			coors,
			num_points_per_voxel,
			voxel_size,
			coors_range,
			max_points,
			max_voxels,
			NDim,
		)
	num_points = int(points.size(0))
	if num_points == 0:
		voxels.zero_()
		coors.zero_()
		num_points_per_voxel.zero_()
		return 0

	temp_coors = torch.empty((num_points, NDim), dtype=torch.int32, device=points.device)
	dynamic_voxelize(points, temp_coors, voxel_size, coors_range, NDim)

	valid_mask = temp_coors[:, 0] != -1
	valid_coors = temp_coors[valid_mask]
	if valid_coors.numel() == 0:
		voxels.zero_()
		coors.zero_()
		num_points_per_voxel.zero_()
		return 0

	voxels.zero_()
	coors.zero_()
	num_points_per_voxel.zero_()
	valid_points = points[valid_mask]
	unique_coors, inverse, counts = _unique_valid_coors_triton(valid_coors, grid_size)
	num_unique = int(unique_coors.size(0))
	if num_unique == 0:
		return 0

	num_valid = int(inverse.numel())
	first_pos = torch.full(
		(num_unique,),
		num_valid,
		device=points.device,
		dtype=torch.int32,
	)
	first_pos_accum = torch.zeros((num_unique,), device=points.device, dtype=torch.int32)
	_group_stats_kernel[(num_valid,)](
		inverse.contiguous(),
		first_pos,
		first_pos_accum,
		num_valid,
		track_first=_GROUP_STATS_MODE['count_and_first'],
	)
	ordered_group_ids = torch.argsort(first_pos.to(torch.int64), stable=True)

	voxel_num = num_unique if max_voxels == -1 else min(num_unique, max_voxels)
	if voxel_num == 0:
		return 0

	selected_group_ids = ordered_group_ids[:voxel_num]
	remap = torch.full((num_unique,), -1, device=points.device, dtype=torch.int64)
	remap[selected_group_ids] = torch.arange(voxel_num, device=points.device, dtype=torch.int64)

	effective_max_points = num_points if max_points == -1 else max_points
	fill_count = torch.zeros((voxel_num,), device=points.device, dtype=torch.int32)
	_scatter_selected_points_kernel[(num_valid,)](
		valid_points.contiguous(),
		inverse.contiguous(),
		remap,
		fill_count,
		voxels,
		num_valid,
		int(valid_points.size(1)),
		effective_max_points,
	)

	selected_counts = counts[selected_group_ids]
	if max_points != -1:
		selected_counts = torch.clamp(selected_counts, max=max_points)
	num_points_per_voxel[:voxel_num] = selected_counts.to(num_points_per_voxel.dtype)

	coors[:voxel_num] = unique_coors[selected_group_ids].to(coors.dtype)
	return voxel_num


def hard_voxelize(
	points: torch.Tensor,
	voxels: torch.Tensor,
	coors: torch.Tensor,
	num_points_per_voxel: torch.Tensor,
	voxel_size: Sequence[float],
	coors_range: Sequence[float],
	max_points: int,
	max_voxels: int,
	NDim: int = 3,
	deterministic: bool = True,
) -> int:
	if points.device.type == 'cpu':
		return _hard_voxelize_cpu_fallback(
			points,
			voxels,
			coors,
			num_points_per_voxel,
			voxel_size,
			coors_range,
			max_points,
			max_voxels,
			NDim,
		)

	return _hard_voxelize_gpu(
		points,
		voxels,
		coors,
		num_points_per_voxel,
		voxel_size,
		coors_range,
		max_points,
		max_voxels,
		NDim,
		deterministic,
	)


__all__ = [
	'dynamic_voxelize',
	'hard_voxelize',
]
