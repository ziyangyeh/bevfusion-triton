from __future__ import annotations

import pytest
import torch

from ops.voxel.scatter_points import DynamicScatter, dynamic_scatter
from ops.voxel.voxel_layer import (
	dynamic_point_to_voxel_backward,
	dynamic_point_to_voxel_forward,
	dynamic_voxelize,
	hard_voxelize,
	voxel_layer_ext,
)
from ops.voxel.voxelize import Voxelization, voxelization
from tests.test_ops._eps import DTYPE_EPS

try:
	import ops.voxel.voxel_layer_legacy as legacy_mod
except Exception:  # pragma: no cover
	legacy_mod = None


VOXEL_SIZE = [1.0, 1.0, 1.0]
COORS_RANGE = [0.0, 0.0, 0.0, 4.0, 4.0, 4.0]


def _build_points(
	*,
	device: str,
	dtype: torch.dtype,
) -> torch.Tensor:
	return torch.tensor(
		[
			[0.2, 0.3, 0.1, 1.0],
			[0.7, 0.9, 0.8, 2.0],
			[1.2, 1.1, 1.0, 3.0],
			[1.9, 1.8, 1.7, 4.0],
			[2.2, 2.4, 2.1, 5.0],
			[3.9, 3.7, 3.8, 6.0],
			[-0.2, 0.1, 0.1, 7.0],
			[4.2, 0.1, 0.1, 8.0],
		],
		device=device,
		dtype=dtype,
	)


def _build_scatter_case(
	*,
	device: str,
	dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
	feats = torch.tensor(
		[
			[1.0, 2.0, -1.0],
			[3.0, 0.5, -2.0],
			[4.0, -1.0, 5.0],
			[2.0, 3.0, 0.0],
			[8.0, 1.0, -3.0],
			[6.0, 4.0, 7.0],
			[9.0, 9.0, 9.0],
		],
		device=device,
		dtype=dtype,
	)
	coors = torch.tensor(
		[
			[0, 0, 0],
			[0, 0, 0],
			[1, 1, 1],
			[1, 1, 1],
			[2, 2, 2],
			[2, 2, 2],
			[-1, -1, -1],
		],
		device=device,
		dtype=torch.int32,
	)
	return feats, coors


def _run_scatter_forward_backward(
	fn,
	feats: torch.Tensor,
	coors: torch.Tensor,
	reduce_type: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
	x = feats.clone().detach().requires_grad_(True)
	voxel_feats, voxel_coors = fn(x, coors, reduce_type)
	if voxel_feats.numel() == 0:
		loss = voxel_feats.sum()
	else:
		upstream = torch.linspace(
			0.1,
			1.0,
			voxel_feats.numel(),
			device=voxel_feats.device,
			dtype=voxel_feats.dtype,
		).view_as(voxel_feats)
		loss = (voxel_feats * upstream).sum()
	loss.backward()
	assert x.grad is not None
	return voxel_feats.detach(), voxel_coors.detach(), x.grad.detach()


def _run_scatter_low_level_forward_backward(
	forward_fn,
	backward_fn,
	feats: torch.Tensor,
	coors: torch.Tensor,
	reduce_type: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
	x = feats.clone().detach().requires_grad_(False)
	voxel_feats, voxel_coors, point2voxel_map, voxel_points_count = forward_fn(
		x, coors, reduce_type
	)
	grad_feats = torch.zeros_like(x)
	if voxel_feats.numel() == 0:
		grad_voxel_feats = torch.zeros_like(voxel_feats)
	else:
		grad_voxel_feats = torch.linspace(
			0.1,
			1.0,
			voxel_feats.numel(),
			device=voxel_feats.device,
			dtype=voxel_feats.dtype,
		).view_as(voxel_feats)
	backward_fn(
		grad_feats,
		grad_voxel_feats.contiguous(),
		x,
		voxel_feats,
		point2voxel_map,
		voxel_points_count,
		reduce_type,
	)
	return voxel_feats.detach(), voxel_coors.detach(), grad_feats.detach()


def _assert_close(a: torch.Tensor, b: torch.Tensor, dtype: torch.dtype):
	rtol, atol = DTYPE_EPS[dtype]
	assert torch.allclose(a, b, rtol=rtol, atol=atol)


@pytest.mark.skipif(legacy_mod is None, reason='Legacy extension module import failed')
def test_00_voxel_legacy_reference_extension_compile_success():
	assert hasattr(legacy_mod, 'voxel_layer_ext')
	assert hasattr(legacy_mod.voxel_layer_ext, 'hard_voxelize')
	assert hasattr(legacy_mod.voxel_layer_ext, 'dynamic_voxelize')
	assert hasattr(legacy_mod.voxel_layer_ext, 'dynamic_point_to_voxel_forward')
	assert hasattr(legacy_mod.voxel_layer_ext, 'dynamic_point_to_voxel_backward')
	assert hasattr(voxel_layer_ext, 'hard_voxelize')
	assert hasattr(voxel_layer_ext, 'dynamic_voxelize')
	assert hasattr(voxel_layer_ext, 'dynamic_point_to_voxel_forward')
	assert hasattr(voxel_layer_ext, 'dynamic_point_to_voxel_backward')


@pytest.mark.skipif(legacy_mod is None, reason='Legacy extension module import failed')
@pytest.mark.parametrize('dtype', [torch.float32, torch.float16])
def test_dynamic_voxelize_cpu_fallback_matches_legacy_reference(dtype: torch.dtype):
	points = _build_points(device='cpu', dtype=dtype)
	coors_tri = torch.zeros((points.size(0), 3), dtype=torch.int32)
	coors_ref = torch.zeros((points.size(0), 3), dtype=torch.int32)

	dynamic_voxelize(points, coors_tri, VOXEL_SIZE, COORS_RANGE, 3)
	legacy_mod.dynamic_voxelize(points, coors_ref, VOXEL_SIZE, COORS_RANGE, 3)

	assert torch.equal(coors_tri, coors_ref)


@pytest.mark.skipif(legacy_mod is None, reason='Legacy extension module import failed')
@pytest.mark.parametrize('dtype', [torch.float32, torch.float16])
@pytest.mark.parametrize('max_points,max_voxels,deterministic', [(2, 8, True), (3, 8, False)])
def test_hard_voxelize_cpu_fallback_matches_legacy_reference(
	dtype: torch.dtype,
	max_points: int,
	max_voxels: int,
	deterministic: bool,
):
	points = _build_points(device='cpu', dtype=dtype)
	voxels_tri = torch.zeros((max_voxels, max_points, points.size(1)), dtype=dtype)
	coors_tri = torch.zeros((max_voxels, 3), dtype=torch.int32)
	num_tri = torch.zeros((max_voxels,), dtype=torch.int32)

	voxels_ref = torch.zeros((max_voxels, max_points, points.size(1)), dtype=dtype)
	coors_ref = torch.zeros((max_voxels, 3), dtype=torch.int32)
	num_ref = torch.zeros((max_voxels,), dtype=torch.int32)

	voxel_num_tri = hard_voxelize(
		points,
		voxels_tri,
		coors_tri,
		num_tri,
		VOXEL_SIZE,
		COORS_RANGE,
		max_points,
		max_voxels,
		3,
		deterministic,
	)
	voxel_num_ref = legacy_mod.hard_voxelize(
		points,
		voxels_ref,
		coors_ref,
		num_ref,
		VOXEL_SIZE,
		COORS_RANGE,
		max_points,
		max_voxels,
		3,
		deterministic,
	)

	assert voxel_num_tri == voxel_num_ref
	assert torch.equal(coors_tri[:voxel_num_tri], coors_ref[:voxel_num_ref])
	assert torch.equal(num_tri[:voxel_num_tri], num_ref[:voxel_num_ref])
	_assert_close(voxels_tri[:voxel_num_tri].float(), voxels_ref[:voxel_num_ref].float(), dtype)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is required')
@pytest.mark.skipif(legacy_mod is None, reason='Legacy extension module import failed')
@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
def test_dynamic_voxelize_cuda_matches_legacy_reference(dtype: torch.dtype):
	points = _build_points(device='cuda', dtype=dtype)
	coors_tri = torch.zeros((points.size(0), 3), device='cuda', dtype=torch.int32)
	coors_ref = torch.zeros((points.size(0), 3), device='cuda', dtype=torch.int32)

	dynamic_voxelize(points, coors_tri, VOXEL_SIZE, COORS_RANGE, 3)
	legacy_mod.dynamic_voxelize(points, coors_ref, VOXEL_SIZE, COORS_RANGE, 3)

	assert torch.equal(coors_tri.cpu(), coors_ref.cpu())


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is required')
@pytest.mark.skipif(legacy_mod is None, reason='Legacy extension module import failed')
@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize('deterministic', [True, False])
def test_hard_voxelize_cuda_matches_legacy_reference(dtype: torch.dtype, deterministic: bool):
	points = _build_points(device='cuda', dtype=dtype)
	max_points = 3
	max_voxels = 8
	voxels_tri = torch.zeros((max_voxels, max_points, points.size(1)), device='cuda', dtype=dtype)
	coors_tri = torch.zeros((max_voxels, 3), device='cuda', dtype=torch.int32)
	num_tri = torch.zeros((max_voxels,), device='cuda', dtype=torch.int32)

	voxels_ref = torch.zeros((max_voxels, max_points, points.size(1)), device='cuda', dtype=dtype)
	coors_ref = torch.zeros((max_voxels, 3), device='cuda', dtype=torch.int32)
	num_ref = torch.zeros((max_voxels,), device='cuda', dtype=torch.int32)

	voxel_num_tri = hard_voxelize(
		points,
		voxels_tri,
		coors_tri,
		num_tri,
		VOXEL_SIZE,
		COORS_RANGE,
		max_points,
		max_voxels,
		3,
		deterministic,
	)
	voxel_num_ref = legacy_mod.hard_voxelize(
		points,
		voxels_ref,
		coors_ref,
		num_ref,
		VOXEL_SIZE,
		COORS_RANGE,
		max_points,
		max_voxels,
		3,
		deterministic,
	)

	assert voxel_num_tri == voxel_num_ref
	assert torch.equal(coors_tri[:voxel_num_tri].cpu(), coors_ref[:voxel_num_ref].cpu())
	assert torch.equal(num_tri[:voxel_num_tri].cpu(), num_ref[:voxel_num_ref].cpu())
	_assert_close(
		voxels_tri[:voxel_num_tri].float().cpu(),
		voxels_ref[:voxel_num_ref].float().cpu(),
		dtype,
	)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is required')
@pytest.mark.skipif(legacy_mod is None, reason='Legacy extension module import failed')
@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize('reduce_type', ['max', 'sum', 'mean'])
def test_dynamic_scatter_cpu_fallback_matches_legacy_reference(
	dtype: torch.dtype,
	reduce_type: str,
):
	feats_cpu, coors_cpu = _build_scatter_case(device='cpu', dtype=dtype)
	out_cpu, out_coors_cpu, grad_cpu = _run_scatter_forward_backward(
		dynamic_scatter,
		feats_cpu,
		coors_cpu,
		reduce_type,
	)

	feats_cuda = feats_cpu.cuda()
	coors_cuda = coors_cpu.cuda()
	out_ref, out_coors_ref, grad_ref = _run_scatter_low_level_forward_backward(
		legacy_mod.dynamic_point_to_voxel_forward,
		legacy_mod.dynamic_point_to_voxel_backward,
		feats_cuda,
		coors_cuda,
		reduce_type,
	)

	assert torch.equal(out_coors_cpu, out_coors_ref.cpu())
	_assert_close(out_cpu.float(), out_ref.float().cpu(), dtype)
	_assert_close(grad_cpu.float(), grad_ref.float().cpu(), dtype)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is required')
@pytest.mark.skipif(legacy_mod is None, reason='Legacy extension module import failed')
@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize('reduce_type', ['max', 'sum', 'mean'])
def test_dynamic_point_to_voxel_cuda_matches_legacy_reference(
	dtype: torch.dtype,
	reduce_type: str,
):
	feats, coors = _build_scatter_case(device='cuda', dtype=dtype)
	out_tri, out_coors_tri, grad_tri = _run_scatter_low_level_forward_backward(
		dynamic_point_to_voxel_forward,
		dynamic_point_to_voxel_backward,
		feats,
		coors,
		reduce_type,
	)
	out_ref, out_coors_ref, grad_ref = _run_scatter_low_level_forward_backward(
		legacy_mod.dynamic_point_to_voxel_forward,
		legacy_mod.dynamic_point_to_voxel_backward,
		feats,
		coors,
		reduce_type,
	)

	assert torch.equal(out_coors_tri.cpu(), out_coors_ref.cpu())
	_assert_close(out_tri.float().cpu(), out_ref.float().cpu(), dtype)
	_assert_close(grad_tri.float().cpu(), grad_ref.float().cpu(), dtype)


def test_voxelization_wrapper_matches_low_level_dynamic():
	points = _build_points(device='cpu', dtype=torch.float32)
	out_wrapper = voxelization(points, VOXEL_SIZE, COORS_RANGE, -1, -1, True)
	out_manual = torch.zeros((points.size(0), 3), dtype=torch.int32)
	dynamic_voxelize(points, out_manual, VOXEL_SIZE, COORS_RANGE, 3)
	assert torch.equal(out_wrapper, out_manual)


def test_voxelization_module_matches_function():
	points = _build_points(device='cpu', dtype=torch.float32)
	module = Voxelization(
		voxel_size=VOXEL_SIZE,
		point_cloud_range=COORS_RANGE,
		max_num_points=3,
		max_voxels=8,
		deterministic=True,
	)
	module.eval()
	voxels_mod, coors_mod, num_mod = module(points)
	voxels_fn, coors_fn, num_fn = voxelization(points, VOXEL_SIZE, COORS_RANGE, 3, 8, True)

	assert torch.equal(coors_mod, coors_fn)
	assert torch.equal(num_mod, num_fn)
	assert torch.equal(voxels_mod, voxels_fn)


def test_dynamic_scatter_module_matches_function():
	feats, coors = _build_scatter_case(device='cpu', dtype=torch.float32)
	module = DynamicScatter(VOXEL_SIZE, COORS_RANGE, average_points=True)
	out_mod, coors_mod = module(feats, coors)
	out_fn, coors_fn = dynamic_scatter(feats, coors, 'mean')
	assert torch.equal(coors_mod, coors_fn)
	assert torch.equal(out_mod, out_fn)
