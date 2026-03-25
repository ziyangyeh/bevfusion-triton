from __future__ import annotations

import pytest
import torch

from ops.roiaware_pool3d import (
	RoIAwarePool3d,
	points_in_boxes_batch,
	points_in_boxes_cpu,
	points_in_boxes_gpu,
)

try:
	import ops.roiaware_pool3d.points_in_boxes_legacy as ref_pib
	import ops.roiaware_pool3d.roiaware_pool3d_legacy as ref_pool

	ref_ext = ref_pib._ext_module
except Exception:  # pragma: no cover
	ref_pib = None
	ref_pool = None
	ref_ext = None


def _build_points_boxes_case(
	*,
	seed: int,
	batch_size: int,
	num_points: int,
	num_boxes: int,
	device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
	torch.manual_seed(seed)
	boxes = torch.empty((batch_size, num_boxes, 7), device=device, dtype=torch.float32)
	boxes[..., 0:3] = torch.randn((batch_size, num_boxes, 3), device=device) * 3.0
	boxes[..., 3:6] = torch.rand((batch_size, num_boxes, 3), device=device) * 2.0 + 0.5
	boxes[..., 6] = (torch.rand((batch_size, num_boxes), device=device) - 0.5) * torch.pi
	points = torch.randn((batch_size, num_points, 3), device=device, dtype=torch.float32) * 4.0
	return points.contiguous(), boxes.contiguous()


def _build_pool_case(
	*,
	seed: int,
	num_rois: int,
	num_points: int,
	num_channels: int,
	device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
	torch.manual_seed(seed)
	rois = torch.empty((num_rois, 7), device=device, dtype=torch.float32)
	rois[:, 0:3] = torch.randn((num_rois, 3), device=device) * 2.5
	rois[:, 3:6] = torch.rand((num_rois, 3), device=device) * 2.0 + 0.75
	rois[:, 6] = (torch.rand((num_rois,), device=device) - 0.5) * torch.pi
	pts = torch.randn((num_points, 3), device=device, dtype=torch.float32) * 3.5
	pts_feature = torch.randn((num_points, num_channels), device=device, dtype=torch.float32)
	return rois.contiguous(), pts.contiguous(), pts_feature.contiguous()


@pytest.mark.skipif(ref_ext is None, reason='Reference extension module import failed')
def test_00_roiaware_pool3d_reference_extension_compile_success():
	assert hasattr(ref_ext, 'forward')
	assert hasattr(ref_ext, 'backward')
	assert hasattr(ref_ext, 'points_in_boxes_gpu')
	assert hasattr(ref_ext, 'points_in_boxes_batch')
	assert hasattr(ref_ext, 'points_in_boxes_cpu')


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is required')
@pytest.mark.skipif(ref_pib is None, reason='Reference module import failed')
@pytest.mark.parametrize(
	'batch_size,num_points,num_boxes,seed',
	[(1, 8, 3, 0), (2, 16, 4, 1), (2, 23, 5, 2)],
)
def test_roiaware_points_in_boxes_gpu_matches_cuda_reference(
	batch_size: int,
	num_points: int,
	num_boxes: int,
	seed: int,
):
	points, boxes = _build_points_boxes_case(
		seed=seed,
		batch_size=batch_size,
		num_points=num_points,
		num_boxes=num_boxes,
		device='cuda',
	)
	out_triton = points_in_boxes_gpu(points, boxes)
	out_ref = ref_pib.points_in_boxes_gpu(points, boxes)
	assert torch.equal(out_triton, out_ref)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is required')
@pytest.mark.skipif(ref_pib is None, reason='Reference module import failed')
@pytest.mark.parametrize(
	'batch_size,num_points,num_boxes,seed',
	[(1, 7, 2, 3), (2, 15, 4, 4), (2, 24, 6, 5)],
)
def test_roiaware_points_in_boxes_batch_matches_cuda_reference(
	batch_size: int,
	num_points: int,
	num_boxes: int,
	seed: int,
):
	points, boxes = _build_points_boxes_case(
		seed=seed,
		batch_size=batch_size,
		num_points=num_points,
		num_boxes=num_boxes,
		device='cuda',
	)
	out_triton = points_in_boxes_batch(points, boxes)
	out_ref = ref_pib.points_in_boxes_batch(points, boxes)
	assert torch.equal(out_triton, out_ref)


def test_roiaware_points_in_boxes_cpu_known_case():
	points = torch.tensor(
		[
			[0.0, 0.0, 0.0],
			[3.0, 0.0, 0.0],
			[10.0, 0.0, 0.0],
		],
		dtype=torch.float32,
	)
	boxes = torch.tensor(
		[
			[0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0],
			[3.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0],
		],
		dtype=torch.float32,
	)
	expected = torch.tensor([[1, 0, 0], [0, 1, 0]], dtype=torch.int32)
	out = points_in_boxes_cpu(points, boxes)
	assert torch.equal(out, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is required')
@pytest.mark.skipif(ref_pool is None, reason='Reference module import failed')
@pytest.mark.parametrize('mode', ['max', 'avg'])
@pytest.mark.parametrize(
	'out_size,max_pts_per_voxel,num_rois,num_points,num_channels,seed',
	[
		((2, 2, 2), 8, 2, 18, 3, 0),
		((3, 2, 2), 10, 3, 21, 4, 1),
	],
)
def test_roiaware_pool3d_matches_cuda_reference(
	mode: str,
	out_size: tuple[int, int, int],
	max_pts_per_voxel: int,
	num_rois: int,
	num_points: int,
	num_channels: int,
	seed: int,
):
	rois, pts, pts_feature = _build_pool_case(
		seed=seed,
		num_rois=num_rois,
		num_points=num_points,
		num_channels=num_channels,
		device='cuda',
	)
	pts_feature_triton = pts_feature.clone().requires_grad_(True)
	pts_feature_ref = pts_feature.clone().requires_grad_(True)

	mod_triton = RoIAwarePool3d(
		out_size=out_size,
		max_pts_per_voxel=max_pts_per_voxel,
		mode=mode,
	)
	mod_ref = ref_pool.RoIAwarePool3d(
		out_size=out_size,
		max_pts_per_voxel=max_pts_per_voxel,
		mode=mode,
	)

	out_triton = mod_triton(rois, pts, pts_feature_triton)
	out_ref = mod_ref(rois, pts, pts_feature_ref)
	assert torch.allclose(out_triton, out_ref, rtol=1e-6, atol=1e-6)

	(out_triton.sum()).backward()
	(out_ref.sum()).backward()
	assert torch.allclose(pts_feature_triton.grad, pts_feature_ref.grad, rtol=1e-6, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is required')
def test_roiaware_pool3d_rejects_non_float32_inputs():
	rois = torch.rand((2, 7), device='cuda', dtype=torch.float16)
	pts = torch.rand((8, 3), device='cuda', dtype=torch.float16)
	pts_feature = torch.rand((8, 4), device='cuda', dtype=torch.float16)
	module = RoIAwarePool3d(out_size=(2, 2, 2), max_pts_per_voxel=8, mode='max')
	with pytest.raises(TypeError):
		module(rois, pts, pts_feature)
