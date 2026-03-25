from __future__ import annotations

import pytest
import torch

from ops.iou3d import boxes_iou_bev, boxes_overlap_bev_gpu, nms_gpu, nms_normal_gpu

try:
	import ops.iou3d.iou3d_utils_legacy as cuda_mod

	boxes_iou_bev_cuda = cuda_mod.boxes_iou_bev
	nms_gpu_cuda = cuda_mod.nms_gpu
	nms_normal_gpu_cuda = cuda_mod.nms_normal_gpu
except Exception:  # pragma: no cover
	cuda_mod = None
	boxes_iou_bev_cuda = None
	nms_gpu_cuda = None
	nms_normal_gpu_cuda = None


def _build_iou_case_tensors(
	*,
	num_a: int,
	num_b: int,
	device: str,
	seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
	torch.manual_seed(seed)
	boxes_a = torch.rand((num_a, 5), device=device, dtype=torch.float32)
	boxes_b = torch.rand((num_b, 5), device=device, dtype=torch.float32)
	boxes_a[:, 2:4] = boxes_a[:, :2] + boxes_a[:, 2:4] + 0.1
	boxes_b[:, 2:4] = boxes_b[:, :2] + boxes_b[:, 2:4] + 0.1
	boxes_a[:, 4] = (boxes_a[:, 4] - 0.5) * 1.57
	boxes_b[:, 4] = (boxes_b[:, 4] - 0.5) * 1.57
	return boxes_a.contiguous(), boxes_b.contiguous()


def _build_nms_case_tensors(
	*,
	num_boxes: int,
	device: str,
	seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
	torch.manual_seed(seed)
	boxes = torch.rand((num_boxes, 5), device=device, dtype=torch.float32)
	boxes[:, 2:4] = boxes[:, :2] + boxes[:, 2:4] + 0.1
	boxes[:, 4] = (boxes[:, 4] - 0.5) * 1.57
	scores = torch.rand((num_boxes,), device=device, dtype=torch.float32)
	return boxes.contiguous(), scores.contiguous()


@pytest.mark.skipif(cuda_mod is None, reason='CUDA extension module import failed')
def test_00_iou3d_reference_extension_compile_success():
	assert hasattr(cuda_mod, 'ext_module')
	assert hasattr(cuda_mod.ext_module, 'boxes_overlap_bev_gpu')
	assert hasattr(cuda_mod.ext_module, 'boxes_iou_bev_gpu')
	assert hasattr(cuda_mod.ext_module, 'nms_gpu')
	assert hasattr(cuda_mod.ext_module, 'nms_normal_gpu')


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is required')
@pytest.mark.skipif(cuda_mod is None, reason='CUDA extension module import failed')
@pytest.mark.parametrize('num_a,num_b,seed', [(1, 1, 3), (4, 3, 4), (7, 5, 5)])
def test_iou3d_boxes_overlap_bev_gpu_matches_cuda_reference(
	num_a: int,
	num_b: int,
	seed: int,
):
	boxes_a, boxes_b = _build_iou_case_tensors(
		num_a=num_a,
		num_b=num_b,
		device='cuda',
		seed=seed,
	)
	out_triton = torch.empty((num_a, num_b), device='cuda', dtype=torch.float32)
	out_cuda = torch.empty((num_a, num_b), device='cuda', dtype=torch.float32)
	ret_triton = boxes_overlap_bev_gpu(boxes_a, boxes_b, out_triton)
	ret_cuda = cuda_mod.ext_module.boxes_overlap_bev_gpu(boxes_a, boxes_b, out_cuda)
	assert ret_triton == 1
	assert ret_cuda == 1
	assert torch.allclose(out_triton, out_cuda, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is required')
@pytest.mark.skipif(boxes_iou_bev_cuda is None, reason='CUDA extension module import failed')
@pytest.mark.parametrize('num_a,num_b,seed', [(1, 1, 0), (4, 3, 1), (7, 5, 2)])
def test_iou3d_boxes_iou_bev_matches_cuda_reference(
	num_a: int,
	num_b: int,
	seed: int,
):
	boxes_a, boxes_b = _build_iou_case_tensors(
		num_a=num_a,
		num_b=num_b,
		device='cuda',
		seed=seed,
	)
	out_triton = boxes_iou_bev(boxes_a, boxes_b)
	out_cuda = boxes_iou_bev_cuda(boxes_a, boxes_b)
	assert torch.allclose(out_triton, out_cuda, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is required')
@pytest.mark.skipif(nms_gpu_cuda is None, reason='CUDA extension module import failed')
@pytest.mark.parametrize(
	'thresh,pre_maxsize,post_max_size,seed',
	[
		(0.1, None, None, 0),
		(0.2, 5, None, 1),
		(0.3, 6, 3, 2),
	],
)
def test_iou3d_nms_gpu_matches_cuda_reference(
	thresh: float,
	pre_maxsize: int | None,
	post_max_size: int | None,
	seed: int,
):
	boxes, scores = _build_nms_case_tensors(num_boxes=8, device='cuda', seed=seed)
	keep_triton = nms_gpu(
		boxes,
		scores,
		thresh,
		pre_maxsize=pre_maxsize,
		post_max_size=post_max_size,
	)
	keep_cuda = nms_gpu_cuda(
		boxes,
		scores,
		thresh,
		pre_maxsize=pre_maxsize,
		post_max_size=post_max_size,
	)
	assert torch.equal(keep_triton, keep_cuda)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is required')
@pytest.mark.skipif(nms_normal_gpu_cuda is None, reason='CUDA extension module import failed')
@pytest.mark.parametrize('thresh,seed', [(0.1, 0), (0.2, 1), (0.3, 2)])
def test_iou3d_nms_normal_gpu_matches_cuda_reference(
	thresh: float,
	seed: int,
):
	boxes, scores = _build_nms_case_tensors(num_boxes=8, device='cuda', seed=seed)
	keep_triton = nms_normal_gpu(boxes, scores, thresh)
	keep_cuda = nms_normal_gpu_cuda(boxes, scores, thresh)
	assert torch.equal(keep_triton, keep_cuda)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is required')
def test_iou3d_rejects_non_float32_inputs():
	boxes = torch.rand((4, 5), device='cuda', dtype=torch.float16).contiguous()
	scores = torch.rand((4,), device='cuda', dtype=torch.float16).contiguous()
	with pytest.raises(TypeError):
		boxes_iou_bev(boxes, boxes)
	with pytest.raises(TypeError):
		nms_gpu(boxes, scores, 0.1)
	with pytest.raises(TypeError):
		nms_normal_gpu(boxes, scores, 0.1)
