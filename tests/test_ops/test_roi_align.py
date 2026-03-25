from __future__ import annotations

import pytest
import torch

from ops.roi_align import RoIAlign, roi_align
from tests.test_ops._eps import DTYPE_EPS

try:
	import ops.roi_align.roi_align_legacy as cuda_mod

	roi_align_cuda = cuda_mod.roi_align
except Exception:  # pragma: no cover
	cuda_mod = None
	roi_align_cuda = None


def _skip_if_bfloat16_unsupported():
	if not torch.cuda.is_available():
		pytest.skip('CUDA is required')
	if not torch.cuda.is_bf16_supported():
		pytest.skip('CUDA bfloat16 is required')


def _build_case_tensors(
	*,
	input_shape: tuple[int, int, int, int],
	device: str,
	dtype: torch.dtype,
	seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
	torch.manual_seed(seed)
	n, _, h, w = input_shape
	x = torch.randn(input_shape, device=device, dtype=dtype)

	rois = torch.tensor(
		[
			[0, 0.0, 0.0, float(w - 1), float(h - 1)],
			[min(1, n - 1), 0.5, 1.0, float(w) - 1.2, float(h) - 1.5],
			[0, 1.3, 0.7, float(w) - 2.1, float(h) - 2.0],
		],
		device=device,
		dtype=torch.float32,
	)
	return x, rois


def _run_forward_backward(
	fn,
	input: torch.Tensor,
	rois: torch.Tensor,
	output_size: tuple[int, int],
	spatial_scale: float,
	sampling_ratio: int,
	pool_mode: str,
	aligned: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
	x = input.clone().detach().requires_grad_(True)
	out = fn(x, rois, output_size, spatial_scale, sampling_ratio, pool_mode, aligned)
	upstream = torch.linspace(0.1, 1.0, out.numel(), device=out.device, dtype=out.dtype).view_as(
		out
	)
	(out * upstream).sum().backward()
	assert x.grad is not None
	return out.detach(), x.grad.detach()


def _assert_forward_close(a: torch.Tensor, b: torch.Tensor, dtype: torch.dtype):
	rtol, atol = DTYPE_EPS[dtype]
	assert torch.allclose(a, b, rtol=rtol, atol=atol)


def _assert_backward_close(a: torch.Tensor, b: torch.Tensor, dtype: torch.dtype):
	rtol, atol = DTYPE_EPS[dtype]
	assert torch.allclose(a, b, rtol=rtol, atol=atol)


def _assert_close(a: torch.Tensor, b: torch.Tensor, dtype: torch.dtype):
	_assert_forward_close(a, b, dtype)


@pytest.mark.parametrize('pool_mode', ['avg', 'max'])
@pytest.mark.parametrize('aligned', [True, False])
@pytest.mark.parametrize('sampling_ratio', [0, 2])
def test_roi_align_cpu_reference_forward_backward(
	pool_mode: str,
	aligned: bool,
	sampling_ratio: int,
):
	x, rois = _build_case_tensors(
		input_shape=(2, 3, 7, 6),
		device='cpu',
		dtype=torch.float32,
		seed=0,
	)
	out, grad = _run_forward_backward(
		roi_align,
		x,
		rois,
		(3, 2),
		1.0,
		sampling_ratio,
		pool_mode,
		aligned,
	)

	assert out.shape == (rois.size(0), x.size(1), 3, 2)
	assert grad.shape == x.shape
	assert torch.isfinite(out).all()
	assert torch.isfinite(grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is required')
@pytest.mark.parametrize('pool_mode', ['avg', 'max'])
@pytest.mark.parametrize('aligned', [True, False])
@pytest.mark.parametrize('sampling_ratio', [0, 2])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
def test_roi_align_cpu_fallback_matches_triton(
	pool_mode: str,
	aligned: bool,
	sampling_ratio: int,
	dtype: torch.dtype,
):
	if dtype is torch.bfloat16:
		_skip_if_bfloat16_unsupported()
	x_cpu, rois_cpu = _build_case_tensors(
		input_shape=(2, 4, 9, 7),
		device='cpu',
		dtype=dtype,
		seed=7,
	)
	out_cpu, grad_cpu = _run_forward_backward(
		roi_align,
		x_cpu,
		rois_cpu,
		(3, 2),
		0.75,
		sampling_ratio,
		pool_mode,
		aligned,
	)

	x_cuda = x_cpu.cuda()
	rois_cuda = rois_cpu.cuda()
	out_cuda, grad_cuda = _run_forward_backward(
		roi_align,
		x_cuda,
		rois_cuda,
		(3, 2),
		0.75,
		sampling_ratio,
		pool_mode,
		aligned,
	)

	_assert_forward_close(out_cpu.float(), out_cuda.cpu().float(), dtype)
	_assert_backward_close(grad_cpu.float(), grad_cuda.cpu().float(), dtype)


def test_roi_align_module_wrapper_matches_function():
	x, rois = _build_case_tensors(
		input_shape=(2, 3, 8, 8),
		device='cpu',
		dtype=torch.float32,
		seed=3,
	)
	module = RoIAlign((2, 3), spatial_scale=1.0, sampling_ratio=2, pool_mode='avg', aligned=True)

	out_module = module(x, rois)
	out_function = roi_align(x, rois, (2, 3), 1.0, 2, 'avg', True)
	_assert_forward_close(out_module, out_function, torch.float32)


@pytest.mark.skipif(cuda_mod is None, reason='CUDA extension module import failed')
def test_00_roi_align_reference_extension_compile_success():
	assert hasattr(cuda_mod, 'ext_module')
	assert hasattr(cuda_mod.ext_module, 'roi_align_forward')
	assert hasattr(cuda_mod.ext_module, 'roi_align_backward')


def _run_cpu_reference_vs_triton_cpu_fallback_case(
	*,
	dtype: torch.dtype,
	pool_mode: str,
	aligned: bool,
	sampling_ratio: int,
):
	x_cpu, rois_cpu = _build_case_tensors(
		input_shape=(2, 4, 9, 7),
		device='cpu',
		dtype=dtype,
		seed=11,
	)

	out_cpu_tri, grad_cpu_tri = _run_forward_backward(
		roi_align,
		x_cpu,
		rois_cpu,
		(3, 2),
		0.75,
		sampling_ratio,
		pool_mode,
		aligned,
	)
	out_cpu_ref, grad_cpu_ref = _run_forward_backward(
		roi_align_cuda,
		x_cpu,
		rois_cpu,
		(3, 2),
		0.75,
		sampling_ratio,
		pool_mode,
		aligned,
	)

	_assert_forward_close(out_cpu_tri.float(), out_cpu_ref.float(), dtype)
	_assert_backward_close(grad_cpu_tri.float(), grad_cpu_ref.float(), dtype)


def _run_cuda_reference_vs_triton_case(
	*,
	dtype: torch.dtype,
	pool_mode: str,
	aligned: bool,
	sampling_ratio: int,
):
	x_cpu, rois_cpu = _build_case_tensors(
		input_shape=(2, 4, 9, 7),
		device='cpu',
		dtype=dtype,
		seed=11,
	)
	x_cuda = x_cpu.cuda()
	rois_cuda = rois_cpu.cuda()
	out_cuda_ref, grad_cuda_ref = _run_forward_backward(
		roi_align_cuda,
		x_cuda,
		rois_cuda,
		(3, 2),
		0.75,
		sampling_ratio,
		pool_mode,
		aligned,
	)
	out_cuda_tri, grad_cuda_tri = _run_forward_backward(
		roi_align,
		x_cuda,
		rois_cuda,
		(3, 2),
		0.75,
		sampling_ratio,
		pool_mode,
		aligned,
	)
	_assert_forward_close(out_cuda_tri.float(), out_cuda_ref.float(), dtype)
	_assert_backward_close(grad_cuda_tri.float(), grad_cuda_ref.float(), dtype)


def _run_cpu_reference_vs_cuda_reference_case(
	*,
	dtype: torch.dtype,
	pool_mode: str,
	aligned: bool,
	sampling_ratio: int,
):
	x_cpu, rois_cpu = _build_case_tensors(
		input_shape=(2, 4, 9, 7),
		device='cpu',
		dtype=dtype,
		seed=13,
	)
	out_cpu_ref, grad_cpu_ref = _run_forward_backward(
		roi_align_cuda,
		x_cpu,
		rois_cpu,
		(3, 2),
		0.75,
		sampling_ratio,
		pool_mode,
		aligned,
	)

	x_cuda = x_cpu.cuda()
	rois_cuda = rois_cpu.cuda()
	out_cuda_ref, grad_cuda_ref = _run_forward_backward(
		roi_align_cuda,
		x_cuda,
		rois_cuda,
		(3, 2),
		0.75,
		sampling_ratio,
		pool_mode,
		aligned,
	)
	_assert_forward_close(out_cpu_ref.float(), out_cuda_ref.cpu().float(), dtype)
	_assert_backward_close(grad_cpu_ref.float(), grad_cuda_ref.cpu().float(), dtype)


def _run_cpu_fallback_vs_cuda_reference_case(
	*,
	dtype: torch.dtype,
	pool_mode: str,
	aligned: bool,
	sampling_ratio: int,
):
	x_cpu, rois_cpu = _build_case_tensors(
		input_shape=(2, 4, 9, 7),
		device='cpu',
		dtype=dtype,
		seed=17,
	)
	out_cpu_tri, grad_cpu_tri = _run_forward_backward(
		roi_align,
		x_cpu,
		rois_cpu,
		(3, 2),
		0.75,
		sampling_ratio,
		pool_mode,
		aligned,
	)

	x_cuda = x_cpu.cuda()
	rois_cuda = rois_cpu.cuda()
	out_cuda_ref, grad_cuda_ref = _run_forward_backward(
		roi_align_cuda,
		x_cuda,
		rois_cuda,
		(3, 2),
		0.75,
		sampling_ratio,
		pool_mode,
		aligned,
	)
	_assert_forward_close(out_cpu_tri.float(), out_cuda_ref.cpu().float(), dtype)
	_assert_backward_close(grad_cpu_tri.float(), grad_cuda_ref.cpu().float(), dtype)


@pytest.mark.parametrize('pool_mode', ['avg', 'max'])
@pytest.mark.parametrize('aligned', [True, False])
@pytest.mark.parametrize('sampling_ratio', [0, 2])
def test_roi_align_cpu_reference_matches_triton_cpu_fallback_float32(
	pool_mode: str,
	aligned: bool,
	sampling_ratio: int,
):
	_run_cpu_reference_vs_triton_cpu_fallback_case(
		dtype=torch.float32,
		pool_mode=pool_mode,
		aligned=aligned,
		sampling_ratio=sampling_ratio,
	)


@pytest.mark.parametrize('pool_mode', ['avg', 'max'])
@pytest.mark.parametrize('aligned', [True, False])
@pytest.mark.parametrize('sampling_ratio', [0, 2])
def test_roi_align_cpu_reference_matches_triton_cpu_fallback_float16(
	pool_mode: str,
	aligned: bool,
	sampling_ratio: int,
):
	_run_cpu_reference_vs_triton_cpu_fallback_case(
		dtype=torch.float16,
		pool_mode=pool_mode,
		aligned=aligned,
		sampling_ratio=sampling_ratio,
	)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is required')
@pytest.mark.parametrize('pool_mode', ['avg', 'max'])
@pytest.mark.parametrize('aligned', [True, False])
@pytest.mark.parametrize('sampling_ratio', [0, 2])
def test_roi_align_cpu_reference_matches_triton_cpu_fallback_bfloat16(
	pool_mode: str,
	aligned: bool,
	sampling_ratio: int,
):
	_skip_if_bfloat16_unsupported()
	_run_cpu_reference_vs_triton_cpu_fallback_case(
		dtype=torch.bfloat16,
		pool_mode=pool_mode,
		aligned=aligned,
		sampling_ratio=sampling_ratio,
	)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is required')
@pytest.mark.skipif(roi_align_cuda is None, reason='CUDA extension module import failed')
@pytest.mark.parametrize('pool_mode', ['avg', 'max'])
@pytest.mark.parametrize('aligned', [True, False])
@pytest.mark.parametrize('sampling_ratio', [0, 2])
def test_roi_align_cuda_reference_matches_triton_float32(
	pool_mode: str,
	aligned: bool,
	sampling_ratio: int,
):
	_run_cuda_reference_vs_triton_case(
		dtype=torch.float32,
		pool_mode=pool_mode,
		aligned=aligned,
		sampling_ratio=sampling_ratio,
	)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is required')
@pytest.mark.skipif(roi_align_cuda is None, reason='CUDA extension module import failed')
@pytest.mark.parametrize('pool_mode', ['avg', 'max'])
@pytest.mark.parametrize('aligned', [True, False])
@pytest.mark.parametrize('sampling_ratio', [0, 2])
def test_roi_align_cuda_reference_matches_triton_float16(
	pool_mode: str,
	aligned: bool,
	sampling_ratio: int,
):
	_run_cuda_reference_vs_triton_case(
		dtype=torch.float16,
		pool_mode=pool_mode,
		aligned=aligned,
		sampling_ratio=sampling_ratio,
	)


@pytest.mark.skipif(roi_align_cuda is None, reason='CUDA extension module import failed')
@pytest.mark.parametrize('pool_mode', ['avg', 'max'])
@pytest.mark.parametrize('aligned', [True, False])
@pytest.mark.parametrize('sampling_ratio', [0, 2])
def test_roi_align_cuda_reference_matches_triton_bfloat16(
	pool_mode: str,
	aligned: bool,
	sampling_ratio: int,
):
	_skip_if_bfloat16_unsupported()
	_run_cuda_reference_vs_triton_case(
		dtype=torch.bfloat16,
		pool_mode=pool_mode,
		aligned=aligned,
		sampling_ratio=sampling_ratio,
	)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is required')
@pytest.mark.skipif(roi_align_cuda is None, reason='CUDA extension module import failed')
@pytest.mark.parametrize('pool_mode', ['avg', 'max'])
@pytest.mark.parametrize('aligned', [True, False])
@pytest.mark.parametrize('sampling_ratio', [0, 2])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
def test_roi_align_cpu_reference_matches_cuda_reference(
	pool_mode: str,
	aligned: bool,
	sampling_ratio: int,
	dtype: torch.dtype,
):
	if dtype is torch.bfloat16:
		_skip_if_bfloat16_unsupported()
	_run_cpu_reference_vs_cuda_reference_case(
		dtype=dtype,
		pool_mode=pool_mode,
		aligned=aligned,
		sampling_ratio=sampling_ratio,
	)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is required')
@pytest.mark.skipif(roi_align_cuda is None, reason='CUDA extension module import failed')
@pytest.mark.parametrize('pool_mode', ['avg', 'max'])
@pytest.mark.parametrize('aligned', [True, False])
@pytest.mark.parametrize('sampling_ratio', [0, 2])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
def test_roi_align_cpu_fallback_matches_cuda_reference(
	pool_mode: str,
	aligned: bool,
	sampling_ratio: int,
	dtype: torch.dtype,
):
	if dtype is torch.bfloat16:
		_skip_if_bfloat16_unsupported()
	_run_cpu_fallback_vs_cuda_reference_case(
		dtype=dtype,
		pool_mode=pool_mode,
		aligned=aligned,
		sampling_ratio=sampling_ratio,
	)
