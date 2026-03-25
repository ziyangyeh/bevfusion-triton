from __future__ import annotations

import pytest
import torch

from ops.sigmoid_focal_loss import sigmoid_focal_loss
from tests.test_ops._eps import DTYPE_EPS, dtype_tiny

try:
	import ops.sigmoid_focal_loss.sigmoid_focal_loss_legacy as cuda_mod

	sigmoid_focal_loss_cuda = cuda_mod.sigmoid_focal_loss
except Exception:  # pragma: no cover
	cuda_mod = None
	sigmoid_focal_loss_cuda = None


def _run_forward_backward(
	fn,
	input: torch.Tensor,
	target: torch.Tensor,
	gamma: float,
	alpha: float,
	weight: torch.Tensor | None,
	reduction: str,
) -> tuple[torch.Tensor, torch.Tensor]:
	x = input.clone().detach().requires_grad_(True)
	out = fn(x, target, gamma, alpha, weight, reduction)

	if out.dim() == 0:
		loss = out
	else:
		upstream = torch.linspace(
			0.1, 1.0, out.numel(), device=out.device, dtype=out.dtype
		).view_as(out)
		loss = (out * upstream).sum()

	loss.backward()
	assert x.grad is not None
	return out.detach(), x.grad.detach()


def _assert_forward_close(a: torch.Tensor, b: torch.Tensor, dtype: torch.dtype):
	rtol, atol = DTYPE_EPS[dtype]
	assert torch.allclose(a, b, rtol=rtol, atol=atol)


def _assert_backward_close(a: torch.Tensor, b: torch.Tensor, dtype: torch.dtype):
	rtol, atol = DTYPE_EPS[dtype]
	assert torch.allclose(a, b, rtol=rtol, atol=atol)


def _build_case_tensors(
	shape: tuple[int, int],
	*,
	device: str,
	use_weight: bool,
	seed: int,
	dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
	torch.manual_seed(seed)
	n, c = shape
	x = torch.randn(n, c, device=device, dtype=dtype)
	target = torch.randint(0, c, (n,), device=device, dtype=torch.long)
	weight = torch.rand(c, device=device, dtype=dtype) if use_weight else None
	return x, target, weight


def _skip_if_bfloat16_unsupported():
	if not torch.cuda.is_available():
		pytest.skip('CUDA is required')
	if not torch.cuda.is_bf16_supported():
		pytest.skip('CUDA bfloat16 is required')


@pytest.mark.skipif(cuda_mod is None, reason='CUDA extension module import failed')
def test_00_sigmoid_focal_loss_reference_extension_compile_success():
	assert hasattr(cuda_mod, 'ext_module')
	assert hasattr(cuda_mod.ext_module, 'sigmoid_focal_loss_forward')
	assert hasattr(cuda_mod.ext_module, 'sigmoid_focal_loss_backward')


def _run_cpu_fallback_vs_reference_case(
	*,
	dtype: torch.dtype,
	reduction: str,
	use_weight: bool,
	gamma: float,
	alpha: float,
	shape: tuple[int, int],
):
	x_cpu, target_cpu, weight_cpu = _build_case_tensors(
		shape, device='cpu', use_weight=use_weight, seed=7, dtype=dtype
	)

	out_cpu, grad_cpu = _run_forward_backward(
		sigmoid_focal_loss,
		x_cpu,
		target_cpu,
		gamma,
		alpha,
		weight_cpu,
		reduction,
	)

	x_cuda = x_cpu.cuda()
	target_cuda = target_cpu.cuda()
	weight_cuda = weight_cpu.cuda() if weight_cpu is not None else None

	out_cuda, grad_cuda = _run_forward_backward(
		sigmoid_focal_loss_cuda,
		x_cuda,
		target_cuda,
		gamma,
		alpha,
		weight_cuda,
		reduction,
	)

	_assert_forward_close(out_cpu.float(), out_cuda.cpu().float(), dtype)
	_assert_backward_close(grad_cpu.float(), grad_cuda.cpu().float(), dtype)


def _run_cuda_triton_vs_reference_case(
	*,
	dtype: torch.dtype,
	reduction: str,
	use_weight: bool,
	gamma: float,
	alpha: float,
	shape: tuple[int, int],
):
	x, target, weight = _build_case_tensors(
		shape,
		device='cuda',
		use_weight=use_weight,
		seed=11,
		dtype=dtype,
	)

	out_tri, grad_tri = _run_forward_backward(
		sigmoid_focal_loss,
		x,
		target,
		gamma,
		alpha,
		weight,
		reduction,
	)
	out_cuda, grad_cuda = _run_forward_backward(
		sigmoid_focal_loss_cuda,
		x,
		target,
		gamma,
		alpha,
		weight,
		reduction,
	)

	_assert_forward_close(out_tri.float(), out_cuda.float(), dtype)
	_assert_backward_close(grad_tri.float(), grad_cuda.float(), dtype)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is required')
@pytest.mark.skipif(sigmoid_focal_loss_cuda is None, reason='CUDA extension module import failed')
@pytest.mark.parametrize('reduction', ['none', 'mean', 'sum'])
@pytest.mark.parametrize('use_weight', [False, True])
@pytest.mark.parametrize('gamma,alpha', [(2.0, 0.25), (1.5, 0.75)])
@pytest.mark.parametrize('shape', [(9, 4), (17, 8)])
def test_sigmoid_focal_loss_cpu_fallback_matches_reference_float32(
	reduction: str,
	use_weight: bool,
	gamma: float,
	alpha: float,
	shape: tuple[int, int],
):
	_run_cpu_fallback_vs_reference_case(
		dtype=torch.float32,
		reduction=reduction,
		use_weight=use_weight,
		gamma=gamma,
		alpha=alpha,
		shape=shape,
	)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is required')
@pytest.mark.skipif(sigmoid_focal_loss_cuda is None, reason='CUDA extension module import failed')
@pytest.mark.parametrize('reduction', ['none', 'mean', 'sum'])
@pytest.mark.parametrize('use_weight', [False, True])
@pytest.mark.parametrize('gamma,alpha', [(2.0, 0.25), (1.5, 0.75)])
@pytest.mark.parametrize('shape', [(9, 4), (17, 8)])
def test_sigmoid_focal_loss_cpu_fallback_matches_reference_float16(
	reduction: str,
	use_weight: bool,
	gamma: float,
	alpha: float,
	shape: tuple[int, int],
):
	_run_cpu_fallback_vs_reference_case(
		dtype=torch.float16,
		reduction=reduction,
		use_weight=use_weight,
		gamma=gamma,
		alpha=alpha,
		shape=shape,
	)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is required')
@pytest.mark.skipif(sigmoid_focal_loss_cuda is None, reason='CUDA extension module import failed')
@pytest.mark.parametrize('reduction', ['none', 'mean', 'sum'])
@pytest.mark.parametrize('use_weight', [False, True])
@pytest.mark.parametrize('gamma,alpha', [(2.0, 0.25), (1.5, 0.75)])
@pytest.mark.parametrize('shape', [(8, 5), (19, 7)])
def test_sigmoid_focal_loss_cuda_matches_reference_float32(
	reduction: str,
	use_weight: bool,
	gamma: float,
	alpha: float,
	shape: tuple[int, int],
):
	_run_cuda_triton_vs_reference_case(
		dtype=torch.float32,
		reduction=reduction,
		use_weight=use_weight,
		gamma=gamma,
		alpha=alpha,
		shape=shape,
	)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is required')
@pytest.mark.skipif(sigmoid_focal_loss_cuda is None, reason='CUDA extension module import failed')
@pytest.mark.parametrize('reduction', ['none', 'mean', 'sum'])
@pytest.mark.parametrize('use_weight', [False, True])
@pytest.mark.parametrize('gamma,alpha', [(2.0, 0.25), (1.5, 0.75)])
@pytest.mark.parametrize('shape', [(8, 5), (19, 7)])
def test_sigmoid_focal_loss_cuda_matches_reference_float16(
	reduction: str,
	use_weight: bool,
	gamma: float,
	alpha: float,
	shape: tuple[int, int],
):
	_run_cuda_triton_vs_reference_case(
		dtype=torch.float16,
		reduction=reduction,
		use_weight=use_weight,
		gamma=gamma,
		alpha=alpha,
		shape=shape,
	)


@pytest.mark.skipif(sigmoid_focal_loss_cuda is None, reason='CUDA extension module import failed')
@pytest.mark.parametrize('reduction', ['none', 'mean', 'sum'])
@pytest.mark.parametrize('use_weight', [False, True])
@pytest.mark.parametrize('gamma,alpha', [(2.0, 0.25), (1.5, 0.75)])
@pytest.mark.parametrize('shape', [(9, 4), (17, 8)])
def test_sigmoid_focal_loss_cpu_fallback_matches_reference_bfloat16(
	reduction: str,
	use_weight: bool,
	gamma: float,
	alpha: float,
	shape: tuple[int, int],
):
	_skip_if_bfloat16_unsupported()
	_run_cpu_fallback_vs_reference_case(
		dtype=torch.bfloat16,
		reduction=reduction,
		use_weight=use_weight,
		gamma=gamma,
		alpha=alpha,
		shape=shape,
	)


@pytest.mark.skipif(sigmoid_focal_loss_cuda is None, reason='CUDA extension module import failed')
@pytest.mark.parametrize('reduction', ['none', 'mean', 'sum'])
@pytest.mark.parametrize('use_weight', [False, True])
@pytest.mark.parametrize('gamma,alpha', [(2.0, 0.25), (1.5, 0.75)])
@pytest.mark.parametrize('shape', [(8, 5), (19, 7)])
def test_sigmoid_focal_loss_cuda_matches_reference_bfloat16(
	reduction: str,
	use_weight: bool,
	gamma: float,
	alpha: float,
	shape: tuple[int, int],
):
	_skip_if_bfloat16_unsupported()
	_run_cuda_triton_vs_reference_case(
		dtype=torch.bfloat16,
		reduction=reduction,
		use_weight=use_weight,
		gamma=gamma,
		alpha=alpha,
		shape=shape,
	)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is required')
@pytest.mark.skipif(sigmoid_focal_loss_cuda is None, reason='CUDA extension module import failed')
@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
def test_sigmoid_focal_loss_eps_alignment_extreme_logits(dtype: torch.dtype):
	if dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
		pytest.skip('CUDA bfloat16 is required')
	x_cpu = torch.tensor(
		[
			[80.0, -80.0, 0.0, -20.0],
			[-80.0, 80.0, -1.0, 20.0],
			[40.0, -40.0, 15.0, -15.0],
		],
		dtype=dtype,
		device='cpu',
	)
	target_cpu = torch.tensor([0, 1, 2], dtype=torch.long, device='cpu')
	weight_cpu = torch.tensor([1.0, 0.5, 2.0, 1.5], dtype=dtype, device='cpu')

	assert dtype_tiny(dtype) == pytest.approx(torch.finfo(dtype).tiny)

	out_cpu, grad_cpu = _run_forward_backward(
		sigmoid_focal_loss,
		x_cpu,
		target_cpu,
		2.0,
		0.25,
		weight_cpu,
		'none',
	)

	x_cuda = x_cpu.cuda()
	target_cuda = target_cpu.cuda()
	weight_cuda = weight_cpu.cuda()

	out_ref, grad_ref = _run_forward_backward(
		sigmoid_focal_loss_cuda,
		x_cuda,
		target_cuda,
		2.0,
		0.25,
		weight_cuda,
		'none',
	)
	out_tri, grad_tri = _run_forward_backward(
		sigmoid_focal_loss,
		x_cuda,
		target_cuda,
		2.0,
		0.25,
		weight_cuda,
		'none',
	)

	assert torch.isfinite(out_cpu).all()
	assert torch.isfinite(grad_cpu).all()
	assert torch.isfinite(out_tri).all()
	assert torch.isfinite(grad_tri).all()

	if torch.isfinite(out_ref).all() and torch.isfinite(grad_ref).all():
		_assert_forward_close(out_cpu.float(), out_ref.cpu().float(), dtype)
		_assert_forward_close(out_tri.float(), out_ref.float(), dtype)
		_assert_backward_close(grad_tri.float(), grad_ref.float(), dtype)

	_assert_forward_close(out_tri.cpu().float(), out_cpu.float(), dtype)
