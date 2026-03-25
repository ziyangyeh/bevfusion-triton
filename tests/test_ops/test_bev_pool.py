from __future__ import annotations

import pytest
import torch

from ops.bev_pool import bev_pool

try:
	import ops.bev_pool.bev_pool_legacy as legacy_ref_mod

	bev_pool_reference = legacy_ref_mod.bev_pool
except Exception:  # pragma: no cover
	legacy_ref_mod = None
	bev_pool_reference = None


def _skip_if_bfloat16_unsupported():
	if not torch.cuda.is_available():
		pytest.skip('CUDA is required')
	if not torch.cuda.is_bf16_supported():
		pytest.skip('CUDA bfloat16 is required')


def _build_case_tensors(
	*,
	num_points: int,
	channels: int,
	B: int,
	D: int,
	H: int,
	W: int,
	device: str,
	dtype: torch.dtype,
	seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
	torch.manual_seed(seed)
	feats = torch.randn((num_points, channels), device=device, dtype=dtype)
	coords = torch.stack(
		[
			torch.randint(0, H, (num_points,), device=device),
			torch.randint(0, W, (num_points,), device=device),
			torch.randint(0, D, (num_points,), device=device),
			torch.randint(0, B, (num_points,), device=device),
		],
		dim=1,
	).to(torch.int32)
	return feats.contiguous(), coords.contiguous()


@pytest.mark.skipif(legacy_ref_mod is None, reason='Legacy reference extension import failed')
def test_00_bev_pool_legacy_reference_extension_compile_success():
	assert hasattr(legacy_ref_mod, 'ext_module')
	assert hasattr(legacy_ref_mod.ext_module, 'bev_pool_forward')
	assert hasattr(legacy_ref_mod.ext_module, 'bev_pool_backward')


def test_bev_pool_cpu_fallback_known_case():
	feats = torch.tensor(
		[
			[1.0, 2.0],
			[3.0, 4.0],
			[5.0, 6.0],
		],
		dtype=torch.float32,
	)
	coords = torch.tensor(
		[
			[0, 0, 0, 0],
			[0, 0, 0, 0],
			[1, 0, 0, 0],
		],
		dtype=torch.int32,
	)
	out = bev_pool(feats, coords, 1, 1, 2, 1)
	expected = torch.tensor(
		[
			[
				[[[4.0], [5.0]]],
				[[[6.0], [6.0]]],
			]
		],
		dtype=torch.float32,
	)
	assert torch.equal(out, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is required')
@pytest.mark.skipif(bev_pool_reference is None, reason='Legacy reference extension import failed')
@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
def test_bev_pool_matches_legacy_reference(dtype: torch.dtype):
	if dtype is torch.bfloat16:
		_skip_if_bfloat16_unsupported()

	B, D, H, W = 2, 3, 4, 5
	feats, coords = _build_case_tensors(
		num_points=64,
		channels=8,
		B=B,
		D=D,
		H=H,
		W=W,
		device='cuda',
		dtype=dtype,
		seed=0,
	)

	feats_t = feats.clone().detach().requires_grad_(True)
	feats_r = feats.clone().detach().requires_grad_(True)
	out_t = bev_pool(feats_t, coords, B, D, H, W)
	out_r = bev_pool_reference(feats_r.float(), coords, B, D, H, W).to(dtype)

	assert torch.allclose(out_t, out_r, rtol=5e-2, atol=5e-2)

	upstream = torch.randn_like(out_t)
	(out_t * upstream).sum().backward()
	(out_r * upstream).sum().backward()

	ref_grad = feats_r.grad.to(dtype)
	assert torch.allclose(feats_t.grad, ref_grad, rtol=5e-2, atol=5e-2)


def test_bev_pool_rejects_invalid_inputs():
	feats = torch.randint(0, 4, (4, 3), dtype=torch.int32)
	coords = torch.zeros((4, 4), dtype=torch.int32)
	with pytest.raises(TypeError):
		bev_pool(feats, coords, 1, 1, 1, 1)

	feats = torch.randn((4, 3), dtype=torch.float32)
	coords = torch.zeros((4, 4), dtype=torch.float32)
	with pytest.raises(TypeError):
		bev_pool(feats, coords, 1, 1, 1, 1)
