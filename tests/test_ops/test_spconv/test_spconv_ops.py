from __future__ import annotations

import pytest
import torch

from ops.spconv.ops import sparse_conv_ext as legacy_reference_ext
from ops.spconv.src import spconv_triton
from tests.test_ops._eps import DTYPE_EPS

_EXPORTED_FUNCTIONS = [
	'get_indice_pairs_2d',
	'get_indice_pairs_3d',
	'get_indice_pairs_4d',
	'get_indice_pairs_grid_2d',
	'get_indice_pairs_grid_3d',
	'indice_conv_fp32',
	'indice_conv_backward_fp32',
	'indice_conv_half',
	'indice_conv_backward_half',
	'fused_indice_conv_fp32',
	'fused_indice_conv_half',
	'indice_maxpool_fp32',
	'indice_maxpool_backward_fp32',
	'indice_maxpool_half',
	'indice_maxpool_backward_half',
]

_PAIRWISE_OPS_FP32 = [
	'indice_conv_fp32',
	'indice_conv_backward_fp32',
	'fused_indice_conv_fp32',
	'indice_maxpool_fp32',
	'indice_maxpool_backward_fp32',
]

_PAIRWISE_OPS_HALF = [
	'indice_conv_half',
	'indice_conv_backward_half',
	'fused_indice_conv_half',
	'indice_maxpool_half',
	'indice_maxpool_backward_half',
]

_CONV_FLAG_CASES = [
	(0, 0),
	(0, 1),
	(1, 0),
	(1, 1),
]

_CASE_CONFIGS_2D = [
	{
		'name': 'single_batch_2x2',
		'indices': [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1]],
		'spatial_shape': [2, 2],
		'out_shape': [2, 2],
		'in_c': 3,
		'out_c': 5,
		'seed': 42,
	},
	{
		'name': 'two_batch_3x3',
		'indices': [[0, 0, 0], [0, 1, 2], [0, 2, 1], [1, 0, 1], [1, 1, 0], [1, 2, 2]],
		'spatial_shape': [3, 3],
		'out_shape': [3, 3],
		'in_c': 4,
		'out_c': 6,
		'seed': 314,
	},
]


def _assert_close(a: torch.Tensor, b: torch.Tensor, dtype: torch.dtype):
	rtol, atol = DTYPE_EPS[dtype]
	assert torch.allclose(a, b, rtol=rtol, atol=atol)


def _assert_item_close(a, b, dtype: torch.dtype):
	if isinstance(a, tuple):
		assert isinstance(b, tuple)
		assert len(a) == len(b)
		for ta, tb in zip(a, b):
			_assert_close(ta, tb, dtype)
		return
	_assert_close(a, b, dtype)


def _compare_result_dicts(a: dict, b: dict, keys: list[str], dtype: torch.dtype):
	for k in keys:
		_assert_item_close(a[k], b[k], dtype)


def _make_indices_2d(device: torch.device) -> torch.Tensor:
	return torch.tensor(
		[[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1]],
		dtype=torch.int32,
		device=device,
	)


def _make_indices_3d(device: torch.device) -> torch.Tensor:
	return torch.tensor(
		[[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]],
		dtype=torch.int32,
		device=device,
	)


def _make_indices_3d_alt(device: torch.device) -> torch.Tensor:
	return torch.tensor(
		[[0, 0, 0, 0], [0, 1, 1, 0], [1, 0, 0, 1], [1, 1, 0, 0]],
		dtype=torch.int32,
		device=device,
	)


def _make_indices_4d(device: torch.device) -> torch.Tensor:
	return torch.tensor(
		[[0, 0, 0, 0, 0], [0, 0, 0, 0, 1], [0, 0, 0, 1, 0], [0, 0, 1, 0, 0]],
		dtype=torch.int32,
		device=device,
	)


def _make_indices_4d_alt(device: torch.device) -> torch.Tensor:
	return torch.tensor(
		[[0, 0, 0, 0, 0], [0, 1, 0, 1, 0], [1, 0, 1, 0, 1], [1, 1, 0, 0, 0]],
		dtype=torch.int32,
		device=device,
	)


def _build_pairs_2d(
	device: torch.device, case_cfg: dict, subm: bool = False, transpose: bool = False
):
	indices = torch.tensor(case_cfg['indices'], dtype=torch.int32, device=device)
	batch_size = int(indices[:, 0].max().item()) + 1
	outids, pairs, pair_num = legacy_reference_ext.get_indice_pairs_2d(
		indices,
		batch_size,
		case_cfg['out_shape'],
		case_cfg['spatial_shape'],
		[1, 1],
		[1, 1],
		[0, 0],
		[1, 1],
		[0, 0],
		int(subm),
		int(transpose),
	)
	return outids, pairs, pair_num


def _build_case_tensors(dtype: torch.dtype, device: torch.device, case_cfg: dict):
	torch.manual_seed(case_cfg['seed'])
	n_in = len(case_cfg['indices'])
	in_c = case_cfg['in_c']
	out_c = case_cfg['out_c']

	features = torch.randn(n_in, in_c, dtype=dtype, device='cpu').to(device)
	filters = torch.randn(1, 1, in_c, out_c, dtype=dtype, device='cpu').to(device)
	bias = torch.randn(out_c, dtype=dtype, device='cpu').to(device)

	out_bp = torch.randn(n_in, out_c, dtype=dtype, device='cpu').to(device)
	pool_bp = torch.randn(n_in, in_c, dtype=dtype, device='cpu').to(device)

	return features, filters, bias, out_bp, pool_bp


def _run_legacy_reference_impl(dtype: torch.dtype, device: torch.device, case_cfg: dict) -> dict:
	outids, pairs, pair_num = _build_pairs_2d(device, case_cfg)
	features, filters, bias, out_bp, pool_bp = _build_case_tensors(dtype, device, case_cfg)

	if dtype == torch.float32:
		out = legacy_reference_ext.indice_conv_fp32(
			features, filters, pairs, pair_num, outids.shape[0], 0, 0
		)
		out_fused = legacy_reference_ext.fused_indice_conv_fp32(
			features, filters, bias, pairs, pair_num, outids.shape[0], 0, 0
		)
		in_bp, filt_bp = legacy_reference_ext.indice_conv_backward_fp32(
			features, filters, out_bp, pairs, pair_num, 0, 0
		)
		pool = legacy_reference_ext.indice_maxpool_fp32(features, pairs, pair_num, outids.shape[0])
		in_pool_bp = legacy_reference_ext.indice_maxpool_backward_fp32(
			features, pool, pool_bp, pairs, pair_num
		)

		return {
			'indice_conv_fp32': out,
			'fused_indice_conv_fp32': out_fused,
			'indice_conv_backward_fp32': (in_bp, filt_bp),
			'indice_maxpool_fp32': pool,
			'indice_maxpool_backward_fp32': in_pool_bp,
		}

	out = legacy_reference_ext.indice_conv_half(
		features, filters, pairs, pair_num, outids.shape[0], 0, 0
	)
	out_fused = legacy_reference_ext.fused_indice_conv_half(
		features, filters, bias, pairs, pair_num, outids.shape[0], 0, 0
	)
	in_bp, filt_bp = legacy_reference_ext.indice_conv_backward_half(
		features, filters, out_bp, pairs, pair_num, 0, 0
	)
	pool = legacy_reference_ext.indice_maxpool_half(features, pairs, pair_num, outids.shape[0])
	in_pool_bp = legacy_reference_ext.indice_maxpool_backward_half(
		features, pool, pool_bp, pairs, pair_num
	)

	return {
		'indice_conv_half': out,
		'fused_indice_conv_half': out_fused,
		'indice_conv_backward_half': (in_bp, filt_bp),
		'indice_maxpool_half': pool,
		'indice_maxpool_backward_half': in_pool_bp,
	}


def _run_legacy_reference_conv_family_with_flags(
	dtype: torch.dtype,
	device: torch.device,
	case_cfg: dict,
	inverse: int,
	subm: int,
) -> dict:
	outids, pairs, pair_num = _build_pairs_2d(device, case_cfg, subm=bool(subm), transpose=False)
	features, filters, bias, out_bp, _ = _build_case_tensors(dtype, device, case_cfg)
	if dtype == torch.float32:
		out = legacy_reference_ext.indice_conv_fp32(
			features, filters, pairs, pair_num, outids.shape[0], inverse, subm
		)
		out_fused = legacy_reference_ext.fused_indice_conv_fp32(
			features, filters, bias, pairs, pair_num, outids.shape[0], inverse, subm
		)
		in_bp, filt_bp = legacy_reference_ext.indice_conv_backward_fp32(
			features, filters, out_bp, pairs, pair_num, inverse, subm
		)
		return {
			'indice_conv_fp32': out,
			'fused_indice_conv_fp32': out_fused,
			'indice_conv_backward_fp32': (in_bp, filt_bp),
		}

	out = legacy_reference_ext.indice_conv_half(
		features, filters, pairs, pair_num, outids.shape[0], inverse, subm
	)
	out_fused = legacy_reference_ext.fused_indice_conv_half(
		features, filters, bias, pairs, pair_num, outids.shape[0], inverse, subm
	)
	in_bp, filt_bp = legacy_reference_ext.indice_conv_backward_half(
		features, filters, out_bp, pairs, pair_num, inverse, subm
	)
	return {
		'indice_conv_half': out,
		'fused_indice_conv_half': out_fused,
		'indice_conv_backward_half': (in_bp, filt_bp),
	}


def _run_triton_impl(dtype: torch.dtype, device: torch.device, case_cfg: dict) -> dict:
	outids, pairs, pair_num = _build_pairs_2d(device, case_cfg)
	features, filters, bias, out_bp, pool_bp = _build_case_tensors(dtype, device, case_cfg)

	out = spconv_triton.indice_conv(
		features, filters, pairs, pair_num, outids.shape[0], False, False
	)
	out_fused = spconv_triton.fused_indice_conv(
		features, filters, bias, pairs, pair_num, outids.shape[0], False, False
	)
	in_bp, filt_bp = spconv_triton.indice_conv_backward(
		features, filters, out_bp, pairs, pair_num, False, False
	)
	pool = spconv_triton.indice_maxpool(features, pairs, pair_num, outids.shape[0])
	in_pool_bp = spconv_triton.indice_maxpool_backward(features, pool, pool_bp, pairs, pair_num)

	if dtype == torch.float32:
		return {
			'indice_conv_fp32': out,
			'fused_indice_conv_fp32': out_fused,
			'indice_conv_backward_fp32': (in_bp, filt_bp),
			'indice_maxpool_fp32': pool,
			'indice_maxpool_backward_fp32': in_pool_bp,
		}

	return {
		'indice_conv_half': out,
		'fused_indice_conv_half': out_fused,
		'indice_conv_backward_half': (in_bp, filt_bp),
		'indice_maxpool_half': pool,
		'indice_maxpool_backward_half': in_pool_bp,
	}


def _run_triton_conv_family_with_flags(
	dtype: torch.dtype,
	device: torch.device,
	case_cfg: dict,
	inverse: int,
	subm: int,
) -> dict:
	outids, pairs, pair_num = _build_pairs_2d(device, case_cfg, subm=bool(subm), transpose=False)
	features, filters, bias, out_bp, _ = _build_case_tensors(dtype, device, case_cfg)
	out = spconv_triton.indice_conv(
		features, filters, pairs, pair_num, outids.shape[0], bool(inverse), bool(subm)
	)
	out_fused = spconv_triton.fused_indice_conv(
		features, filters, bias, pairs, pair_num, outids.shape[0], bool(inverse), bool(subm)
	)
	in_bp, filt_bp = spconv_triton.indice_conv_backward(
		features, filters, out_bp, pairs, pair_num, bool(inverse), bool(subm)
	)
	if dtype == torch.float32:
		return {
			'indice_conv_fp32': out,
			'fused_indice_conv_fp32': out_fused,
			'indice_conv_backward_fp32': (in_bp, filt_bp),
		}
	return {
		'indice_conv_half': out,
		'fused_indice_conv_half': out_fused,
		'indice_conv_backward_half': (in_bp, filt_bp),
	}


def _to_cpu_result_dict(result: dict) -> dict:
	out = {}
	for k, v in result.items():
		out[k] = tuple(x.cpu() for x in v) if isinstance(v, tuple) else v.cpu()
	return out


def _run_spconv_legacy_reference_cpu_vs_legacy_reference_cuda(
	dtype: torch.dtype, case_cfg: dict, keys: list[str]
):
	legacy_ref_cpu = _run_legacy_reference_impl(dtype, torch.device('cpu'), case_cfg)
	legacy_ref_cuda = _to_cpu_result_dict(
		_run_legacy_reference_impl(dtype, torch.device('cuda'), case_cfg)
	)
	_compare_result_dicts(legacy_ref_cpu, legacy_ref_cuda, keys, dtype)


def _run_spconv_triton_cpu_vs_triton_cuda(dtype: torch.dtype, case_cfg: dict, keys: list[str]):
	tri_cpu = _run_triton_impl(dtype, torch.device('cpu'), case_cfg)
	tri_cuda = _to_cpu_result_dict(_run_triton_impl(dtype, torch.device('cuda'), case_cfg))
	_compare_result_dicts(tri_cpu, tri_cuda, keys, dtype)


def test_00_spconv_ops_exported_symbols_exist():
	for name in _EXPORTED_FUNCTIONS:
		assert hasattr(ext, name), f'missing symbol: {name}'


def test_00b_spconv_ops_exported_symbols_callable():
	for name in _EXPORTED_FUNCTIONS:
		assert callable(getattr(ext, name))


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16])
@pytest.mark.parametrize('case_cfg', _CASE_CONFIGS_2D, ids=[c['name'] for c in _CASE_CONFIGS_2D])
def test_05_exported_ops_output_contracts_cpu(dtype: torch.dtype, case_cfg: dict):
	outids, _, _ = _build_pairs_2d(torch.device('cpu'), case_cfg)
	result = _run_legacy_reference_impl(dtype, torch.device('cpu'), case_cfg)
	in_c, out_c = case_cfg['in_c'], case_cfg['out_c']
	out_n = outids.shape[0]

	conv_key = 'indice_conv_fp32' if dtype == torch.float32 else 'indice_conv_half'
	fused_key = 'fused_indice_conv_fp32' if dtype == torch.float32 else 'fused_indice_conv_half'
	bp_key = 'indice_conv_backward_fp32' if dtype == torch.float32 else 'indice_conv_backward_half'
	pool_key = 'indice_maxpool_fp32' if dtype == torch.float32 else 'indice_maxpool_half'
	pool_bp_key = (
		'indice_maxpool_backward_fp32' if dtype == torch.float32 else 'indice_maxpool_backward_half'
	)

	conv = result[conv_key]
	fused = result[fused_key]
	in_bp, filt_bp = result[bp_key]
	pool = result[pool_key]
	pool_bp = result[pool_bp_key]

	assert conv.shape == (out_n, out_c)
	assert fused.shape == (out_n, out_c)
	assert in_bp.shape == (len(case_cfg['indices']), in_c)
	assert filt_bp.shape == (1, 1, in_c, out_c)
	assert pool.shape == (out_n, in_c)
	assert pool_bp.shape == (len(case_cfg['indices']), in_c)

	assert conv.dtype == dtype
	assert fused.dtype == dtype
	assert in_bp.dtype == dtype
	assert filt_bp.dtype == dtype
	assert pool.dtype == dtype
	assert pool_bp.dtype == dtype

	assert torch.isfinite(conv).all()
	assert torch.isfinite(fused).all()
	assert torch.isfinite(in_bp).all()
	assert torch.isfinite(filt_bp).all()
	assert torch.isfinite(pool).all()
	assert torch.isfinite(pool_bp).all()


@pytest.mark.parametrize(
	'fn_name,ndim,out_shape,spatial_shape,ksize,stride,padding,dilation,out_padding,use_grid,use_alt_indices',
	[
		(
			'get_indice_pairs_2d',
			2,
			[2, 2],
			[2, 2],
			[1, 1],
			[1, 1],
			[0, 0],
			[1, 1],
			[0, 0],
			False,
			False,
		),
		(
			'get_indice_pairs_2d',
			2,
			[3, 3],
			[3, 3],
			[1, 1],
			[1, 1],
			[0, 0],
			[1, 1],
			[0, 0],
			False,
			True,
		),
		(
			'get_indice_pairs_3d',
			3,
			[2, 2, 2],
			[2, 2, 2],
			[1, 1, 1],
			[1, 1, 1],
			[0, 0, 0],
			[1, 1, 1],
			[0, 0, 0],
			False,
			False,
		),
		(
			'get_indice_pairs_3d',
			3,
			[2, 2, 2],
			[2, 2, 2],
			[1, 1, 1],
			[1, 1, 1],
			[0, 0, 0],
			[1, 1, 1],
			[0, 0, 0],
			False,
			True,
		),
		(
			'get_indice_pairs_4d',
			4,
			[2, 2, 2, 2],
			[2, 2, 2, 2],
			[1, 1, 1, 1],
			[1, 1, 1, 1],
			[0, 0, 0, 0],
			[1, 1, 1, 1],
			[0, 0, 0, 0],
			False,
			False,
		),
		(
			'get_indice_pairs_4d',
			4,
			[2, 2, 2, 2],
			[2, 2, 2, 2],
			[1, 1, 1, 1],
			[1, 1, 1, 1],
			[0, 0, 0, 0],
			[1, 1, 1, 1],
			[0, 0, 0, 0],
			False,
			True,
		),
		(
			'get_indice_pairs_grid_2d',
			2,
			[2, 2],
			[2, 2],
			[1, 1],
			[1, 1],
			[0, 0],
			[1, 1],
			[0, 0],
			True,
			False,
		),
		(
			'get_indice_pairs_grid_2d',
			2,
			[3, 3],
			[3, 3],
			[1, 1],
			[1, 1],
			[0, 0],
			[1, 1],
			[0, 0],
			True,
			True,
		),
		(
			'get_indice_pairs_grid_3d',
			3,
			[2, 2, 2],
			[2, 2, 2],
			[1, 1, 1],
			[1, 1, 1],
			[0, 0, 0],
			[1, 1, 1],
			[0, 0, 0],
			True,
			False,
		),
		(
			'get_indice_pairs_grid_3d',
			3,
			[2, 2, 2],
			[2, 2, 2],
			[1, 1, 1],
			[1, 1, 1],
			[0, 0, 0],
			[1, 1, 1],
			[0, 0, 0],
			True,
			True,
		),
	],
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is required')
def test_10_get_indice_pairs_cpu_cuda_consistency(
	fn_name,
	ndim,
	out_shape,
	spatial_shape,
	ksize,
	stride,
	padding,
	dilation,
	out_padding,
	use_grid,
	use_alt_indices,
):
	batch_size = 1
	if ndim == 2:
		if use_alt_indices:
			idx_cpu = torch.tensor(_CASE_CONFIGS_2D[1]['indices'], dtype=torch.int32, device='cpu')
			idx_cuda = torch.tensor(
				_CASE_CONFIGS_2D[1]['indices'], dtype=torch.int32, device='cuda'
			)
		else:
			idx_cpu = _make_indices_2d(torch.device('cpu'))
			idx_cuda = _make_indices_2d(torch.device('cuda'))
	elif ndim == 3:
		idx_cpu = (
			_make_indices_3d_alt(torch.device('cpu'))
			if use_alt_indices
			else _make_indices_3d(torch.device('cpu'))
		)
		idx_cuda = (
			_make_indices_3d_alt(torch.device('cuda'))
			if use_alt_indices
			else _make_indices_3d(torch.device('cuda'))
		)
	else:
		idx_cpu = (
			_make_indices_4d_alt(torch.device('cpu'))
			if use_alt_indices
			else _make_indices_4d(torch.device('cpu'))
		)
		idx_cuda = (
			_make_indices_4d_alt(torch.device('cuda'))
			if use_alt_indices
			else _make_indices_4d(torch.device('cuda'))
		)
	batch_size = int(idx_cpu[:, 0].max().item()) + 1

	func = getattr(ext, fn_name)
	if use_grid:
		volume = 1
		for s in out_shape:
			volume *= s
		grid_size = volume * batch_size
		grid_cpu = torch.full((grid_size,), -1, dtype=torch.int32)
		grid_cuda = torch.full((grid_size,), -1, dtype=torch.int32, device='cuda')
		out_cpu = func(
			idx_cpu,
			grid_cpu,
			batch_size,
			out_shape,
			spatial_shape,
			ksize,
			stride,
			padding,
			dilation,
			out_padding,
			0,
			0,
		)
		out_cuda = func(
			idx_cuda,
			grid_cuda,
			batch_size,
			out_shape,
			spatial_shape,
			ksize,
			stride,
			padding,
			dilation,
			out_padding,
			0,
			0,
		)
	else:
		out_cpu = func(
			idx_cpu,
			batch_size,
			out_shape,
			spatial_shape,
			ksize,
			stride,
			padding,
			dilation,
			out_padding,
			0,
			0,
		)
		out_cuda = func(
			idx_cuda,
			batch_size,
			out_shape,
			spatial_shape,
			ksize,
			stride,
			padding,
			dilation,
			out_padding,
			0,
			0,
		)

	assert len(out_cpu) == 3
	assert len(out_cuda) == 3
	for a, b in zip(out_cpu, out_cuda):
		assert torch.equal(a.cpu(), b.cpu())


@pytest.mark.parametrize(
	'dtype,keys', [(torch.float32, _PAIRWISE_OPS_FP32), (torch.float16, _PAIRWISE_OPS_HALF)]
)
@pytest.mark.parametrize('case_cfg', _CASE_CONFIGS_2D, ids=[c['name'] for c in _CASE_CONFIGS_2D])
def test_20_cpu_vs_cpu_fallback_forward_backward(
	dtype: torch.dtype, keys: list[str], case_cfg: dict
):
	legacy_ref_cpu = _run_legacy_reference_impl(dtype, torch.device('cpu'), case_cfg)
	tri_cpu = _run_triton_impl(dtype, torch.device('cpu'), case_cfg)
	_compare_result_dicts(legacy_ref_cpu, tri_cpu, keys, dtype)


@pytest.mark.parametrize(
	'dtype,keys', [(torch.float32, _PAIRWISE_OPS_FP32[:3]), (torch.float16, _PAIRWISE_OPS_HALF[:3])]
)
@pytest.mark.parametrize('case_cfg', _CASE_CONFIGS_2D, ids=[c['name'] for c in _CASE_CONFIGS_2D])
@pytest.mark.parametrize('inverse,subm', _CONV_FLAG_CASES)
def test_21_cpu_legacy_reference_vs_triton_conv_flag_modes(
	dtype: torch.dtype,
	keys: list[str],
	case_cfg: dict,
	inverse: int,
	subm: int,
):
	legacy_ref_cpu = _run_legacy_reference_conv_family_with_flags(
		dtype, torch.device('cpu'), case_cfg, inverse, subm
	)
	tri_cpu = _run_triton_conv_family_with_flags(
		dtype, torch.device('cpu'), case_cfg, inverse, subm
	)
	_compare_result_dicts(legacy_ref_cpu, tri_cpu, keys, dtype)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is required')
@pytest.mark.parametrize(
	'dtype,keys', [(torch.float32, _PAIRWISE_OPS_FP32), (torch.float16, _PAIRWISE_OPS_HALF)]
)
@pytest.mark.parametrize('case_cfg', _CASE_CONFIGS_2D, ids=[c['name'] for c in _CASE_CONFIGS_2D])
def test_30_cuda_vs_triton_forward_backward(dtype: torch.dtype, keys: list[str], case_cfg: dict):
	legacy_ref_cuda = _run_legacy_reference_impl(dtype, torch.device('cuda'), case_cfg)
	tri_cuda = _run_triton_impl(dtype, torch.device('cuda'), case_cfg)
	_compare_result_dicts(legacy_ref_cuda, tri_cuda, keys, dtype)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is required')
@pytest.mark.parametrize(
	'dtype,keys', [(torch.float32, _PAIRWISE_OPS_FP32), (torch.float16, _PAIRWISE_OPS_HALF)]
)
@pytest.mark.parametrize('case_cfg', _CASE_CONFIGS_2D, ids=[c['name'] for c in _CASE_CONFIGS_2D])
def test_35_cpu_reference_vs_cuda_reference_forward_backward(
	dtype: torch.dtype,
	keys: list[str],
	case_cfg: dict,
):
	_run_spconv_legacy_reference_cpu_vs_legacy_reference_cuda(dtype, case_cfg, keys)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is required')
@pytest.mark.parametrize(
	'dtype,keys', [(torch.float32, _PAIRWISE_OPS_FP32), (torch.float16, _PAIRWISE_OPS_HALF)]
)
@pytest.mark.parametrize('case_cfg', _CASE_CONFIGS_2D, ids=[c['name'] for c in _CASE_CONFIGS_2D])
def test_40_cpu_vs_triton_forward_backward(dtype: torch.dtype, keys: list[str], case_cfg: dict):
	legacy_ref_cpu = _run_legacy_reference_impl(dtype, torch.device('cpu'), case_cfg)
	tri_cuda = _to_cpu_result_dict(_run_triton_impl(dtype, torch.device('cuda'), case_cfg))
	_compare_result_dicts(legacy_ref_cpu, tri_cuda, keys, dtype)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is required')
@pytest.mark.parametrize(
	'dtype,keys', [(torch.float32, _PAIRWISE_OPS_FP32), (torch.float16, _PAIRWISE_OPS_HALF)]
)
@pytest.mark.parametrize('case_cfg', _CASE_CONFIGS_2D, ids=[c['name'] for c in _CASE_CONFIGS_2D])
def test_45_cpu_fallback_vs_triton_forward_backward(
	dtype: torch.dtype,
	keys: list[str],
	case_cfg: dict,
):
	_run_spconv_triton_cpu_vs_triton_cuda(dtype, case_cfg, keys)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is required')
@pytest.mark.parametrize(
	'dtype,keys', [(torch.float32, _PAIRWISE_OPS_FP32), (torch.float16, _PAIRWISE_OPS_HALF)]
)
@pytest.mark.parametrize('case_cfg', _CASE_CONFIGS_2D, ids=[c['name'] for c in _CASE_CONFIGS_2D])
def test_50_cpu_fallback_vs_cuda_forward_backward(
	dtype: torch.dtype, keys: list[str], case_cfg: dict
):
	tri_cpu = _run_triton_impl(dtype, torch.device('cpu'), case_cfg)
	legacy_ref_cuda = _to_cpu_result_dict(
		_run_legacy_reference_impl(dtype, torch.device('cuda'), case_cfg)
	)
	_compare_result_dicts(tri_cpu, legacy_ref_cuda, keys, dtype)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is required')
@pytest.mark.parametrize(
	'dtype,keys', [(torch.float32, _PAIRWISE_OPS_FP32[:3]), (torch.float16, _PAIRWISE_OPS_HALF[:3])]
)
@pytest.mark.parametrize('case_cfg', _CASE_CONFIGS_2D, ids=[c['name'] for c in _CASE_CONFIGS_2D])
@pytest.mark.parametrize('inverse,subm', _CONV_FLAG_CASES)
def test_60_cuda_legacy_reference_vs_triton_conv_flag_modes(
	dtype: torch.dtype,
	keys: list[str],
	case_cfg: dict,
	inverse: int,
	subm: int,
):
	legacy_ref_cuda = _run_legacy_reference_conv_family_with_flags(
		dtype, torch.device('cuda'), case_cfg, inverse, subm
	)
	tri_cuda = _run_triton_conv_family_with_flags(
		dtype, torch.device('cuda'), case_cfg, inverse, subm
	)
	_compare_result_dicts(legacy_ref_cuda, tri_cuda, keys, dtype)
