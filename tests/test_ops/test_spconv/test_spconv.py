from __future__ import annotations

from contextlib import contextmanager

import pytest
import torch

import ops.spconv as spconv
from ops.spconv import ops as ops_mod
from ops.spconv import ops_legacy as legacy_reference_ops
from ops.spconv.conv import SparseConv2d as SparseConv2dImpl
from ops.spconv.conv import SparseConv3d as SparseConv3dImpl
from ops.spconv.conv import SparseConvTranspose2d as SparseConvTranspose2dImpl
from ops.spconv.conv import SparseConvTranspose3d as SparseConvTranspose3dImpl
from ops.spconv.conv import SparseInverseConv2d as SparseInverseConv2dImpl
from ops.spconv.conv import SparseInverseConv3d as SparseInverseConv3dImpl
from ops.spconv.conv import SubMConv2d as SubMConv2dImpl
from ops.spconv.conv import SubMConv3d as SubMConv3dImpl
from ops.spconv.modules import SparseModule as SparseModuleImpl
from ops.spconv.modules import SparseSequential as SparseSequentialImpl
from ops.spconv.pool import SparseMaxPool2d as SparseMaxPool2dImpl
from ops.spconv.pool import SparseMaxPool3d as SparseMaxPool3dImpl
from ops.spconv.src import spconv_triton
from ops.spconv.structure import SparseConvTensor, scatter_nd
from tests.test_ops._eps import DTYPE_EPS

_EXPECTED_ALL = [
	'SparseConv2d',
	'SparseConv3d',
	'SubMConv2d',
	'SubMConv3d',
	'SparseConvTranspose2d',
	'SparseConvTranspose3d',
	'SparseInverseConv2d',
	'SparseInverseConv3d',
	'SparseModule',
	'SparseSequential',
	'SparseMaxPool2d',
	'SparseMaxPool3d',
	'SparseConvTensor',
	'scatter_nd',
]

_INPUT_CASES_2D = [
	{
		'name': 'full_2x2',
		'indices': [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1]],
		'spatial_shape': [2, 2],
		'batch_size': 1,
	},
	{
		'name': 'sparse_3x3',
		'indices': [[0, 0, 0], [0, 0, 2], [0, 1, 1], [0, 2, 0], [0, 2, 2]],
		'spatial_shape': [3, 3],
		'batch_size': 1,
	},
]

_INPUT_CASES_3D = [
	{
		'name': 'full_2x2x2',
		'indices': [
			[0, 0, 0, 0],
			[0, 0, 0, 1],
			[0, 0, 1, 0],
			[0, 0, 1, 1],
			[0, 1, 0, 0],
			[0, 1, 0, 1],
			[0, 1, 1, 0],
			[0, 1, 1, 1],
		],
		'spatial_shape': [2, 2, 2],
		'batch_size': 1,
	},
	{
		'name': 'sparse_3x3x2',
		'indices': [[0, 0, 0, 0], [0, 0, 1, 2], [0, 1, 2, 1], [0, 1, 0, 2], [0, 0, 2, 0]],
		'spatial_shape': [2, 3, 3],
		'batch_size': 1,
	},
]

_SPARSE_CONV_CASES_2D = [
	{'in_channels': 3, 'out_channels': 4, 'kernel_size': 1, 'bias': True},
	{'in_channels': 3, 'out_channels': 2, 'kernel_size': 1, 'bias': False},
	{'in_channels': 3, 'out_channels': 2, 'kernel_size': 3, 'padding': 1, 'bias': True},
]

_SUBM_CONV_CASES_2D = [
	{'in_channels': 3, 'out_channels': 4, 'kernel_size': 1, 'bias': True},
	{'in_channels': 3, 'out_channels': 2, 'kernel_size': 1, 'bias': False},
	{'in_channels': 3, 'out_channels': 4, 'kernel_size': 3, 'padding': 1, 'bias': True},
]

_SPARSE_CONV_CASES_3D = [
	{'in_channels': 3, 'out_channels': 4, 'kernel_size': 1, 'bias': True},
	{'in_channels': 3, 'out_channels': 2, 'kernel_size': 1, 'bias': False},
	{'in_channels': 3, 'out_channels': 2, 'kernel_size': 3, 'padding': 1, 'bias': True},
]

_SUBM_CONV_CASES_3D = [
	{'in_channels': 3, 'out_channels': 4, 'kernel_size': 1, 'bias': True},
	{'in_channels': 3, 'out_channels': 2, 'kernel_size': 1, 'bias': False},
	{'in_channels': 3, 'out_channels': 4, 'kernel_size': 3, 'padding': 1, 'bias': True},
]

_MAXPOOL_CASES_2D = [
	{'kernel_size': 1, 'stride': 1, 'padding': 0},
	{'kernel_size': 2, 'stride': 1, 'padding': 1},
]

_MAXPOOL_CASES_3D = [
	{'kernel_size': 1, 'stride': 1, 'padding': 0},
	{'kernel_size': 2, 'stride': 1, 'padding': 1},
]


def _assert_close(a: torch.Tensor, b: torch.Tensor, dtype: torch.dtype):
	rtol, atol = DTYPE_EPS[dtype]
	assert torch.allclose(a, b, rtol=rtol, atol=atol)


@contextmanager
def _patch_ops_backend(use_legacy_reference: bool):
	names = [
		'get_indice_pairs',
		'indice_conv',
		'fused_indice_conv',
		'indice_conv_backward',
		'indice_maxpool',
		'indice_maxpool_backward',
	]
	saved = {n: getattr(ops_mod, n) for n in names}

	try:
		src = legacy_reference_ops if use_legacy_reference else spconv_triton
		for n in names:
			setattr(ops_mod, n, getattr(src, n))
		yield
	finally:
		for n, v in saved.items():
			setattr(ops_mod, n, v)


def _build_sparse_input(
	device: torch.device,
	in_channels: int,
	dtype: torch.dtype,
	input_case: dict | None = None,
) -> SparseConvTensor:
	case = _INPUT_CASES_2D[0] if input_case is None else input_case
	indices = torch.tensor(case['indices'], dtype=torch.int32, device=device)
	features = torch.randn(
		indices.size(0), in_channels, dtype=dtype, device=device, requires_grad=True
	)
	return SparseConvTensor(
		features, indices, spatial_shape=case['spatial_shape'], batch_size=case['batch_size']
	)


def _run_module_once(module, input_tensor: SparseConvTensor, use_legacy_reference: bool):
	x = SparseConvTensor(
		input_tensor.features.clone().detach().requires_grad_(True),
		input_tensor.indices,
		input_tensor.spatial_shape,
		input_tensor.batch_size,
	)

	with _patch_ops_backend(use_legacy_reference=use_legacy_reference):
		out = module(x)
		upstream = torch.ones_like(out.features).contiguous()
		loss = (out.features * upstream).sum()
		loss.backward()

	result = {
		'out': out.features.detach(),
		'in_grad': x.features.grad.detach(),
	}
	if hasattr(module, 'weight') and module.weight is not None and module.weight.grad is not None:
		result['weight_grad'] = module.weight.grad.detach().clone()
	if hasattr(module, 'bias') and module.bias is not None and module.bias.grad is not None:
		result['bias_grad'] = module.bias.grad.detach().clone()
	return result


def _compare_module_legacy_reference_vs_triton(
	module_ctor,
	ctor_kwargs: dict,
	device: torch.device,
	dtype: torch.dtype,
	check_backward: bool = True,
	input_case: dict | None = None,
):
	torch.manual_seed(123)
	module_cpp = module_ctor(**ctor_kwargs).to(device=device, dtype=dtype)
	module_tri = module_ctor(**ctor_kwargs).to(device=device, dtype=dtype)
	module_tri.load_state_dict(module_cpp.state_dict())

	input_tensor = _build_sparse_input(
		device,
		in_channels=ctor_kwargs.get('in_channels', 3),
		dtype=dtype,
		input_case=input_case,
	)

	module_cpp.zero_grad(set_to_none=True)
	res_cpp = _run_module_once(module_cpp, input_tensor, use_legacy_reference=True)

	module_tri.zero_grad(set_to_none=True)
	res_tri = _run_module_once(module_tri, input_tensor, use_legacy_reference=False)

	_assert_close(res_cpp['out'], res_tri['out'], dtype)
	if check_backward:
		_assert_close(res_cpp['in_grad'], res_tri['in_grad'], dtype)
		if 'weight_grad' in res_cpp:
			_assert_close(res_cpp['weight_grad'], res_tri['weight_grad'], dtype)
		if 'bias_grad' in res_cpp:
			_assert_close(res_cpp['bias_grad'], res_tri['bias_grad'], dtype)


def test_00_spconv_init_exports_exist():
	for name in spconv.__all__:
		assert hasattr(spconv, name), f'missing export: {name}'


def test_00b_spconv_init_exports_exact():
	assert spconv.__all__ == _EXPECTED_ALL


def test_00c_spconv_init_symbol_binding():
	assert spconv.SparseConv2d is SparseConv2dImpl
	assert spconv.SparseConv3d is SparseConv3dImpl
	assert spconv.SubMConv2d is SubMConv2dImpl
	assert spconv.SubMConv3d is SubMConv3dImpl
	assert spconv.SparseConvTranspose2d is SparseConvTranspose2dImpl
	assert spconv.SparseConvTranspose3d is SparseConvTranspose3dImpl
	assert spconv.SparseInverseConv2d is SparseInverseConv2dImpl
	assert spconv.SparseInverseConv3d is SparseInverseConv3dImpl
	assert spconv.SparseModule is SparseModuleImpl
	assert spconv.SparseSequential is SparseSequentialImpl
	assert spconv.SparseMaxPool2d is SparseMaxPool2dImpl
	assert spconv.SparseMaxPool3d is SparseMaxPool3dImpl
	assert spconv.SparseConvTensor is SparseConvTensor
	assert spconv.scatter_nd is scatter_nd


def test_00d_spconv_init_constructibility():
	conv2d = spconv.SparseConv2d(3, 4, 3, stride=1, padding=1)
	conv3d = spconv.SparseConv3d(3, 4, 3, stride=1, padding=1)
	subm2d = spconv.SubMConv2d(3, 4, 3)
	subm3d = spconv.SubMConv3d(3, 4, 3)
	deconv2d = spconv.SparseConvTranspose2d(3, 4, 3, stride=1, padding=1)
	deconv3d = spconv.SparseConvTranspose3d(3, 4, 3, stride=1, padding=1)
	inv2d = spconv.SparseInverseConv2d(3, 4, 3, indice_key='k2d')
	inv3d = spconv.SparseInverseConv3d(3, 4, 3, indice_key='k3d')
	pool2d = spconv.SparseMaxPool2d(1, stride=1, padding=0)
	pool3d = spconv.SparseMaxPool3d(1, stride=1, padding=0)
	seq = spconv.SparseSequential(conv2d)

	assert isinstance(conv2d, spconv.SparseModule)
	assert isinstance(conv3d, spconv.SparseModule)
	assert isinstance(subm2d, spconv.SparseModule)
	assert isinstance(subm3d, spconv.SparseModule)
	assert isinstance(deconv2d, spconv.SparseModule)
	assert isinstance(deconv3d, spconv.SparseModule)
	assert isinstance(inv2d, spconv.SparseModule)
	assert isinstance(inv3d, spconv.SparseModule)
	assert isinstance(pool2d, spconv.SparseModule)
	assert isinstance(pool3d, spconv.SparseModule)
	assert isinstance(seq, spconv.SparseSequential)


def test_00e_spconv_init_sparse_utils():
	idx = torch.tensor([[0, 0], [1, 1]], dtype=torch.long)
	updates = torch.tensor([1.0, 2.0], dtype=torch.float32)
	dense = spconv.scatter_nd(idx, updates, [2, 2])
	assert dense.shape == (2, 2)
	assert dense[0, 0].item() == 1.0
	assert dense[1, 1].item() == 2.0

	feats = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
	inds = torch.tensor([[0, 0, 0], [0, 0, 1]], dtype=torch.int32)
	st = spconv.SparseConvTensor(feats, inds, spatial_shape=[1, 2], batch_size=1)
	dense_cf = st.dense(channels_first=True)
	dense_cl = st.dense(channels_first=False)
	assert dense_cf.shape == (1, 2, 1, 2)
	assert dense_cl.shape == (1, 1, 2, 2)


@pytest.mark.parametrize(
	'ctor_kwargs',
	_SPARSE_CONV_CASES_2D,
)
@pytest.mark.parametrize('input_case', _INPUT_CASES_2D, ids=[c['name'] for c in _INPUT_CASES_2D])
def test_spconv_init_sparseconv2d_cpu_legacy_reference_vs_triton(
	ctor_kwargs: dict, input_case: dict
):
	_compare_module_legacy_reference_vs_triton(
		spconv.SparseConv2d,
		ctor_kwargs,
		torch.device('cpu'),
		torch.float32,
		check_backward=True,
		input_case=input_case,
	)


@pytest.mark.parametrize(
	'ctor_kwargs',
	_MAXPOOL_CASES_2D,
)
@pytest.mark.parametrize('input_case', _INPUT_CASES_2D, ids=[c['name'] for c in _INPUT_CASES_2D])
def test_spconv_init_sparsemaxpool2d_cpu_legacy_reference_vs_triton(
	ctor_kwargs: dict, input_case: dict
):
	_compare_module_legacy_reference_vs_triton(
		spconv.SparseMaxPool2d,
		ctor_kwargs,
		torch.device('cpu'),
		torch.float32,
		check_backward=False,
		input_case=input_case,
	)


@pytest.mark.parametrize(
	'ctor_kwargs',
	_SUBM_CONV_CASES_2D,
)
@pytest.mark.parametrize('input_case', _INPUT_CASES_2D, ids=[c['name'] for c in _INPUT_CASES_2D])
def test_spconv_init_submconv2d_cpu_legacy_reference_vs_triton(ctor_kwargs: dict, input_case: dict):
	_compare_module_legacy_reference_vs_triton(
		spconv.SubMConv2d,
		ctor_kwargs,
		torch.device('cpu'),
		torch.float32,
		check_backward=True,
		input_case=input_case,
	)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is required')
@pytest.mark.parametrize('dtype', [torch.float32, torch.float16])
@pytest.mark.parametrize(
	'ctor_kwargs',
	_SPARSE_CONV_CASES_2D,
)
@pytest.mark.parametrize('input_case', _INPUT_CASES_2D, ids=[c['name'] for c in _INPUT_CASES_2D])
def test_spconv_init_sparseconv2d_cuda_legacy_reference_vs_triton(
	dtype, ctor_kwargs: dict, input_case: dict
):
	_compare_module_legacy_reference_vs_triton(
		spconv.SparseConv2d,
		ctor_kwargs,
		torch.device('cuda'),
		dtype,
		check_backward=True,
		input_case=input_case,
	)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is required')
@pytest.mark.parametrize('dtype', [torch.float32, torch.float16])
@pytest.mark.parametrize(
	'ctor_kwargs',
	_MAXPOOL_CASES_2D,
)
@pytest.mark.parametrize('input_case', _INPUT_CASES_2D, ids=[c['name'] for c in _INPUT_CASES_2D])
def test_spconv_init_sparsemaxpool2d_cuda_legacy_reference_vs_triton(
	dtype, ctor_kwargs: dict, input_case: dict
):
	_compare_module_legacy_reference_vs_triton(
		spconv.SparseMaxPool2d,
		ctor_kwargs,
		torch.device('cuda'),
		dtype,
		check_backward=False,
		input_case=input_case,
	)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is required')
@pytest.mark.parametrize('dtype', [torch.float32, torch.float16])
@pytest.mark.parametrize(
	'ctor_kwargs',
	_SUBM_CONV_CASES_2D,
)
@pytest.mark.parametrize('input_case', _INPUT_CASES_2D, ids=[c['name'] for c in _INPUT_CASES_2D])
def test_spconv_init_submconv2d_cuda_legacy_reference_vs_triton(
	dtype, ctor_kwargs: dict, input_case: dict
):
	_compare_module_legacy_reference_vs_triton(
		spconv.SubMConv2d,
		ctor_kwargs,
		torch.device('cuda'),
		dtype,
		check_backward=True,
		input_case=input_case,
	)


@pytest.mark.parametrize('ctor_kwargs', _SPARSE_CONV_CASES_3D)
@pytest.mark.parametrize('input_case', _INPUT_CASES_3D, ids=[c['name'] for c in _INPUT_CASES_3D])
def test_spconv_init_sparseconv3d_cpu_legacy_reference_vs_triton(
	ctor_kwargs: dict, input_case: dict
):
	_compare_module_legacy_reference_vs_triton(
		spconv.SparseConv3d,
		ctor_kwargs,
		torch.device('cpu'),
		torch.float32,
		check_backward=True,
		input_case=input_case,
	)


@pytest.mark.parametrize('ctor_kwargs', _MAXPOOL_CASES_3D)
@pytest.mark.parametrize('input_case', _INPUT_CASES_3D, ids=[c['name'] for c in _INPUT_CASES_3D])
def test_spconv_init_sparsemaxpool3d_cpu_legacy_reference_vs_triton(
	ctor_kwargs: dict, input_case: dict
):
	_compare_module_legacy_reference_vs_triton(
		spconv.SparseMaxPool3d,
		ctor_kwargs,
		torch.device('cpu'),
		torch.float32,
		check_backward=False,
		input_case=input_case,
	)


@pytest.mark.parametrize('ctor_kwargs', _SUBM_CONV_CASES_3D)
@pytest.mark.parametrize('input_case', _INPUT_CASES_3D, ids=[c['name'] for c in _INPUT_CASES_3D])
def test_spconv_init_submconv3d_cpu_legacy_reference_vs_triton(ctor_kwargs: dict, input_case: dict):
	_compare_module_legacy_reference_vs_triton(
		spconv.SubMConv3d,
		ctor_kwargs,
		torch.device('cpu'),
		torch.float32,
		check_backward=True,
		input_case=input_case,
	)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is required')
@pytest.mark.parametrize('dtype', [torch.float32, torch.float16])
@pytest.mark.parametrize('ctor_kwargs', _SPARSE_CONV_CASES_3D)
@pytest.mark.parametrize('input_case', _INPUT_CASES_3D, ids=[c['name'] for c in _INPUT_CASES_3D])
def test_spconv_init_sparseconv3d_cuda_legacy_reference_vs_triton(
	dtype, ctor_kwargs: dict, input_case: dict
):
	_compare_module_legacy_reference_vs_triton(
		spconv.SparseConv3d,
		ctor_kwargs,
		torch.device('cuda'),
		dtype,
		check_backward=True,
		input_case=input_case,
	)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is required')
@pytest.mark.parametrize('dtype', [torch.float32, torch.float16])
@pytest.mark.parametrize('ctor_kwargs', _MAXPOOL_CASES_3D)
@pytest.mark.parametrize('input_case', _INPUT_CASES_3D, ids=[c['name'] for c in _INPUT_CASES_3D])
def test_spconv_init_sparsemaxpool3d_cuda_legacy_reference_vs_triton(
	dtype, ctor_kwargs: dict, input_case: dict
):
	_compare_module_legacy_reference_vs_triton(
		spconv.SparseMaxPool3d,
		ctor_kwargs,
		torch.device('cuda'),
		dtype,
		check_backward=False,
		input_case=input_case,
	)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is required')
@pytest.mark.parametrize('dtype', [torch.float32, torch.float16])
@pytest.mark.parametrize('ctor_kwargs', _SUBM_CONV_CASES_3D)
@pytest.mark.parametrize('input_case', _INPUT_CASES_3D, ids=[c['name'] for c in _INPUT_CASES_3D])
def test_spconv_init_submconv3d_cuda_legacy_reference_vs_triton(
	dtype, ctor_kwargs: dict, input_case: dict
):
	_compare_module_legacy_reference_vs_triton(
		spconv.SubMConv3d,
		ctor_kwargs,
		torch.device('cuda'),
		dtype,
		check_backward=True,
		input_case=input_case,
	)
