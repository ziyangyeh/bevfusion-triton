from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pytest
import torch
import triton
import triton.language as tl
from torch.utils.cpp_extension import load_inline

from ops.common_helper.src.common_triton_helper import (
	bilinear_interpolate_gradient_kernel,
	bilinear_interpolate_kernel,
)
from tests.test_ops._eps import DTYPE_EPS

_BLOCK_SIZE = 128
_COMMON_HELPER_DIR = Path(__file__).resolve().parents[2] / 'ops' / 'common_helper' / 'csrc'

_CPP_SOURCE = r"""
#include <torch/extension.h>

torch::Tensor bilinear_interpolate_cuda(
    torch::Tensor planes,
    torch::Tensor y,
    torch::Tensor x,
    torch::Tensor plane_ids);

std::vector<torch::Tensor> bilinear_interpolate_gradient_cuda(
    torch::Tensor y,
    torch::Tensor x,
    int64_t height,
    int64_t width);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("bilinear_interpolate_cuda", &bilinear_interpolate_cuda);
  m.def("bilinear_interpolate_gradient_cuda", &bilinear_interpolate_gradient_cuda);
}
"""

_CUDA_SOURCE = r"""
#include <torch/extension.h>

#include "common_cuda_helper.hpp"

namespace {

template <typename scalar_t>
__global__ void bilinear_interpolate_test_kernel(
    const scalar_t* planes,
    const scalar_t* y,
    const scalar_t* x,
    const int32_t* plane_ids,
    scalar_t* out,
    int64_t height,
    int64_t width,
    int64_t numel) {
  CUDA_1D_KERNEL_LOOP(index, numel) {
    const scalar_t* plane_ptr = planes + static_cast<int64_t>(plane_ids[index]) * height * width;
    out[index] = bilinear_interpolate(
        plane_ptr, static_cast<int>(height), static_cast<int>(width), y[index], x[index], index);
  }
}

template <typename scalar_t>
__global__ void bilinear_interpolate_gradient_test_kernel(
    const scalar_t* y,
    const scalar_t* x,
    scalar_t* w1,
    scalar_t* w2,
    scalar_t* w3,
    scalar_t* w4,
    int32_t* y_low,
    int32_t* x_low,
    int32_t* y_high,
    int32_t* x_high,
    int64_t height,
    int64_t width,
    int64_t numel) {
  CUDA_1D_KERNEL_LOOP(index, numel) {
    int x_low_val = 0;
    int x_high_val = 0;
    int y_low_val = 0;
    int y_high_val = 0;
    bilinear_interpolate_gradient(
        static_cast<int>(height),
        static_cast<int>(width),
        y[index],
        x[index],
        w1[index],
        w2[index],
        w3[index],
        w4[index],
        x_low_val,
        x_high_val,
        y_low_val,
        y_high_val,
        index);
    y_low[index] = y_low_val;
    x_low[index] = x_low_val;
    y_high[index] = y_high_val;
    x_high[index] = x_high_val;
  }
}

}  // namespace

torch::Tensor bilinear_interpolate_cuda(
    torch::Tensor planes,
    torch::Tensor y,
    torch::Tensor x,
    torch::Tensor plane_ids) {
  TORCH_CHECK(planes.is_cuda(), "planes must be CUDA");
  TORCH_CHECK(y.is_cuda(), "y must be CUDA");
  TORCH_CHECK(x.is_cuda(), "x must be CUDA");
  TORCH_CHECK(plane_ids.is_cuda(), "plane_ids must be CUDA");
  TORCH_CHECK(planes.dim() == 3, "planes must have shape [P, H, W]");
  TORCH_CHECK(y.sizes() == x.sizes(), "y and x must have the same shape");
  TORCH_CHECK(y.sizes() == plane_ids.sizes(), "plane_ids must match y/x shape");

  auto out = torch::zeros_like(y);
  const auto numel = y.numel();
  const auto blocks = GET_BLOCKS(numel);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      planes.scalar_type(),
      "bilinear_interpolate_cuda",
      [&] {
        bilinear_interpolate_test_kernel<scalar_t><<<blocks, THREADS_PER_BLOCK>>>(
            planes.data_ptr<scalar_t>(),
            y.data_ptr<scalar_t>(),
            x.data_ptr<scalar_t>(),
            plane_ids.data_ptr<int32_t>(),
            out.data_ptr<scalar_t>(),
            planes.size(1),
            planes.size(2),
            numel);
      });

  return out;
}

std::vector<torch::Tensor> bilinear_interpolate_gradient_cuda(
    torch::Tensor y,
    torch::Tensor x,
    int64_t height,
    int64_t width) {
  TORCH_CHECK(y.is_cuda(), "y must be CUDA");
  TORCH_CHECK(x.is_cuda(), "x must be CUDA");
  TORCH_CHECK(y.sizes() == x.sizes(), "y and x must have the same shape");

  auto w1 = torch::zeros_like(y);
  auto w2 = torch::zeros_like(y);
  auto w3 = torch::zeros_like(y);
  auto w4 = torch::zeros_like(y);
  auto y_low = torch::empty(y.sizes(), y.options().dtype(torch::kInt32));
  auto x_low = torch::empty(x.sizes(), y.options().dtype(torch::kInt32));
  auto y_high = torch::empty(x.sizes(), y.options().dtype(torch::kInt32));
  auto x_high = torch::empty(x.sizes(), y.options().dtype(torch::kInt32));
  const auto numel = y.numel();
  const auto blocks = GET_BLOCKS(numel);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      y.scalar_type(),
      "bilinear_interpolate_gradient_cuda",
      [&] {
        bilinear_interpolate_gradient_test_kernel<scalar_t><<<blocks, THREADS_PER_BLOCK>>>(
            y.data_ptr<scalar_t>(),
            x.data_ptr<scalar_t>(),
            w1.data_ptr<scalar_t>(),
            w2.data_ptr<scalar_t>(),
            w3.data_ptr<scalar_t>(),
            w4.data_ptr<scalar_t>(),
            y_low.data_ptr<int32_t>(),
            x_low.data_ptr<int32_t>(),
            y_high.data_ptr<int32_t>(),
            x_high.data_ptr<int32_t>(),
            height,
            width,
            numel);
      });

  return {w1, w2, w3, w4, y_low, x_low, y_high, x_high};
}
"""


def _skip_if_bfloat16_unsupported(dtype: torch.dtype):
	if not torch.cuda.is_available():
		pytest.skip('CUDA is required')
	if dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
		pytest.skip('CUDA bfloat16 is required')


@lru_cache(maxsize=1)
def _load_common_helper_test_extension():
	return load_inline(
		name='common_helper_test_ext',
		cpp_sources=[_CPP_SOURCE],
		cuda_sources=[_CUDA_SOURCE],
		functions=None,
		extra_include_paths=[str(_COMMON_HELPER_DIR)],
		with_cuda=True,
		verbose=False,
	)


def _make_test_inputs(dtype: torch.dtype):
	planes = torch.tensor(
		[
			[
				[0.5, -1.0, 2.0, 3.0, -0.5],
				[1.0, 4.0, -2.0, 0.5, 2.5],
				[3.0, -4.0, 1.5, 2.0, 1.0],
				[0.0, 2.0, -1.5, 4.5, -3.0],
			],
			[
				[-2.0, 1.0, 0.0, 3.5, 2.0],
				[4.0, -1.0, 2.5, -2.5, 0.5],
				[1.5, 3.0, -3.5, 2.0, 1.0],
				[2.0, -0.5, 1.0, -4.0, 5.0],
			],
		],
		dtype=dtype,
		device='cuda',
	)
	y = torch.tensor([-1.5, -0.2, 0.3, 1.6, 2.8, 3.2, 4.1], dtype=dtype, device='cuda')
	x = torch.tensor([-1.3, 0.0, 1.2, 2.7, 3.9, 4.0, 5.2], dtype=dtype, device='cuda')
	plane_ids = torch.tensor([0, 0, 0, 1, 1, 1, 1], dtype=torch.int32, device='cuda')
	return planes, y, x, plane_ids


@triton.jit
def _bilinear_interpolate_test_kernel(
	plane_ptr,
	y_ptr,
	x_ptr,
	plane_id_ptr,
	out_ptr,
	height,
	width,
	numel,
	BLOCK_SIZE: tl.constexpr,
):
	offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
	mask = offsets < numel

	y = tl.load(y_ptr + offsets, mask=mask, other=0)
	x = tl.load(x_ptr + offsets, mask=mask, other=0)
	plane_id = tl.load(plane_id_ptr + offsets, mask=mask, other=0)
	plane_stride = height * width
	plane_base_ptr = plane_ptr + plane_id * plane_stride

	val = bilinear_interpolate_kernel(plane_base_ptr, y, x, height, width)
	tl.store(out_ptr + offsets, val, mask=mask)


@triton.jit
def _bilinear_interpolate_gradient_test_kernel(
	y_ptr,
	x_ptr,
	w1_ptr,
	w2_ptr,
	w3_ptr,
	w4_ptr,
	y_low_ptr,
	x_low_ptr,
	y_high_ptr,
	x_high_ptr,
	height,
	width,
	numel,
	BLOCK_SIZE: tl.constexpr,
):
	offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
	mask = offsets < numel

	y = tl.load(y_ptr + offsets, mask=mask, other=0)
	x = tl.load(x_ptr + offsets, mask=mask, other=0)
	w1, w2, w3, w4, y_low, x_low, y_high, x_high = bilinear_interpolate_gradient_kernel(
		y, x, height, width
	)
	tl.store(w1_ptr + offsets, w1, mask=mask)
	tl.store(w2_ptr + offsets, w2, mask=mask)
	tl.store(w3_ptr + offsets, w3, mask=mask)
	tl.store(w4_ptr + offsets, w4, mask=mask)
	tl.store(y_low_ptr + offsets, y_low, mask=mask)
	tl.store(x_low_ptr + offsets, x_low, mask=mask)
	tl.store(y_high_ptr + offsets, y_high, mask=mask)
	tl.store(x_high_ptr + offsets, x_high, mask=mask)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is required')
@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
def test_bilinear_interpolate_kernel_matches_cuda_helper(dtype: torch.dtype):
	_skip_if_bfloat16_unsupported(dtype)
	ext = _load_common_helper_test_extension()
	planes, y, x, plane_ids = _make_test_inputs(dtype)

	triton_out = torch.empty_like(y)
	grid = lambda meta: (triton.cdiv(y.numel(), meta['BLOCK_SIZE']),)
	_bilinear_interpolate_test_kernel[grid](
		planes,
		y,
		x,
		plane_ids,
		triton_out,
		planes.shape[1],
		planes.shape[2],
		y.numel(),
		BLOCK_SIZE=_BLOCK_SIZE,
	)

	cuda_out = ext.bilinear_interpolate_cuda(planes, y, x, plane_ids)

	rtol, atol = DTYPE_EPS[dtype]
	assert torch.allclose(triton_out.float(), cuda_out.float(), rtol=rtol, atol=atol)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is required')
@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
def test_bilinear_interpolate_gradient_kernel_matches_cuda_helper(dtype: torch.dtype):
	_skip_if_bfloat16_unsupported(dtype)
	ext = _load_common_helper_test_extension()
	planes, y, x, _ = _make_test_inputs(dtype)

	w1 = torch.empty_like(y)
	w2 = torch.empty_like(y)
	w3 = torch.empty_like(y)
	w4 = torch.empty_like(y)
	y_low = torch.empty(y.numel(), device='cuda', dtype=torch.int32)
	x_low = torch.empty(y.numel(), device='cuda', dtype=torch.int32)
	y_high = torch.empty(y.numel(), device='cuda', dtype=torch.int32)
	x_high = torch.empty(y.numel(), device='cuda', dtype=torch.int32)

	grid = lambda meta: (triton.cdiv(y.numel(), meta['BLOCK_SIZE']),)
	_bilinear_interpolate_gradient_test_kernel[grid](
		y,
		x,
		w1,
		w2,
		w3,
		w4,
		y_low,
		x_low,
		y_high,
		x_high,
		planes.shape[1],
		planes.shape[2],
		y.numel(),
		BLOCK_SIZE=_BLOCK_SIZE,
	)

	cuda_w1, cuda_w2, cuda_w3, cuda_w4, cuda_y_low, cuda_x_low, cuda_y_high, cuda_x_high = (
		ext.bilinear_interpolate_gradient_cuda(y, x, planes.shape[1], planes.shape[2])
	)

	rtol, atol = DTYPE_EPS[dtype]
	assert torch.allclose(w1.float(), cuda_w1.float(), rtol=rtol, atol=atol)
	assert torch.allclose(w2.float(), cuda_w2.float(), rtol=rtol, atol=atol)
	assert torch.allclose(w3.float(), cuda_w3.float(), rtol=rtol, atol=atol)
	assert torch.allclose(w4.float(), cuda_w4.float(), rtol=rtol, atol=atol)
	assert torch.equal(y_low, cuda_y_low)
	assert torch.equal(x_low, cuda_x_low)
	assert torch.equal(y_high, cuda_y_high)
	assert torch.equal(x_high, cuda_x_high)
