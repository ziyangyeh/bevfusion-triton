from __future__ import annotations

import torch
import triton
import triton.language as tl


_CHANNEL_BLOCK = 128


@triton.jit
def _sparse_gather_forward_kernel(
	features_ptr,
	indices_ptr,
	output_ptr,
	num_channels,
	num_rows,
	num_channel_blocks,
	BLOCK_SIZE: tl.constexpr,
):
	pid = tl.program_id(axis=0)
	if pid >= num_rows * num_channel_blocks:
		return
	row = pid // num_channel_blocks
	channel_block = pid % num_channel_blocks
	col = channel_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
	mask = col < num_channels
	src_row = tl.load(indices_ptr + row).to(tl.int64)
	src_offsets = src_row * num_channels + col
	values = tl.load(features_ptr + src_offsets, mask=mask, other=0.0)
	tl.store(output_ptr + row * num_channels + col, values, mask=mask)


@triton.jit
def _sparse_scatter_add_kernel(
	src_ptr,
	indices_ptr,
	out_ptr,
	num_channels,
	num_rows,
	num_channel_blocks,
	BLOCK_SIZE: tl.constexpr,
):
	pid = tl.program_id(axis=0)
	if pid >= num_rows * num_channel_blocks:
		return
	row = pid // num_channel_blocks
	channel_block = pid % num_channel_blocks
	col = channel_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
	mask = col < num_channels
	dst_row = tl.load(indices_ptr + row).to(tl.int64)
	dst_offsets = dst_row * num_channels + col
	src = tl.load(src_ptr + row * num_channels + col, mask=mask, other=0.0)
	tl.atomic_add(out_ptr + dst_offsets, src, mask=mask)


def sparse_gather_cpu_fallback(features: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
	return features.index_select(0, indices.to(torch.long))


def sparse_scatter_add_cpu_fallback(
	out_features: torch.Tensor,
	buffer: torch.Tensor,
	indices: torch.Tensor,
) -> torch.Tensor:
	out_features.index_add_(0, indices.to(torch.long), buffer)
	return out_features


def sparse_gather(features: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
	assert features.dim() == 2
	assert indices.dim() == 1

	if features.is_cuda:
		features = features.contiguous()
		indices = indices.contiguous()
		output = torch.empty(
			(indices.numel(), features.size(1)),
			device=features.device,
			dtype=features.dtype,
		)
		num_rows = int(indices.numel())
		if num_rows > 0:
			num_channel_blocks = triton.cdiv(features.size(1), _CHANNEL_BLOCK)
			grid = lambda meta: (num_rows * num_channel_blocks,)
			_sparse_gather_forward_kernel[grid](
				features,
				indices,
				output,
				features.size(1),
				num_rows,
				num_channel_blocks,
				BLOCK_SIZE=_CHANNEL_BLOCK,
			)
		return output

	return sparse_gather_cpu_fallback(features, indices)


def sparse_scatter_add(
	out_features: torch.Tensor,
	buffer: torch.Tensor,
	indices: torch.Tensor,
) -> torch.Tensor:
	assert out_features.dim() == 2
	assert buffer.dim() == 2
	assert indices.dim() == 1
	assert buffer.size(0) == indices.numel()
	assert buffer.size(1) == out_features.size(1)

	if out_features.is_cuda:
		out_features = out_features.contiguous()
		buffer = buffer.contiguous()
		indices = indices.contiguous()
		num_rows = int(indices.numel())
		if num_rows > 0:
			num_channel_blocks = triton.cdiv(out_features.size(1), _CHANNEL_BLOCK)
			grid = lambda meta: (num_rows * num_channel_blocks,)
			_sparse_scatter_add_kernel[grid](
				buffer,
				indices,
				out_features,
				out_features.size(1),
				num_rows,
				num_channel_blocks,
				BLOCK_SIZE=_CHANNEL_BLOCK,
			)
		return out_features

	return sparse_scatter_add_cpu_fallback(out_features, buffer, indices)


sparse_gather_triton = sparse_gather
sparse_scatter_add_triton = sparse_scatter_add

__all__ = [
	'sparse_gather',
	'sparse_gather_triton',
	'sparse_gather_cpu_fallback',
	'sparse_scatter_add',
	'sparse_scatter_add_triton',
	'sparse_scatter_add_cpu_fallback',
]
