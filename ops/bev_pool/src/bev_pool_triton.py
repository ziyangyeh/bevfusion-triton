from __future__ import annotations

import torch
import triton
import triton.language as tl


_BLOCK_N = 64
_BLOCK_C = 64


def _check_inputs(
	feats: torch.Tensor,
	coords: torch.Tensor,
	B: int,
	D: int,
	H: int,
	W: int,
) -> None:
	if feats.dim() != 2:
		raise ValueError('feats must have shape [N, C].')
	if coords.dim() != 2 or coords.size(-1) != 4:
		raise ValueError('coords must have shape [N, 4].')
	if feats.size(0) != coords.size(0):
		raise ValueError('feats and coords must have the same number of points.')
	if coords.dtype not in (torch.int32, torch.int64):
		raise TypeError('coords must have dtype torch.int32 or torch.int64.')
	if feats.dtype not in (torch.float16, torch.float32, torch.bfloat16):
		raise TypeError('feats must have dtype torch.float16, torch.float32, or torch.bfloat16.')
	if min(B, D, H, W) <= 0:
		raise ValueError('B, D, H, W must be positive integers.')


def _flatten_ranks(coords: torch.Tensor, B: int, D: int, H: int, W: int) -> torch.Tensor:
	x = coords[:, 0].long()
	y = coords[:, 1].long()
	z = coords[:, 2].long()
	b = coords[:, 3].long()
	return ((b * D + z) * H + x) * W + y


def _bev_pool_cpu(
	feats: torch.Tensor, coords: torch.Tensor, B: int, D: int, H: int, W: int
) -> torch.Tensor:
	ranks = _flatten_ranks(coords, B, D, H, W)
	out = feats.new_zeros((B * D * H * W, feats.size(1)))
	out.index_add_(0, ranks, feats)
	return out.view(B, D, H, W, feats.size(1))


@triton.jit
def _bev_pool_forward_kernel(
	feats_ptr,
	coords_ptr,
	out_ptr,
	N,
	C,
	D,
	H,
	W,
	stride_feats_n,
	stride_feats_c,
	stride_coords_n,
	stride_out_b,
	stride_out_d,
	stride_out_h,
	stride_out_w,
	stride_out_c,
	BLOCK_N: tl.constexpr,
	BLOCK_C: tl.constexpr,
):
	pid_n = tl.program_id(axis=0)
	pid_c = tl.program_id(axis=1)

	n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
	c_offsets = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
	mask_n = n_offsets < N
	mask_c = c_offsets < C

	x = tl.load(coords_ptr + n_offsets * stride_coords_n + 0, mask=mask_n, other=0).to(tl.int32)
	y = tl.load(coords_ptr + n_offsets * stride_coords_n + 1, mask=mask_n, other=0).to(tl.int32)
	z = tl.load(coords_ptr + n_offsets * stride_coords_n + 2, mask=mask_n, other=0).to(tl.int32)
	b = tl.load(coords_ptr + n_offsets * stride_coords_n + 3, mask=mask_n, other=0).to(tl.int32)

	feat_ptrs = (
		feats_ptr + n_offsets[:, None] * stride_feats_n + c_offsets[None, :] * stride_feats_c
	)
	feat_mask = mask_n[:, None] & mask_c[None, :]
	values = tl.load(feat_ptrs, mask=feat_mask, other=0)

	out_ptrs = (
		out_ptr
		+ b[:, None] * stride_out_b
		+ z[:, None] * stride_out_d
		+ x[:, None] * stride_out_h
		+ y[:, None] * stride_out_w
		+ c_offsets[None, :] * stride_out_c
	)
	tl.atomic_add(out_ptrs, values, mask=feat_mask)


@triton.jit
def _bev_pool_backward_kernel(
	out_grad_ptr,
	coords_ptr,
	x_grad_ptr,
	N,
	C,
	D,
	H,
	W,
	stride_coords_n,
	stride_out_b,
	stride_out_c,
	stride_out_d,
	stride_out_h,
	stride_out_w,
	stride_x_grad_n,
	stride_x_grad_c,
	BLOCK_N: tl.constexpr,
	BLOCK_C: tl.constexpr,
):
	pid_n = tl.program_id(axis=0)
	pid_c = tl.program_id(axis=1)

	n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
	c_offsets = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
	mask_n = n_offsets < N
	mask_c = c_offsets < C

	x = tl.load(coords_ptr + n_offsets * stride_coords_n + 0, mask=mask_n, other=0).to(tl.int32)
	y = tl.load(coords_ptr + n_offsets * stride_coords_n + 1, mask=mask_n, other=0).to(tl.int32)
	z = tl.load(coords_ptr + n_offsets * stride_coords_n + 2, mask=mask_n, other=0).to(tl.int32)
	b = tl.load(coords_ptr + n_offsets * stride_coords_n + 3, mask=mask_n, other=0).to(tl.int32)

	out_grad_ptrs = (
		out_grad_ptr
		+ b[:, None] * stride_out_b
		+ c_offsets[None, :] * stride_out_c
		+ z[:, None] * stride_out_d
		+ x[:, None] * stride_out_h
		+ y[:, None] * stride_out_w
	)
	mask = mask_n[:, None] & mask_c[None, :]
	values = tl.load(out_grad_ptrs, mask=mask, other=0)

	x_grad_ptrs = (
		x_grad_ptr + n_offsets[:, None] * stride_x_grad_n + c_offsets[None, :] * stride_x_grad_c
	)
	tl.store(x_grad_ptrs, values, mask=mask)


class _BevPoolTritonFunction(torch.autograd.Function):
	@staticmethod
	def forward(ctx, feats, coords, B, D, H, W):
		coords_i32 = coords.contiguous().to(torch.int32)
		feats_contig = feats.contiguous()
		out = feats.new_zeros((B, D, H, W, feats.size(1)))
		if feats.numel() > 0:
			grid = (
				triton.cdiv(feats.size(0), _BLOCK_N),
				triton.cdiv(feats.size(1), _BLOCK_C),
			)
			_bev_pool_forward_kernel[grid](
				feats_contig,
				coords_i32,
				out,
				feats.size(0),
				feats.size(1),
				D,
				H,
				W,
				feats_contig.stride(0),
				feats_contig.stride(1),
				coords_i32.stride(0),
				out.stride(0),
				out.stride(1),
				out.stride(2),
				out.stride(3),
				out.stride(4),
				BLOCK_N=_BLOCK_N,
				BLOCK_C=_BLOCK_C,
			)
		ctx.save_for_backward(coords_i32)
		ctx.saved_shape = (feats.size(0), feats.size(1), B, D, H, W)
		return out

	@staticmethod
	def backward(ctx, out_grad):
		(coords_i32,) = ctx.saved_tensors
		N, C, B, D, H, W = ctx.saved_shape
		out_grad = out_grad.contiguous()
		x_grad = out_grad.new_empty((N, C))
		if x_grad.numel() > 0:
			grid = (triton.cdiv(N, _BLOCK_N), triton.cdiv(C, _BLOCK_C))
			_bev_pool_backward_kernel[grid](
				out_grad,
				coords_i32,
				x_grad,
				N,
				C,
				D,
				H,
				W,
				coords_i32.stride(0),
				out_grad.stride(0),
				out_grad.stride(4),
				out_grad.stride(1),
				out_grad.stride(2),
				out_grad.stride(3),
				x_grad.stride(0),
				x_grad.stride(1),
				BLOCK_N=_BLOCK_N,
				BLOCK_C=_BLOCK_C,
			)
		return x_grad, None, None, None, None, None


def bev_pool(
	feats: torch.Tensor, coords: torch.Tensor, B: int, D: int, H: int, W: int
) -> torch.Tensor:
	_check_inputs(feats, coords, B, D, H, W)
	if feats.is_cuda:
		out = _BevPoolTritonFunction.apply(feats, coords, B, D, H, W)
	else:
		out = _bev_pool_cpu(feats, coords, B, D, H, W)
	return out.permute(0, 4, 1, 2, 3).contiguous()
