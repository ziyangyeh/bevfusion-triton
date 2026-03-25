from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl
from torch.amp.autocast_mode import custom_bwd, custom_fwd
from torch.autograd import Function
from torch.autograd.function import once_differentiable

_BLOCK_SIZE = 256
_REDUCTION_TO_ID = {'none': 0, 'mean': 1, 'sum': 2}


def _sigmoid_focal_loss_eps(dtype: torch.dtype) -> float:
	return float(torch.finfo(dtype).tiny)


@triton.jit
def _sigmoid_focal_loss_forward_kernel(
	input_ptr,
	target_ptr,
	weight_ptr,
	output_ptr,
	gamma,
	alpha,
	eps,
	num_classes,
	numel,
	has_weight: tl.constexpr,
	BLOCK_SIZE: tl.constexpr,
):
	pid = tl.program_id(axis=0)
	offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
	mask = offsets < numel

	x_native = tl.load(input_ptr + offsets, mask=mask, other=0.0)
	x = x_native.to(tl.float32)
	n = offsets // num_classes
	c = offsets % num_classes

	t = tl.load(target_ptr + n, mask=mask, other=0).to(tl.int64)
	flag_p = (t == c).to(tl.float32)
	flag_n = 1.0 - flag_p

	p = tl.sigmoid(x)
	pow_1mp = tl.exp(gamma * tl.log(tl.maximum(1.0 - p, eps)))
	term_p = pow_1mp * tl.log(tl.maximum(p, eps))
	pow_p = tl.exp(gamma * tl.log(tl.maximum(p, eps)))
	term_n = pow_p * tl.log(tl.maximum(1.0 - p, eps))

	out = tl.fma(-flag_p * alpha, term_p, -flag_n * (1.0 - alpha) * term_n)
	if has_weight:
		wt = tl.load(weight_ptr + t, mask=mask, other=1.0).to(tl.float32)
		out = out * wt

	tl.store(output_ptr + offsets, out.to(x_native.dtype), mask=mask)


@triton.jit
def _sigmoid_focal_loss_backward_kernel(
	input_ptr,
	target_ptr,
	weight_ptr,
	grad_input_ptr,
	gamma,
	alpha,
	eps,
	num_classes,
	numel,
	has_weight: tl.constexpr,
	BLOCK_SIZE: tl.constexpr,
):
	pid = tl.program_id(axis=0)
	offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
	mask = offsets < numel

	x_native = tl.load(input_ptr + offsets, mask=mask, other=0.0)
	x = x_native.to(tl.float32)
	n = offsets // num_classes
	c = offsets % num_classes

	t = tl.load(target_ptr + n, mask=mask, other=0).to(tl.int64)
	flag_p = (t == c).to(tl.float32)
	flag_n = 1.0 - flag_p

	p = tl.sigmoid(x)
	pow_1mp = tl.exp(gamma * tl.log(tl.maximum(1.0 - p, eps)))
	term_p = pow_1mp * (1.0 - p - gamma * p * tl.log(tl.maximum(p, eps)))
	pow_p = tl.exp(gamma * tl.log(tl.maximum(p, eps)))
	term_n = pow_p * (gamma * (1.0 - p) * tl.log(tl.maximum(1.0 - p, eps)) - p)

	grad = tl.fma(-flag_p * alpha, term_p, -flag_n * (1.0 - alpha) * term_n)
	if has_weight:
		wt = tl.load(weight_ptr + t, mask=mask, other=1.0).to(tl.float32)
		grad = grad * wt

	tl.store(grad_input_ptr + offsets, grad.to(x_native.dtype), mask=mask)


def sigmoid_focal_loss_cpu_fallback(
	input: torch.Tensor,
	target: torch.LongTensor,
	gamma: float = 2.0,
	alpha: float = 0.25,
	weight: Optional[torch.Tensor] = None,
	reduction: str = 'mean',
) -> torch.Tensor:
	n, c = input.shape
	one = torch.tensor(1.0, device=input.device, dtype=input.dtype)
	alpha_t = torch.tensor(alpha, device=input.device, dtype=input.dtype)
	eps = torch.tensor(
		_sigmoid_focal_loss_eps(input.dtype),
		device=input.device,
		dtype=input.dtype,
	)
	p = torch.sigmoid(input)

	cls = torch.arange(c, device=input.device).view(1, c)
	pos_mask = (target.view(n, 1) == cls).to(input.dtype)
	neg_mask = one - pos_mask

	term_p = ((one - p) ** gamma) * torch.log(torch.clamp_min(p, eps))
	term_n = (p**gamma) * torch.log(torch.clamp_min(one - p, eps))
	loss = -alpha_t * pos_mask * term_p - (one - alpha_t) * neg_mask * term_n

	if weight is not None:
		loss = loss * weight.gather(0, target).view(n, 1)

	if reduction == 'none':
		return loss
	if reduction == 'mean':
		return loss.sum() / n
	if reduction == 'sum':
		return loss.sum()
	raise ValueError(f'Unsupported reduction: {reduction}')


def _validate_sigmoid_focal_loss_inputs(
	input: torch.Tensor,
	target: torch.Tensor,
	weight: Optional[torch.Tensor],
) -> None:
	assert target.dtype == torch.long
	assert input.dim() == 2
	assert target.dim() == 1
	assert input.size(0) == target.size(0)
	if weight is not None:
		assert weight.dim() == 1
		assert weight.size(0) == input.size(1)


class SigmoidFocalLossTritonFunction(Function):
	@staticmethod
	@custom_fwd(device_type='cuda')
	def forward(
		ctx,
		input: torch.Tensor,
		target: torch.LongTensor,
		gamma: float = 2.0,
		alpha: float = 0.25,
		weight: Optional[torch.Tensor] = None,
		reduction: str = 'mean',
	) -> torch.Tensor:
		assert input.is_cuda, 'input must be CUDA tensor'
		assert target.is_cuda, 'target must be CUDA tensor'
		_validate_sigmoid_focal_loss_inputs(input, target, weight)

		input = input.contiguous()
		target = target.contiguous()

		if weight is None:
			weight = input.new_empty(0)
		else:
			assert weight.is_cuda, 'weight must be CUDA tensor'
			assert weight.dim() == 1
			assert input.size(1) == weight.size(0)
			weight = weight.contiguous()

		assert reduction in _REDUCTION_TO_ID
		reduction_id = _REDUCTION_TO_ID[reduction]

		out = torch.empty_like(input)
		numel = input.numel()
		if numel > 0:
			grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
			_sigmoid_focal_loss_forward_kernel[grid](
				input,
				target,
				weight,
				out,
				float(gamma),
				float(alpha),
				_sigmoid_focal_loss_eps(input.dtype),
				input.size(1),
				numel,
				has_weight=weight.numel() > 0,
				BLOCK_SIZE=_BLOCK_SIZE,
			)

		if reduction_id == _REDUCTION_TO_ID['mean']:
			out_reduced = out.sum() / input.size(0)
		elif reduction_id == _REDUCTION_TO_ID['sum']:
			out_reduced = out.sum()
		else:
			out_reduced = out

		ctx.gamma = float(gamma)
		ctx.alpha = float(alpha)
		ctx.reduction = reduction_id
		ctx.reduction_dict = _REDUCTION_TO_ID
		ctx.save_for_backward(input, target, weight)
		return out_reduced

	@staticmethod
	@custom_bwd(device_type='cuda')
	@once_differentiable
	def backward(ctx, grad_output: torch.Tensor) -> tuple:
		input, target, weight = ctx.saved_tensors

		grad_input = torch.empty_like(input)
		numel = input.numel()
		if numel > 0:
			grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
			_sigmoid_focal_loss_backward_kernel[grid](
				input,
				target,
				weight,
				grad_input,
				ctx.gamma,
				ctx.alpha,
				_sigmoid_focal_loss_eps(input.dtype),
				input.size(1),
				numel,
				has_weight=weight.numel() > 0,
				BLOCK_SIZE=_BLOCK_SIZE,
			)
		else:
			grad_input.zero_()

		grad_input *= grad_output
		if ctx.reduction == ctx.reduction_dict['mean']:
			grad_input /= input.size(0)

		return grad_input, None, None, None, None, None


def sigmoid_focal_loss(
	input: torch.Tensor,
	target: torch.LongTensor,
	gamma: float = 2.0,
	alpha: float = 0.25,
	weight: Optional[torch.Tensor] = None,
	reduction: str = 'mean',
) -> torch.Tensor:
	_validate_sigmoid_focal_loss_inputs(input, target, weight)

	if input.is_cuda:
		return SigmoidFocalLossTritonFunction.apply(input, target, gamma, alpha, weight, reduction)

	return sigmoid_focal_loss_cpu_fallback(input, target, gamma, alpha, weight, reduction)


sigmoid_focal_loss_triton = sigmoid_focal_loss

__all__ = [
	'sigmoid_focal_loss',
	'sigmoid_focal_loss_triton',
	'sigmoid_focal_loss_cpu_fallback',
	'SigmoidFocalLossTritonFunction',
]
