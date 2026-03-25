from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from torch.amp.autocast_mode import custom_bwd, custom_fwd
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from utils.lazy_inline_extension import LazyInlineExtension

_CSRC_DIR = Path(__file__).resolve().parent / 'csrc'
_INCLUDE_DIR = Path(__file__).resolve().parent / 'include'
_COMMON_HELPER_DIR = _CSRC_DIR.parent.parent / 'common_helper' / 'csrc'
_EXT_EXPORTED_FUNCTIONS = [
	'sigmoid_focal_loss_forward',
	'sigmoid_focal_loss_backward',
]


ext_module = LazyInlineExtension(
	exported_names=_EXT_EXPORTED_FUNCTIONS,
	name='sigmoid_focal_loss_ext',
	source_dir=_CSRC_DIR,
	cpp_filenames=['sigmoid_focal_loss.cpp'],
	cuda_filenames=['sigmoid_focal_loss_cuda.cu'],
	extra_include_paths=[_CSRC_DIR, _INCLUDE_DIR, _COMMON_HELPER_DIR],
	with_cuda=True,
	verbose=False,
)


class SigmoidFocalLossFunction(Function):
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
		assert target.dtype == torch.long
		assert input.dim() == 2
		assert target.dim() == 1
		assert input.size(0) == target.size(0)

		input = input.contiguous()
		target = target.contiguous()

		if weight is None:
			weight = input.new_empty(0)
		else:
			assert weight.is_cuda, 'weight must be CUDA tensor'
			assert weight.dim() == 1
			assert input.size(1) == weight.size(0)
			weight = weight.contiguous()

		ctx.reduction_dict = {'none': 0, 'mean': 1, 'sum': 2}
		assert reduction in ctx.reduction_dict

		ctx.gamma = float(gamma)
		ctx.alpha = float(alpha)
		ctx.reduction = ctx.reduction_dict[reduction]

		output = input.new_zeros(input.size())
		ext_module.sigmoid_focal_loss_forward(
			input,
			target,
			weight,
			output,
			gamma=ctx.gamma,
			alpha=ctx.alpha,
		)

		if ctx.reduction == ctx.reduction_dict['mean']:
			output = output.sum() / input.size(0)
		elif ctx.reduction == ctx.reduction_dict['sum']:
			output = output.sum()

		ctx.save_for_backward(input, target, weight)
		return output

	@staticmethod
	@custom_bwd(device_type='cuda')
	@once_differentiable
	def backward(ctx, grad_output: torch.Tensor) -> tuple:
		input, target, weight = ctx.saved_tensors

		grad_input = input.new_zeros(input.size())
		ext_module.sigmoid_focal_loss_backward(
			input,
			target,
			weight,
			grad_input,
			gamma=ctx.gamma,
			alpha=ctx.alpha,
		)

		grad_input *= grad_output
		if ctx.reduction == ctx.reduction_dict['mean']:
			grad_input /= input.size(0)

		return grad_input, None, None, None, None, None


sigmoid_focal_loss = SigmoidFocalLossFunction.apply
