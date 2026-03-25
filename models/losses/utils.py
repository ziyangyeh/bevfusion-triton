import functools

import torch
import torch.nn.functional as F


def reduce_loss(loss: torch.Tensor, reduction: str) -> torch.Tensor:
	reduction_enum = F._Reduction.get_enum(reduction)
	if reduction_enum == 0:
		return loss
	if reduction_enum == 1:
		return loss.mean()
	if reduction_enum == 2:
		return loss.sum()
	raise ValueError(f'Unsupported reduction: {reduction}')


def weight_reduce_loss(
	loss: torch.Tensor,
	weight: torch.Tensor | None = None,
	reduction: str = 'mean',
	avg_factor: float | None = None,
) -> torch.Tensor:
	if weight is not None:
		loss = loss * weight

	if avg_factor is None:
		return reduce_loss(loss, reduction)

	if reduction == 'mean':
		eps = torch.finfo(loss.dtype if loss.is_floating_point() else torch.float32).eps
		return loss.sum() / (avg_factor + eps)
	if reduction != 'none':
		raise ValueError('avg_factor can not be used with reduction="sum"')
	return loss


def weighted_loss(loss_func):
	@functools.wraps(loss_func)
	def wrapper(
		pred,
		target,
		weight=None,
		reduction: str = 'mean',
		avg_factor: float | None = None,
		**kwargs,
	):
		loss = loss_func(pred, target, **kwargs)
		return weight_reduce_loss(loss, weight, reduction, avg_factor)

	return wrapper
