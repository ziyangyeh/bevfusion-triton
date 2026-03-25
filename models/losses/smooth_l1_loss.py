import torch
import torch.nn as nn

from .utils import weighted_loss


@weighted_loss
def smooth_l1_loss(pred: torch.Tensor, target: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
	assert beta > 0
	if target.numel() == 0:
		return pred.sum() * 0
	assert pred.size() == target.size()
	diff = torch.abs(pred - target)
	return torch.where(diff < beta, 0.5 * diff * diff / beta, diff - 0.5 * beta)


class SmoothL1Loss(nn.Module):
	def __init__(
		self, beta: float = 1.0, reduction: str = 'mean', loss_weight: float = 1.0
	) -> None:
		super().__init__()
		self.beta = beta
		self.reduction = reduction
		self.loss_weight = loss_weight

	def forward(
		self,
		pred: torch.Tensor,
		target: torch.Tensor,
		weight: torch.Tensor | None = None,
		avg_factor: float | None = None,
		reduction_override: str | None = None,
		**kwargs,
	) -> torch.Tensor:
		assert reduction_override in (None, 'none', 'mean', 'sum')
		reduction = reduction_override or self.reduction
		return self.loss_weight * smooth_l1_loss(
			pred,
			target,
			weight,
			beta=self.beta,
			reduction=reduction,
			avg_factor=avg_factor,
			**kwargs,
		)
