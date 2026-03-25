import torch
import torch.nn as nn

from .utils import weighted_loss


@weighted_loss
def l1_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
	if target.numel() == 0:
		return pred.sum() * 0
	assert pred.size() == target.size()
	return torch.abs(pred - target)


class L1Loss(nn.Module):
	def __init__(self, reduction: str = 'mean', loss_weight: float = 1.0) -> None:
		super().__init__()
		self.reduction = reduction
		self.loss_weight = loss_weight

	def forward(
		self,
		pred: torch.Tensor,
		target: torch.Tensor,
		weight: torch.Tensor | None = None,
		avg_factor: float | None = None,
		reduction_override: str | None = None,
	) -> torch.Tensor:
		assert reduction_override in (None, 'none', 'mean', 'sum')
		reduction = reduction_override or self.reduction
		return self.loss_weight * l1_loss(
			pred,
			target,
			weight,
			reduction=reduction,
			avg_factor=avg_factor,
		)
