import torch
import torch.nn as nn

from .utils import weighted_loss


@weighted_loss
def gaussian_focal_loss(
	pred: torch.Tensor,
	gaussian_target: torch.Tensor,
	alpha: float = 2.0,
	gamma: float = 4.0,
) -> torch.Tensor:
	eps = 1e-12
	pos_weights = gaussian_target.eq(1)
	neg_weights = (1 - gaussian_target).pow(gamma)
	pos_loss = -(pred + eps).log() * (1 - pred).pow(alpha) * pos_weights
	neg_loss = -(1 - pred + eps).log() * pred.pow(alpha) * neg_weights
	return pos_loss + neg_loss


class GaussianFocalLoss(nn.Module):
	def __init__(
		self,
		alpha: float = 2.0,
		gamma: float = 4.0,
		reduction: str = 'mean',
		loss_weight: float = 1.0,
	) -> None:
		super().__init__()
		self.alpha = alpha
		self.gamma = gamma
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
		return self.loss_weight * gaussian_focal_loss(
			pred,
			target,
			weight,
			alpha=self.alpha,
			gamma=self.gamma,
			reduction=reduction,
			avg_factor=avg_factor,
		)
