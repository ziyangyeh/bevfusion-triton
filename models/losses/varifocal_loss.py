import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import weight_reduce_loss


def varifocal_loss(
	pred: torch.Tensor,
	target: torch.Tensor,
	weight: torch.Tensor | None = None,
	alpha: float = 0.75,
	gamma: float = 2.0,
	iou_weighted: bool = True,
	reduction: str = 'mean',
	avg_factor: float | None = None,
) -> torch.Tensor:
	assert pred.size() == target.size()
	pred_sigmoid = pred.sigmoid()
	target = target.type_as(pred)
	if iou_weighted:
		focal_weight = (
			target * (target > 0.0).float()
			+ alpha * (pred_sigmoid - target).abs().pow(gamma) * (target <= 0.0).float()
		)
	else:
		focal_weight = (target > 0.0).float() + alpha * (pred_sigmoid - target).abs().pow(gamma) * (
			target <= 0.0
		).float()
	loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none') * focal_weight
	return weight_reduce_loss(loss, weight, reduction, avg_factor)


class VarifocalLoss(nn.Module):
	def __init__(
		self,
		use_sigmoid: bool = True,
		alpha: float = 0.75,
		gamma: float = 2.0,
		iou_weighted: bool = True,
		reduction: str = 'mean',
		loss_weight: float = 1.0,
	) -> None:
		super().__init__()
		assert use_sigmoid is True, 'Only sigmoid varifocal loss supported now.'
		assert alpha >= 0.0
		self.use_sigmoid = use_sigmoid
		self.alpha = alpha
		self.gamma = gamma
		self.iou_weighted = iou_weighted
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
		return self.loss_weight * varifocal_loss(
			pred,
			target,
			weight,
			alpha=self.alpha,
			gamma=self.gamma,
			iou_weighted=self.iou_weighted,
			reduction=reduction,
			avg_factor=avg_factor,
		)
