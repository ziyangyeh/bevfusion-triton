import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import weight_reduce_loss

try:
	from ops.sigmoid_focal_loss import sigmoid_focal_loss as _sigmoid_focal_loss
except Exception:
	_sigmoid_focal_loss = None


def _reshape_weight(loss: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
	if weight.shape != loss.shape:
		if weight.size(0) == loss.size(0):
			weight = weight.view(-1, 1)
		else:
			assert weight.numel() == loss.numel()
			weight = weight.view(loss.size(0), -1)
	assert weight.ndim == loss.ndim
	return weight


def py_sigmoid_focal_loss(
	pred: torch.Tensor,
	target: torch.Tensor,
	weight: torch.Tensor | None = None,
	gamma: float = 2.0,
	alpha: float = 0.25,
	reduction: str = 'mean',
	avg_factor: float | None = None,
) -> torch.Tensor:
	pred_sigmoid = pred.sigmoid()
	target = target.type_as(pred)
	pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
	focal_weight = (alpha * target + (1 - alpha) * (1 - target)) * pt.pow(gamma)
	loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none') * focal_weight
	if weight is not None:
		weight = _reshape_weight(loss, weight)
	return weight_reduce_loss(loss, weight, reduction, avg_factor)


def py_focal_loss_with_prob(
	pred: torch.Tensor,
	target: torch.Tensor,
	weight: torch.Tensor | None = None,
	gamma: float = 2.0,
	alpha: float = 0.25,
	reduction: str = 'mean',
	avg_factor: float | None = None,
) -> torch.Tensor:
	if pred.dim() != target.dim():
		num_classes = pred.size(1)
		target = F.one_hot(target, num_classes=num_classes + 1)[..., :num_classes]

	target = target.type_as(pred)
	pt = (1 - pred) * target + pred * (1 - target)
	focal_weight = (alpha * target + (1 - alpha) * (1 - target)) * pt.pow(gamma)
	loss = F.binary_cross_entropy(pred, target, reduction='none') * focal_weight
	if weight is not None:
		weight = _reshape_weight(loss, weight)
	return weight_reduce_loss(loss, weight, reduction, avg_factor)


def sigmoid_focal_loss(
	pred: torch.Tensor,
	target: torch.Tensor,
	weight: torch.Tensor | None = None,
	gamma: float = 2.0,
	alpha: float = 0.25,
	reduction: str = 'mean',
	avg_factor: float | None = None,
) -> torch.Tensor:
	if _sigmoid_focal_loss is None:
		return py_sigmoid_focal_loss(pred, target, weight, gamma, alpha, reduction, avg_factor)

	loss = _sigmoid_focal_loss(pred.contiguous(), target.contiguous(), gamma, alpha, None, 'none')
	if weight is not None:
		weight = _reshape_weight(loss, weight)
	return weight_reduce_loss(loss, weight, reduction, avg_factor)


class FocalLoss(nn.Module):
	def __init__(
		self,
		use_sigmoid: bool = True,
		gamma: float = 2.0,
		alpha: float = 0.25,
		reduction: str = 'mean',
		loss_weight: float = 1.0,
		activated: bool = False,
	) -> None:
		super().__init__()
		assert use_sigmoid is True, 'Only sigmoid focal loss supported now.'
		self.use_sigmoid = use_sigmoid
		self.gamma = gamma
		self.alpha = alpha
		self.reduction = reduction
		self.loss_weight = loss_weight
		self.activated = activated

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
		if self.activated:
			loss_func = py_focal_loss_with_prob
		else:
			if pred.dim() == target.dim():
				loss_func = py_sigmoid_focal_loss
			elif pred.is_cuda and _sigmoid_focal_loss is not None:
				loss_func = sigmoid_focal_loss
			else:
				num_classes = pred.size(1)
				target = F.one_hot(target, num_classes=num_classes + 1)[..., :num_classes]
				loss_func = py_sigmoid_focal_loss

		return self.loss_weight * loss_func(
			pred,
			target,
			weight,
			gamma=self.gamma,
			alpha=self.alpha,
			reduction=reduction,
			avg_factor=avg_factor,
		)
