import torch
import torch.nn.functional as F
from torch import nn as nn

from .utils import weight_reduce_loss


def _expand_onehot_labels(labels, label_weights, label_channels, ignore_index):
	"""Expand onehot labels to match the size of prediction."""
	bin_labels = labels.new_full((labels.size(0), label_channels), 0)
	valid_mask = (labels >= 0) & (labels != ignore_index)
	inds = torch.nonzero(valid_mask & (labels < label_channels), as_tuple=False)

	if inds.numel() > 0:
		bin_labels[inds, labels[inds]] = 1

	valid_mask = valid_mask.view(-1, 1).expand(labels.size(0), label_channels).float()
	if label_weights is None:
		bin_label_weights = valid_mask
	else:
		bin_label_weights = label_weights.view(-1, 1).repeat(1, label_channels)
		bin_label_weights *= valid_mask

	return bin_labels, bin_label_weights


def binary_cross_entropy(
	pred,
	label,
	weight=None,
	reduction='mean',
	avg_factor=None,
	class_weight=None,
	ignore_index=-100,
):
	"""Calculate the binary CrossEntropy loss.

	Args:
	    pred (torch.Tensor): The prediction with shape (N, 1).
	    label (torch.Tensor): The learning label of the prediction.
	    weight (torch.Tensor, optional): Sample-wise loss weight.
	    reduction (str, optional): The method used to reduce the loss.
	        Options are "none", "mean" and "sum".
	    avg_factor (int, optional): Average factor that is used to average
	        the loss. Defaults to None.
	    class_weight (list[float], optional): The weight for each class.
	    ignore_index (int | None): The label index to be ignored.
	        If None, it will be set to default value. Default: -100.

	Returns:
	    torch.Tensor: The calculated loss.
	"""
	# The default value of ignore_index is the same as F.cross_entropy
	ignore_index = -100 if ignore_index is None else ignore_index
	if pred.dim() != label.dim():
		label, weight = _expand_onehot_labels(label, weight, pred.size(-1), ignore_index)

	# weighted element-wise losses
	if weight is not None:
		weight = weight.float()
	loss = F.binary_cross_entropy_with_logits(
		pred, label.float(), pos_weight=class_weight, reduction='none'
	)
	# do the reduction for the weighted loss
	loss = weight_reduce_loss(loss, weight, reduction=reduction, avg_factor=avg_factor)

	return loss
