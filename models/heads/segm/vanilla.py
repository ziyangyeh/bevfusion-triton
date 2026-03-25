from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

__all__ = ['BEVGridTransform', 'BEVSegmentationHead']


def sigmoid_xent_loss(
	inputs: torch.Tensor,
	targets: torch.Tensor,
	reduction: str = 'mean',
) -> torch.Tensor:
	return F.binary_cross_entropy_with_logits(inputs.float(), targets.float(), reduction=reduction)


def sigmoid_focal_loss(
	inputs: torch.Tensor,
	targets: torch.Tensor,
	alpha: float = -1.0,
	gamma: float = 2.0,
	reduction: str = 'mean',
) -> torch.Tensor:
	inputs = inputs.float()
	targets = targets.float()
	probs = torch.sigmoid(inputs)
	ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
	probs_t = probs * targets + (1 - probs) * (1 - targets)
	loss = ce_loss * ((1 - probs_t) ** gamma)

	if alpha >= 0:
		alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
		loss = alpha_t * loss

	if reduction == 'mean':
		return loss.mean()
	if reduction == 'sum':
		return loss.sum()
	return loss


class BEVGridTransform(nn.Module):
	def __init__(
		self,
		*,
		input_scope: list[tuple[float, float, float]],
		output_scope: list[tuple[float, float, float]],
		prescale_factor: float = 1.0,
	) -> None:
		super().__init__()
		self.input_scope = input_scope
		self.output_scope = output_scope
		self.prescale_factor = prescale_factor

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		if self.prescale_factor != 1:
			x = F.interpolate(
				x,
				scale_factor=self.prescale_factor,
				mode='bilinear',
				align_corners=False,
			)

		coords = []
		for (imin, imax, _), (omin, omax, ostep) in zip(self.input_scope, self.output_scope):
			coord = torch.arange(omin + ostep / 2, omax, ostep, device=x.device, dtype=x.dtype)
			coord = (coord - imin) / (imax - imin) * 2 - 1
			coords.append(coord)

		u, v = torch.meshgrid(coords, indexing='ij')
		grid = torch.stack([v, u], dim=-1)
		grid = torch.stack([grid] * x.shape[0], dim=0)

		return F.grid_sample(x, grid, mode='bilinear', align_corners=False)


class BEVSegmentationHead(nn.Module):
	def __init__(
		self,
		in_channels: int,
		grid_transform: dict[str, Any],
		classes: list[str],
		loss: str = 'xent',
	) -> None:
		super().__init__()
		self.in_channels = in_channels
		self.classes = list(classes)
		self.loss = loss

		self.transform = BEVGridTransform(**grid_transform)
		self.classifier = nn.Sequential(
			nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
			nn.BatchNorm2d(in_channels),
			nn.ReLU(True),
			nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
			nn.BatchNorm2d(in_channels),
			nn.ReLU(True),
			nn.Conv2d(in_channels, len(classes), 1),
		)

	def forward(
		self,
		x: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...],
		target: torch.Tensor | None = None,
	) -> torch.Tensor | dict[str, torch.Tensor]:
		if isinstance(x, (list, tuple)):
			x = x[0]

		logits = self.classifier(self.transform(x))

		if self.training:
			if target is None:
				raise ValueError('BEVSegmentationHead expects gt_masks_bev during training.')
			losses: dict[str, torch.Tensor] = {}
			for index, name in enumerate(self.classes):
				if self.loss == 'xent':
					loss = sigmoid_xent_loss(logits[:, index], target[:, index])
				elif self.loss == 'focal':
					loss = sigmoid_focal_loss(logits[:, index], target[:, index])
				else:
					raise ValueError(f'unsupported loss: {self.loss}')
				losses[f'{name}/{self.loss}'] = loss
			losses['loss'] = torch.stack(list(losses.values())).sum()
			return losses

		return torch.sigmoid(logits)
