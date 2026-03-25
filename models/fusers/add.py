from typing import List

import torch
from einops import rearrange
from torch import nn

__all__ = ['AddFuser']


class AddFuser(nn.Module):
	def __init__(self, in_channels=(80, 256), out_channels=256, dropout: float = 0.0) -> None:
		super().__init__()
		self.in_channels = tuple(in_channels)
		self.out_channels = out_channels
		self.dropout = dropout

		self.transforms = nn.ModuleList(
			[
				nn.Sequential(
					nn.Conv2d(channels, out_channels, 3, padding=1, bias=False),
					nn.BatchNorm2d(out_channels),
					nn.ReLU(True),
				)
				for channels in self.in_channels
			]
		)

	def _dropout_weights(self, num_inputs: int, device: torch.device) -> torch.Tensor:
		weights = torch.ones(num_inputs, device=device)
		if self.training and torch.rand((), device=device) < self.dropout:
			drop_idx = torch.randint(num_inputs, (), device=device)
			weights[drop_idx] = 0
		return weights

	def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
		# Project each modality to a shared channel space before fusion.
		features = [transform(x) for transform, x in zip(self.transforms, inputs)]
		stacked = torch.stack(features, dim=0)

		# Average the available modality features after optional dropout.
		weight_tensor = rearrange(
			self._dropout_weights(len(inputs), stacked.device), 'm -> m 1 1 1 1'
		)
		return (stacked * weight_tensor).sum(dim=0) / weight_tensor.sum()


if __name__ == '__main__':
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model = AddFuser().to(device).eval()

	inputs = [
		torch.randn(2, 80, 128, 128, device=device),
		torch.randn(2, 256, 128, 128, device=device),
	]

	with torch.no_grad():
		output = model(inputs)

	print(output.shape)
