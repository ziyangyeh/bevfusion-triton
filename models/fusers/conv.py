from typing import List

import torch
from torch import nn

__all__ = ['ConvFuser']


class ConvFuser(nn.Sequential):
	def __init__(self, in_channels=(80, 256), out_channels=256) -> None:
		self.in_channels = tuple(in_channels)
		self.out_channels = out_channels
		super().__init__(
			nn.Conv2d(sum(self.in_channels), out_channels, 3, padding=1, bias=False),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(True),
		)

	def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
		# Concatenate multi-modal BEV features along the channel dimension.
		return super().forward(torch.cat(inputs, dim=1))


if __name__ == '__main__':
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model = ConvFuser().to(device).eval()

	inputs = [
		torch.randn(2, 80, 128, 128, device=device),
		torch.randn(2, 256, 128, 128, device=device),
	]

	with torch.no_grad():
		output = model(inputs)

	print(output.shape)
