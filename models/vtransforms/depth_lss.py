from typing import Tuple

import torch
from einops import rearrange, repeat
from torch import nn

from models.helper import force_fp32
from .base import BaseDepthTransform

__all__ = ['DepthLSSTransform']


class DepthLSSTransform(BaseDepthTransform):
	def __init__(
		self,
		in_channels: int = 256,
		out_channels: int = 80,
		image_size: Tuple[int, int] = (256, 704),
		feature_size: Tuple[int, int] = (32, 88),
		xbound: Tuple[float, float, float] = (-51.2, 51.2, 0.4),
		ybound: Tuple[float, float, float] = (-51.2, 51.2, 0.4),
		zbound: Tuple[float, float, float] = (-10.0, 10.0, 20.0),
		dbound: Tuple[float, float, float] = (1.0, 60.0, 0.5),
		downsample: int = 2,
	) -> None:
		super().__init__(
			in_channels=in_channels,
			out_channels=out_channels,
			image_size=image_size,
			feature_size=feature_size,
			xbound=xbound,
			ybound=ybound,
			zbound=zbound,
			dbound=dbound,
			add_depth_features=False,
		)
		self.dtransform = nn.Sequential(
			nn.Conv2d(1, 8, 1),
			nn.BatchNorm2d(8),
			nn.ReLU(True),
			nn.Conv2d(8, 32, 5, stride=4, padding=2),
			nn.BatchNorm2d(32),
			nn.ReLU(True),
			nn.Conv2d(32, 64, 5, stride=2, padding=2),
			nn.BatchNorm2d(64),
			nn.ReLU(True),
		)
		self.depthnet = nn.Sequential(
			nn.Conv2d(in_channels + 64, in_channels, 3, padding=1),
			nn.BatchNorm2d(in_channels),
			nn.ReLU(True),
			nn.Conv2d(in_channels, in_channels, 3, padding=1),
			nn.BatchNorm2d(in_channels),
			nn.ReLU(True),
			nn.Conv2d(in_channels, self.D + self.C, 1),
		)
		if downsample > 1:
			assert downsample == 2, downsample
			self.downsample = nn.Sequential(
				nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
				nn.BatchNorm2d(out_channels),
				nn.ReLU(True),
				nn.Conv2d(
					out_channels,
					out_channels,
					3,
					stride=downsample,
					padding=1,
					bias=False,
				),
				nn.BatchNorm2d(out_channels),
				nn.ReLU(True),
				nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
				nn.BatchNorm2d(out_channels),
				nn.ReLU(True),
			)
		else:
			self.downsample = nn.Identity()

	@force_fp32()
	def get_cam_feats(self, x, d, mats_dict=None):
		B, N, C, _, _ = x.shape

		d = rearrange(d, 'b n c h w -> (b n) c h w')
		x = rearrange(x, 'b n c h w -> (b n) c h w')

		d = self.dtransform(d)
		x = torch.cat([d, x], dim=1)
		x = self.depthnet(x)

		depth = x[:, : self.D].softmax(dim=1)
		x = depth.unsqueeze(1) * x[:, self.D : (self.D + self.C)].unsqueeze(2)

		return rearrange(x, '(b n) c d h w -> b n d h w c', b=B, n=N)

	def forward(self, *args, **kwargs):
		x = super().forward(*args, **kwargs)
		x = self.downsample(x)
		return x


if __name__ == '__main__':
	if not torch.cuda.is_available():
		print(
			'DepthLSSTransform demo expects CUDA for bev_pool; skipping forward on CPU-only environment.'
		)
	else:
		device = torch.device('cuda')
		model = DepthLSSTransform().to(device).eval()

		batch_size = 2
		num_cams = 6
		num_points = 128

		img = torch.randn(
			batch_size,
			num_cams,
			model.in_channels,
			model.feature_size[0],
			model.feature_size[1],
			device=device,
		)

		points = []
		for _ in range(batch_size):
			xyz = torch.rand(num_points, 3, device=device)
			xyz[:, 0] = xyz[:, 0] * 20.0 - 10.0
			xyz[:, 1] = xyz[:, 1] * 10.0 - 5.0
			xyz[:, 2] = xyz[:, 2] * 20.0 + 5.0
			feat = torch.rand(num_points, 1, device=device)
			points.append(torch.cat([xyz, feat], dim=1))

		radar = [p.clone() for p in points]

		eye4 = torch.eye(4, device=device)
		sensor2ego = repeat(eye4, 'i j -> b n i j', b=batch_size, n=num_cams)
		lidar2ego = repeat(eye4, 'i j -> b i j', b=batch_size)
		lidar2camera = repeat(eye4, 'i j -> b n i j', b=batch_size, n=num_cams)
		lidar2image = repeat(eye4, 'i j -> b n i j', b=batch_size, n=num_cams)
		cam_intrinsic = repeat(eye4, 'i j -> b n i j', b=batch_size, n=num_cams)
		cam_intrinsic[..., 0, 0] = 200.0
		cam_intrinsic[..., 1, 1] = 200.0
		cam_intrinsic[..., 0, 2] = model.image_size[1] / 2.0
		cam_intrinsic[..., 1, 2] = model.image_size[0] / 2.0
		camera2lidar = repeat(eye4, 'i j -> b n i j', b=batch_size, n=num_cams)
		img_aug_matrix = repeat(eye4, 'i j -> b n i j', b=batch_size, n=num_cams)
		lidar_aug_matrix = repeat(eye4, 'i j -> b i j', b=batch_size)
		metas = [{} for _ in range(batch_size)]

		with torch.no_grad():
			output = model(
				img,
				points,
				radar,
				sensor2ego,
				lidar2ego,
				lidar2camera,
				lidar2image,
				cam_intrinsic,
				camera2lidar,
				img_aug_matrix,
				lidar_aug_matrix,
				metas,
			)

		print(output.shape)
