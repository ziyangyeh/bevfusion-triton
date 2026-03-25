from typing import Tuple

import torch
from einops import rearrange, repeat
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from models.helper import force_fp32
from ops.bev_pool import bev_pool

__all__ = ['BaseTransform', 'BaseDepthTransform']


def boolmask2idx(mask):
	# A utility function, workaround for ONNX not supporting 'nonzero'
	return torch.nonzero(mask).squeeze(1).tolist()


def gen_dx_bx(xbound, ybound, zbound):
	dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
	bx = torch.Tensor([row[0] + row[2] / 2.0 for row in [xbound, ybound, zbound]])
	nx = torch.LongTensor([(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]])
	return dx, bx, nx


class BaseTransform(nn.Module):
	def __init__(
		self,
		in_channels: int,
		out_channels: int,
		image_size: Tuple[int, int],
		feature_size: Tuple[int, int],
		xbound: Tuple[float, float, float],
		ybound: Tuple[float, float, float],
		zbound: Tuple[float, float, float],
		dbound: Tuple[float, float, float],
		use_points='lidar',
		depth_input='scalar',
		height_expand=True,
		add_depth_features=True,
	) -> None:
		super().__init__()
		self.in_channels = in_channels
		self.image_size = image_size
		self.feature_size = feature_size
		self.xbound = xbound
		self.ybound = ybound
		self.zbound = zbound
		self.dbound = dbound
		self.use_points = use_points
		assert use_points in ['radar', 'lidar']
		self.depth_input = depth_input
		assert depth_input in ['scalar', 'one-hot']
		self.height_expand = height_expand
		self.add_depth_features = add_depth_features

		dx, bx, nx = gen_dx_bx(self.xbound, self.ybound, self.zbound)
		self.dx = nn.Parameter(dx, requires_grad=False)
		self.bx = nn.Parameter(bx, requires_grad=False)
		self.nx = nn.Parameter(nx, requires_grad=False)

		self.C = out_channels
		self.frustum = self.create_frustum()
		self.D = self.frustum.shape[0]
		self.fp16_enabled = False

	@force_fp32()
	def create_frustum(self):
		iH, iW = self.image_size
		fH, fW = self.feature_size

		depth_bins = torch.arange(*self.dbound, dtype=torch.float)
		D = depth_bins.shape[0]
		ds = repeat(depth_bins, 'd -> d h w', h=fH, w=fW)

		xs = repeat(torch.linspace(0, iW - 1, fW, dtype=torch.float), 'w -> d h w', d=D, h=fH)
		ys = repeat(torch.linspace(0, iH - 1, fH, dtype=torch.float), 'h -> d h w', d=D, w=fW)

		frustum = torch.stack((xs, ys, ds), -1)
		return nn.Parameter(frustum, requires_grad=False)

	@force_fp32()
	def get_geometry(
		self,
		camera2lidar_rots,
		camera2lidar_trans,
		intrins,
		post_rots,
		post_trans,
		**kwargs,
	):
		B, N, _ = camera2lidar_trans.shape

		# undo post-transformation
		# B x N x D x H x W x 3
		points = self.frustum - rearrange(post_trans, 'b n c -> b n 1 1 1 c')
		points = (
			post_rots.transpose(-1, -2).reshape(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))
		)
		# cam_to_lidar
		points = torch.cat(
			(
				points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
				points[:, :, :, :, :, 2:3],
			),
			5,
		)
		combine = torch.linalg.solve(
			intrins.transpose(-1, -2),
			camera2lidar_rots.transpose(-1, -2),
		).transpose(-1, -2)
		points = combine.reshape(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
		points += rearrange(camera2lidar_trans, 'b n c -> b n 1 1 1 c')

		if 'extra_rots' in kwargs:
			extra_rots = kwargs['extra_rots']
			points = (
				repeat(extra_rots, 'b i j -> b n 1 1 1 i j', n=N)
				.matmul(points.unsqueeze(-1))
				.squeeze(-1)
			)
		if 'extra_trans' in kwargs:
			extra_trans = kwargs['extra_trans']
			points += repeat(extra_trans, 'b c -> b n 1 1 1 c', n=N)

		return points

	def get_cam_feats(self, x):
		raise NotImplementedError

	@force_fp32()
	def bev_pool(self, geom_feats, x):
		B, N, D, H, W, C = x.shape

		# flatten x
		x = rearrange(x, 'b n d h w c -> (b n d h w) c')

		# flatten indices
		geom_feats = ((geom_feats - (self.bx - self.dx / 2.0)) / self.dx).long()
		geom_feats = rearrange(geom_feats, 'b n d h w c -> (b n d h w) c')
		batch_ix = repeat(
			torch.arange(B, device=x.device, dtype=torch.long),
			'b -> (b n) 1',
			n=N * D * H * W,
		)
		geom_feats = torch.cat((geom_feats, batch_ix), 1)

		# filter out points that are outside box
		kept = (
			(geom_feats[:, 0] >= 0)
			& (geom_feats[:, 0] < self.nx[0])
			& (geom_feats[:, 1] >= 0)
			& (geom_feats[:, 1] < self.nx[1])
			& (geom_feats[:, 2] >= 0)
			& (geom_feats[:, 2] < self.nx[2])
		)
		x = x[kept]
		geom_feats = geom_feats[kept]

		x = bev_pool(x, geom_feats, B, self.nx[2], self.nx[0], self.nx[1])

		# collapse Z
		final = torch.cat(x.unbind(dim=2), 1)

		return final

	@force_fp32()
	def forward(
		self,
		img,
		points,
		radar,
		camera2ego,
		lidar2ego,
		lidar2camera,
		lidar2image,
		camera_intrinsics,
		camera2lidar,
		img_aug_matrix,
		lidar_aug_matrix,
		**kwargs,
	):
		rots = camera2ego[..., :3, :3]
		trans = camera2ego[..., :3, 3]
		intrins = camera_intrinsics[..., :3, :3]
		post_rots = img_aug_matrix[..., :3, :3]
		post_trans = img_aug_matrix[..., :3, 3]
		lidar2ego_rots = lidar2ego[..., :3, :3]
		lidar2ego_trans = lidar2ego[..., :3, 3]
		camera2lidar_rots = camera2lidar[..., :3, :3]
		camera2lidar_trans = camera2lidar[..., :3, 3]

		extra_rots = lidar_aug_matrix[..., :3, :3]
		extra_trans = lidar_aug_matrix[..., :3, 3]

		geom = self.get_geometry(
			camera2lidar_rots,
			camera2lidar_trans,
			intrins,
			post_rots,
			post_trans,
			extra_rots=extra_rots,
			extra_trans=extra_trans,
		)
		mats_dict = {
			'intrin_mats': camera_intrinsics,
			'ida_mats': img_aug_matrix,
			'bda_mat': lidar_aug_matrix,
			'sensor2ego_mats': camera2ego,
		}
		x = self.get_cam_feats(img, mats_dict)

		use_depth = False
		if type(x) == tuple:
			x, depth = x
			use_depth = True

		x = self.bev_pool(geom, x)

		if use_depth:
			return x, depth
		else:
			return x


class BaseDepthTransform(BaseTransform):
	@force_fp32()
	def forward(
		self,
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
		**kwargs,
	):
		rots = sensor2ego[..., :3, :3]
		trans = sensor2ego[..., :3, 3]
		intrins = cam_intrinsic[..., :3, :3]
		post_rots = img_aug_matrix[..., :3, :3]
		post_trans = img_aug_matrix[..., :3, 3]
		lidar2ego_rots = lidar2ego[..., :3, :3]
		lidar2ego_trans = lidar2ego[..., :3, 3]
		camera2lidar_rots = camera2lidar[..., :3, :3]
		camera2lidar_trans = camera2lidar[..., :3, 3]

		if self.use_points == 'radar':
			points = radar

		if self.height_expand:
			height_steps = torch.arange(0.25, 2.25, 0.25, device=points[0].device)
			num_heights = height_steps.numel()
			expanded_points = []
			for point_set in points:
				expanded = repeat(point_set, 'p c -> (p h) c', h=num_heights).clone()
				expanded[:, 2] = repeat(height_steps, 'h -> (p h)', p=point_set.shape[0])
				expanded_points.append(expanded)
			points = expanded_points

		batch_size = len(points)
		depth_in_channels = 1 if self.depth_input == 'scalar' else self.D
		if self.add_depth_features:
			depth_in_channels += points[0].shape[1]

		depth = torch.zeros(
			batch_size, img.shape[1], depth_in_channels, *self.image_size, device=points[0].device
		)
		point_lengths = torch.tensor(
			[point_set.shape[0] for point_set in points], device=points[0].device, dtype=torch.long
		)
		padded_points = pad_sequence(points, batch_first=True)
		max_points = padded_points.shape[1]
		valid_points = torch.arange(max_points, device=padded_points.device).unsqueeze(
			0
		) < point_lengths.unsqueeze(1)

		point_xyz = padded_points[..., :3]

		# inverse lidar augmentation for row-vector points
		point_xyz = point_xyz - lidar_aug_matrix[:, None, :3, 3]
		point_xyz = point_xyz.matmul(lidar_aug_matrix[:, :3, :3])

		# project lidar points to each camera
		point_xyz = repeat(point_xyz, 'b p c -> b n p c', n=img.shape[1])
		lidar2image_rot = lidar2image[:, :, :3, :3]
		lidar2image_trans = lidar2image[:, :, :3, 3]
		point_xyz = point_xyz.matmul(lidar2image_rot.transpose(-1, -2))
		point_xyz = point_xyz + lidar2image_trans.unsqueeze(2)

		dist = point_xyz[..., 2]
		point_xyz[..., 2] = torch.clamp(point_xyz[..., 2], 1e-5, 1e5)
		point_xyz[..., :2] = point_xyz[..., :2] / point_xyz[..., 2:3]

		img_aug_rot = img_aug_matrix[:, :, :3, :3]
		img_aug_trans = img_aug_matrix[:, :, :3, 3]
		point_xyz = point_xyz.matmul(img_aug_rot.transpose(-1, -2))
		point_xyz = point_xyz + img_aug_trans.unsqueeze(2)

		cur_coords = point_xyz[..., :2][..., [1, 0]]
		on_img = (
			valid_points.unsqueeze(1)
			& (cur_coords[..., 0] < self.image_size[0])
			& (cur_coords[..., 0] >= 0)
			& (cur_coords[..., 1] < self.image_size[1])
			& (cur_coords[..., 1] >= 0)
		)

		batch_idx, cam_idx, point_idx = torch.nonzero(on_img, as_tuple=True)
		if batch_idx.numel() > 0:
			masked_coords = cur_coords[batch_idx, cam_idx, point_idx].long()
			masked_dist = dist[batch_idx, cam_idx, point_idx]
			y_idx = masked_coords[:, 0]
			x_idx = masked_coords[:, 1]

			if self.depth_input == 'scalar':
				depth[batch_idx, cam_idx, 0, y_idx, x_idx] = masked_dist
			elif self.depth_input == 'one-hot':
				# Clamp depths that are too big to D. These can arise when
				# the point range filter is different from the dbound.
				masked_dist = torch.clamp(masked_dist, max=self.D - 1)
				depth[batch_idx, cam_idx, masked_dist.long(), y_idx, x_idx] = 1.0

			if self.add_depth_features:
				point_feats = padded_points[batch_idx, point_idx]
				feature_start = depth_in_channels - padded_points.shape[-1]
				feature_idx = (
					torch.arange(padded_points.shape[-1], device=padded_points.device)
					+ feature_start
				)
				depth[
					batch_idx[:, None],
					cam_idx[:, None],
					feature_idx[None, :],
					y_idx[:, None],
					x_idx[:, None],
				] = point_feats

		extra_rots = lidar_aug_matrix[..., :3, :3]
		extra_trans = lidar_aug_matrix[..., :3, 3]
		geom = self.get_geometry(
			camera2lidar_rots,
			camera2lidar_trans,
			intrins,
			post_rots,
			post_trans,
			extra_rots=extra_rots,
			extra_trans=extra_trans,
		)

		mats_dict = {
			'intrin_mats': intrins,
			'ida_mats': img_aug_matrix,
			'bda_mat': lidar_aug_matrix,
			'sensor2ego_mats': sensor2ego,
		}
		x = self.get_cam_feats(img, depth, mats_dict)

		use_depth = False
		if type(x) == tuple:
			x, depth = x
			use_depth = True

		x = self.bev_pool(geom, x)

		if use_depth:
			return x, depth
		else:
			return x
