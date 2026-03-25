from typing import Any, Dict

import numpy as np
import torch
import torchvision
from PIL import Image

from core.bbox import CameraInstance3DBoxes, DepthInstance3DBoxes, LiDARInstance3DBoxes


class GTDepth:
	def __init__(self, keyframe_only=False):
		self.keyframe_only = keyframe_only

	def __call__(self, data):
		if 'points' not in data or 'img' not in data:
			return data

		img = data['img']
		points = data['points']
		lidar_aug_matrix = torch.as_tensor(np.asarray(data['lidar_aug_matrix']))
		lidar2image = torch.as_tensor(np.asarray(data['lidar2image']))
		img_aug_matrix = torch.as_tensor(np.asarray(data['img_aug_matrix']))

		if self.keyframe_only and points.tensor.shape[1] > 4:
			points_tensor = points.tensor[points.tensor[:, 4] == 0]
		else:
			points_tensor = points.tensor

		depth = torch.zeros(len(img), img[0].shape[-2], img[0].shape[-1], dtype=torch.float32)
		cur_coords = points_tensor[:, :3]

		cur_coords = cur_coords - lidar_aug_matrix[:3, 3]
		cur_coords = torch.inverse(lidar_aug_matrix[:3, :3]).matmul(cur_coords.transpose(1, 0))
		cur_coords = lidar2image[:, :3, :3].matmul(cur_coords)
		cur_coords += lidar2image[:, :3, 3].reshape(-1, 3, 1)

		dist = cur_coords[:, 2, :]
		cur_coords[:, 2, :] = torch.clamp(cur_coords[:, 2, :], 1e-5, 1e5)
		cur_coords[:, :2, :] /= cur_coords[:, 2:3, :]

		cur_coords = img_aug_matrix[:, :3, :3].matmul(cur_coords)
		cur_coords += img_aug_matrix[:, :3, 3].reshape(-1, 3, 1)
		cur_coords = cur_coords[:, :2, :].transpose(1, 2)
		cur_coords = cur_coords[..., [1, 0]]

		on_img = (
			(cur_coords[..., 0] < img[0].shape[-2])
			& (cur_coords[..., 0] >= 0)
			& (cur_coords[..., 1] < img[0].shape[-1])
			& (cur_coords[..., 1] >= 0)
		)
		for cam_idx in range(on_img.shape[0]):
			masked_coords = cur_coords[cam_idx, on_img[cam_idx]].long()
			masked_dist = dist[cam_idx, on_img[cam_idx]]
			depth[cam_idx, masked_coords[:, 0], masked_coords[:, 1]] = masked_dist

		data['depths'] = depth
		return data


class ImageAug3D:
	def __init__(self, final_dim, resize_lim, bot_pct_lim, rot_lim, rand_flip, is_train):
		self.final_dim = final_dim
		self.resize_lim = resize_lim
		self.bot_pct_lim = bot_pct_lim
		self.rand_flip = rand_flip
		self.rot_lim = rot_lim
		self.is_train = is_train

	def sample_augmentation(self, results):
		W, H = results['ori_shape']
		fH, fW = self.final_dim
		if self.is_train:
			resize = np.random.uniform(*self.resize_lim)
			resize_dims = (int(W * resize), int(H * resize))
			newW, newH = resize_dims
			crop_h = int((1 - np.random.uniform(*self.bot_pct_lim)) * newH) - fH
			crop_w = int(np.random.uniform(0, max(0, newW - fW)))
			crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
			flip = bool(self.rand_flip and np.random.choice([0, 1]))
			rotate = np.random.uniform(*self.rot_lim)
		else:
			resize = float(np.mean(self.resize_lim))
			resize_dims = (int(W * resize), int(H * resize))
			newW, newH = resize_dims
			crop_h = int((1 - np.mean(self.bot_pct_lim)) * newH) - fH
			crop_w = int(max(0, newW - fW) / 2)
			crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
			flip = False
			rotate = 0.0
		return resize, resize_dims, crop, flip, rotate

	def img_transform(self, img, rotation, translation, resize, resize_dims, crop, flip, rotate):
		img = img.resize(resize_dims)
		img = img.crop(crop)
		if flip:
			img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
		img = img.rotate(rotate)

		rotation = rotation * resize
		translation = translation - torch.tensor(crop[:2], dtype=torch.float32)
		if flip:
			A = torch.tensor([[-1, 0], [0, 1]], dtype=torch.float32)
			b = torch.tensor([crop[2] - crop[0], 0], dtype=torch.float32)
			rotation = A.matmul(rotation)
			translation = A.matmul(translation) + b
		theta = rotate / 180.0 * np.pi
		A = torch.tensor(
			[[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]],
			dtype=torch.float32,
		)
		b = torch.tensor([crop[2] - crop[0], crop[3] - crop[1]], dtype=torch.float32) / 2
		b = A.matmul(-b) + b
		rotation = A.matmul(rotation)
		translation = A.matmul(translation) + b
		return img, rotation, translation

	def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
		new_imgs = []
		transforms = []
		for img in data['img']:
			resize, resize_dims, crop, flip, rotate = self.sample_augmentation(data)
			post_rot = torch.eye(2, dtype=torch.float32)
			post_tran = torch.zeros(2, dtype=torch.float32)
			new_img, rotation, translation = self.img_transform(
				img,
				post_rot,
				post_tran,
				resize=resize,
				resize_dims=resize_dims,
				crop=crop,
				flip=flip,
				rotate=rotate,
			)
			transform = torch.eye(4, dtype=torch.float32)
			transform[:2, :2] = rotation
			transform[:2, 3] = translation
			new_imgs.append(new_img)
			transforms.append(transform.numpy())
		data['img'] = new_imgs
		data['img_aug_matrix'] = transforms
		return data


class GlobalRotScaleTrans:
	def __init__(self, resize_lim, rot_lim, trans_lim, is_train):
		self.resize_lim = resize_lim
		self.rot_lim = rot_lim
		self.trans_lim = trans_lim
		self.is_train = is_train

	def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
		transform = np.eye(4, dtype=np.float32)
		if self.is_train:
			scale = np.random.uniform(*self.resize_lim)
			theta = np.random.uniform(*self.rot_lim)
			translation = np.array(
				[np.random.normal(0, self.trans_lim) for _ in range(3)], dtype=np.float32
			)
			rotation = np.eye(3, dtype=np.float32)

			if 'points' in data:
				data['points'].rotate(-theta)
				data['points'].translate(translation)
				data['points'].scale(scale)
			if 'radar' in data:
				data['radar'].rotate(-theta)
				data['radar'].translate(translation)
				data['radar'].scale(scale)
			if 'gt_bboxes_3d' in data:
				gt_boxes = data['gt_bboxes_3d']
				rotation = rotation @ gt_boxes.rotate(theta).numpy()
				gt_boxes.translate(translation)
				gt_boxes.scale(scale)
				data['gt_bboxes_3d'] = gt_boxes

			transform[:3, :3] = rotation.T * scale
			transform[:3, 3] = translation * scale

		data['lidar_aug_matrix'] = transform
		return data


class GridMask:
	def __init__(
		self,
		use_h,
		use_w,
		max_epoch,
		rotate=1,
		offset=False,
		ratio=0.5,
		mode=0,
		prob=1.0,
		fixed_prob=False,
	):
		self.use_h = use_h
		self.use_w = use_w
		self.rotate = rotate
		self.offset = offset
		self.ratio = ratio
		self.mode = mode
		self.st_prob = prob
		self.prob = prob
		self.epoch = None
		self.max_epoch = max_epoch
		self.fixed_prob = fixed_prob

	def set_epoch(self, epoch):
		self.epoch = epoch
		if not self.fixed_prob and self.max_epoch > 0:
			self.prob = self.st_prob * self.epoch / self.max_epoch

	def __call__(self, results):
		if np.random.rand() > self.prob:
			return results
		imgs = results['img']
		if not imgs or not torch.is_tensor(imgs[0]):
			return results
		h, w = imgs[0].shape[-2:]
		d = np.random.randint(2, min(h, w))
		l = (
			np.random.randint(1, d)
			if self.ratio == 1
			else min(max(int(d * self.ratio + 0.5), 1), d - 1)
		)
		hh = int(1.5 * h)
		ww = int(1.5 * w)
		mask = np.ones((hh, ww), np.float32)
		st_h = np.random.randint(d)
		st_w = np.random.randint(d)
		if self.use_h:
			for i in range(hh // d):
				s = d * i + st_h
				t = min(s + l, hh)
				mask[s:t, :] *= 0
		if self.use_w:
			for i in range(ww // d):
				s = d * i + st_w
				t = min(s + l, ww)
				mask[:, s:t] *= 0
		r = np.random.randint(max(self.rotate, 1))
		mask = Image.fromarray(np.uint8(mask))
		mask = np.asarray(mask.rotate(r))
		mask = mask[(hh - h) // 2 : (hh - h) // 2 + h, (ww - w) // 2 : (ww - w) // 2 + w].astype(
			np.float32
		)
		mask = torch.from_numpy(mask).to(imgs[0].device).unsqueeze(0)
		if self.mode == 1:
			mask = 1 - mask
		if self.offset:
			offset = (
				torch.from_numpy(2 * (np.random.rand(h, w) - 0.5))
				.float()
				.to(imgs[0].device)
				.unsqueeze(0)
			)
			offset = (1 - mask) * offset
			imgs = [x * mask + offset for x in imgs]
		else:
			imgs = [x * mask for x in imgs]
		results['img'] = imgs
		return results


class RandomFlip3D:
	def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
		flip_horizontal = np.random.choice([0, 1])
		flip_vertical = np.random.choice([0, 1])
		rotation = np.eye(3, dtype=np.float32)

		if 'lidar_aug_matrix' not in data:
			data['lidar_aug_matrix'] = np.eye(4, dtype=np.float32)

		if flip_horizontal:
			rotation = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=np.float32) @ rotation
			if 'points' in data:
				data['points'].flip('horizontal')
			if 'radar' in data:
				data['radar'].flip('horizontal')
			if 'gt_bboxes_3d' in data:
				data['gt_bboxes_3d'].flip('horizontal')
			if 'gt_masks_bev' in data:
				data['gt_masks_bev'] = data['gt_masks_bev'][:, :, ::-1].copy()

		if flip_vertical:
			rotation = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32) @ rotation
			if 'points' in data:
				data['points'].flip('vertical')
			if 'radar' in data:
				data['radar'].flip('vertical')
			if 'gt_bboxes_3d' in data:
				data['gt_bboxes_3d'].flip('vertical')
			if 'gt_masks_bev' in data:
				data['gt_masks_bev'] = data['gt_masks_bev'][:, ::-1, :].copy()

		data['lidar_aug_matrix'][:3, :] = rotation @ data['lidar_aug_matrix'][:3, :]
		return data


class PointShuffle:
	def __call__(self, data):
		if 'points' in data:
			data['points'].shuffle()
		return data


class ObjectRangeFilter:
	def __init__(self, point_cloud_range):
		self.pcd_range = np.array(point_cloud_range, dtype=np.float32)

	def __call__(self, data):
		if 'gt_bboxes_3d' not in data or 'gt_labels_3d' not in data:
			return data
		if isinstance(data['gt_bboxes_3d'], (LiDARInstance3DBoxes, DepthInstance3DBoxes)):
			bev_range = self.pcd_range[[0, 1, 3, 4]]
		elif isinstance(data['gt_bboxes_3d'], CameraInstance3DBoxes):
			bev_range = self.pcd_range[[0, 2, 3, 5]]
		else:
			return data

		mask = data['gt_bboxes_3d'].in_range_bev(bev_range)
		data['gt_bboxes_3d'] = data['gt_bboxes_3d'][mask]
		data['gt_labels_3d'] = data['gt_labels_3d'][mask.cpu().numpy().astype(np.bool_)]
		data['gt_bboxes_3d'].limit_yaw(offset=0.5, period=2 * np.pi)
		return data


class PointsRangeFilter:
	def __init__(self, point_cloud_range):
		self.pcd_range = np.array(point_cloud_range, dtype=np.float32)

	def __call__(self, data):
		if 'points' in data:
			mask = data['points'].in_range_3d(self.pcd_range)
			data['points'] = data['points'][mask]
		if 'radar' in data and hasattr(data['radar'], 'in_range_bev'):
			mask = data['radar'].in_range_bev([-55.0, -55.0, 55.0, 55.0])
			data['radar'] = data['radar'][mask]
		return data


class ObjectNameFilter:
	def __init__(self, classes):
		self.classes = classes
		self.labels = list(range(len(classes)))

	def __call__(self, data):
		if 'gt_labels_3d' not in data or 'gt_bboxes_3d' not in data:
			return data
		mask = np.array([label in self.labels for label in data['gt_labels_3d']], dtype=np.bool_)
		data['gt_bboxes_3d'] = data['gt_bboxes_3d'][mask]
		data['gt_labels_3d'] = data['gt_labels_3d'][mask]
		return data


class ImageNormalize:
	def __init__(self, mean, std):
		self.mean = mean
		self.std = std
		self.compose = torchvision.transforms.Compose(
			[
				torchvision.transforms.ToTensor(),
				torchvision.transforms.Normalize(mean=mean, std=std),
			]
		)

	def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
		data['img'] = [self.compose(img) for img in data['img']]
		data['img_norm_cfg'] = dict(mean=self.mean, std=self.std)
		return data
