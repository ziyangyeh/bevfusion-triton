from typing import Any

import numpy as np
import torch
from PIL import Image

from core.bbox import BaseInstance3DBoxes
from core.points import BasePoints


def _to_tensor(value):
	if torch.is_tensor(value):
		return value
	if isinstance(value, np.ndarray):
		return torch.from_numpy(value)
	if isinstance(value, (int, float, bool, np.number)):
		return torch.tensor(value)
	return value


def _format_image(img: Any):
	if torch.is_tensor(img):
		return img
	if isinstance(img, Image.Image):
		img = np.asarray(img)
	if isinstance(img, np.ndarray):
		if img.ndim == 3:
			img = torch.from_numpy(img.transpose(2, 0, 1))
		else:
			img = torch.from_numpy(img)
	return img


class DefaultFormatBundle3D:
	def __init__(self, classes, with_gt=True, with_label=True) -> None:
		self.class_names = list(classes)
		self.with_gt = with_gt
		self.with_label = with_label

	def __call__(self, results):
		if 'img' in results:
			results['img'] = [_format_image(img) for img in results['img']]

		if self.with_gt:
			if 'gt_bboxes_3d_mask' in results:
				mask = results['gt_bboxes_3d_mask']
				results['gt_bboxes_3d'] = results['gt_bboxes_3d'][mask]
				if 'gt_names_3d' in results:
					results['gt_names_3d'] = results['gt_names_3d'][mask]
			if 'gt_bboxes_mask' in results:
				mask = results['gt_bboxes_mask']
				if 'gt_bboxes' in results:
					results['gt_bboxes'] = results['gt_bboxes'][mask]
				if 'gt_names' in results:
					results['gt_names'] = results['gt_names'][mask]
			if self.with_label:
				if 'gt_names' in results and 'gt_labels' not in results:
					if len(results['gt_names']) == 0:
						results['gt_labels'] = np.array([], dtype=np.int64)
					elif isinstance(results['gt_names'][0], list):
						results['gt_labels'] = [
							np.array(
								[self.class_names.index(name) for name in names], dtype=np.int64
							)
							for names in results['gt_names']
						]
					else:
						results['gt_labels'] = np.array(
							[self.class_names.index(name) for name in results['gt_names']],
							dtype=np.int64,
						)
				if 'gt_names_3d' in results and 'gt_labels_3d' not in results:
					results['gt_labels_3d'] = np.array(
						[self.class_names.index(name) for name in results['gt_names_3d']],
						dtype=np.int64,
					)

		for key in [
			'proposals',
			'gt_bboxes',
			'gt_bboxes_ignore',
			'gt_labels',
			'gt_labels_3d',
			'attr_labels',
			'centers2d',
			'depths',
			'gt_masks_bev',
		]:
			if key not in results:
				continue
			if isinstance(results[key], list):
				results[key] = [_to_tensor(item) for item in results[key]]
			elif not isinstance(results[key], (BaseInstance3DBoxes, BasePoints)):
				results[key] = _to_tensor(results[key])
		return results


class Collect3D:
	def __init__(
		self,
		keys,
		meta_keys=(
			'camera_intrinsics',
			'camera2ego',
			'img_aug_matrix',
			'lidar_aug_matrix',
		),
		meta_lis_keys=(
			'filename',
			'timestamp',
			'ori_shape',
			'img_shape',
			'lidar2image',
			'depth2img',
			'cam2img',
			'pad_shape',
			'scale_factor',
			'flip',
			'pcd_horizontal_flip',
			'pcd_vertical_flip',
			'box_mode_3d',
			'box_type_3d',
			'img_norm_cfg',
			'pcd_trans',
			'token',
			'pcd_scale_factor',
			'pcd_rotation',
			'lidar_path',
			'transformation_3d_flow',
		),
	) -> None:
		self.keys = keys
		self.meta_keys = meta_keys
		self.meta_lis_keys = meta_lis_keys

	def __call__(self, results):
		data = {}
		for key in self.keys:
			if key in results:
				data[key] = results[key]

		for key in self.meta_keys:
			if key in results:
				data[key] = results[key]

		metas = {}
		for key in self.meta_lis_keys:
			if key in results:
				metas[key] = results[key]
		data['metas'] = metas
		return data
