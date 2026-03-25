from __future__ import annotations

from typing import Any

import numpy as np
import torch


def to_label_tensor(labels: Any) -> torch.Tensor:
	if torch.is_tensor(labels):
		return labels.long()
	if isinstance(labels, np.ndarray):
		return torch.from_numpy(labels).long()
	return torch.as_tensor(labels, dtype=torch.long)


def extract_meta(batch: dict[str, Any], index: int) -> dict[str, Any]:
	meta: dict[str, Any] = {}
	for key in (
		'token',
		'sample_idx',
		'box_type_3d',
		'box_mode_3d',
		'image_paths',
		'lidar_path',
		'timestamp',
	):
		if key in batch and isinstance(batch[key], list) and len(batch[key]) > index:
			meta[key] = batch[key][index]
	return meta


def prepare_bevfusion_batch(batch: dict[str, Any]) -> dict[str, Any]:
	device = batch['img'].device if torch.is_tensor(batch.get('img')) else None

	points = [point.tensor if hasattr(point, 'tensor') else point for point in batch['points']]
	if device is not None:
		points = [point.to(device) if torch.is_tensor(point) else point for point in points]

	model_inputs: dict[str, Any] = {
		'img': batch['img'],
		'points': points,
		'camera2ego': batch['camera2ego'],
		'lidar2ego': batch['lidar2ego'],
		'lidar2camera': batch['lidar2camera'],
		'lidar2image': batch['lidar2image'],
		'camera_intrinsics': batch['camera_intrinsics'],
		'camera2lidar': batch['camera2lidar'],
		'img_aug_matrix': batch['img_aug_matrix'],
		'lidar_aug_matrix': batch['lidar_aug_matrix'],
		'metas': [extract_meta(batch, index) for index in range(len(points))],
	}

	if batch.get('radar') is not None:
		radar = [radar.tensor if hasattr(radar, 'tensor') else radar for radar in batch['radar']]
		if device is not None:
			radar = [point.to(device) if torch.is_tensor(point) else point for point in radar]
		model_inputs['radar'] = radar

	if batch.get('depths') is not None:
		model_inputs['depths'] = batch['depths']

	if batch.get('gt_masks_bev') is not None:
		model_inputs['gt_masks_bev'] = batch['gt_masks_bev']

	if batch.get('gt_bboxes_3d') is not None:
		model_inputs['gt_bboxes_3d'] = batch['gt_bboxes_3d']

	if batch.get('gt_labels_3d') is not None:
		labels = [to_label_tensor(label) for label in batch['gt_labels_3d']]
		if device is not None:
			labels = [label.to(device) for label in labels]
		model_inputs['gt_labels_3d'] = labels

	return model_inputs
