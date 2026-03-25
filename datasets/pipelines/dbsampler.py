import copy
import os
import pickle
from typing import Any

import numpy as np

from core.bbox import box_np_ops

from .loading import LoadPointsFromFile
from .utils import box_collision_test


class BatchSampler:
	def __init__(self, sampled_list, name=None, shuffle=True) -> None:
		self._sampled_list = sampled_list
		self._indices = np.arange(len(sampled_list))
		if shuffle:
			np.random.shuffle(self._indices)
		self._idx = 0
		self._example_num = len(sampled_list)
		self._name = name
		self._shuffle = shuffle

	def _sample(self, num: int) -> np.ndarray:
		if self._idx + num >= self._example_num:
			ret = self._indices[self._idx :].copy()
			self._reset()
		else:
			ret = self._indices[self._idx : self._idx + num]
			self._idx += num
		return ret

	def _reset(self) -> None:
		if self._shuffle:
			np.random.shuffle(self._indices)
		self._idx = 0

	def sample(self, num: int):
		indices = self._sample(num)
		return [self._sampled_list[i] for i in indices]


class DataBaseSampler:
	def __init__(
		self,
		info_path,
		dataset_root,
		rate,
		prepare,
		sample_groups,
		classes=None,
		points_loader=None,
	) -> None:
		self.dataset_root = dataset_root
		self.info_path = info_path
		self.rate = rate
		self.prepare = prepare or {}
		self.classes = classes or []
		self.cat2label = {name: i for i, name in enumerate(self.classes)}
		self.label2cat = {i: name for i, name in enumerate(self.classes)}

		points_loader = (
			dict(
				coord_type='LIDAR',
				load_dim=4,
				use_dim=[0, 1, 2, 3],
			)
			if points_loader is None
			else dict(points_loader)
		)
		points_loader.pop('type', None)
		self.points_loader = LoadPointsFromFile(**points_loader)

		with open(info_path, 'rb') as f:
			db_infos = pickle.load(f)

		for prep_func, val in self.prepare.items():
			db_infos = getattr(self, prep_func)(db_infos, val)
		self.db_infos = db_infos

		self.sample_groups = [{name: int(num)} for name, num in sample_groups.items()]
		self.group_db_infos = self.db_infos
		self.sample_classes = []
		self.sample_max_nums = []
		for group_info in self.sample_groups:
			self.sample_classes += list(group_info.keys())
			self.sample_max_nums += list(group_info.values())

		self.sampler_dict = {
			key: BatchSampler(value, key, shuffle=True)
			for key, value in self.group_db_infos.items()
		}

	@staticmethod
	def filter_by_difficulty(db_infos, removed_difficulty):
		new_db_infos = {}
		for key, infos in db_infos.items():
			new_db_infos[key] = [
				info for info in infos if info['difficulty'] not in removed_difficulty
			]
		return new_db_infos

	@staticmethod
	def filter_by_min_points(db_infos, min_gt_points_dict):
		for name, min_num in min_gt_points_dict.items():
			min_num = int(min_num)
			if min_num <= 0 or name not in db_infos:
				continue
			db_infos[name] = [
				info for info in db_infos[name] if info['num_points_in_gt'] >= min_num
			]
		return db_infos

	def sample_all(self, gt_bboxes, gt_labels, img=None):
		del img
		sample_num_per_class = []
		for class_name, max_sample_num in zip(self.sample_classes, self.sample_max_nums):
			class_label = self.cat2label[class_name]
			sampled_num = int(max_sample_num - np.sum(gt_labels == class_label))
			sampled_num = int(np.round(self.rate * sampled_num))
			sample_num_per_class.append(sampled_num)

		sampled = []
		sampled_gt_bboxes = []
		avoid_coll_boxes = gt_bboxes

		for class_name, sampled_num in zip(self.sample_classes, sample_num_per_class):
			if sampled_num <= 0:
				continue
			sampled_cls = self.sample_class_v2(class_name, sampled_num, avoid_coll_boxes)
			sampled += sampled_cls
			if not sampled_cls:
				continue
			if len(sampled_cls) == 1:
				sampled_gt_box = sampled_cls[0]['box3d_lidar'][np.newaxis, ...]
			else:
				sampled_gt_box = np.stack([s['box3d_lidar'] for s in sampled_cls], axis=0)
			sampled_gt_bboxes.append(sampled_gt_box)
			avoid_coll_boxes = np.concatenate([avoid_coll_boxes, sampled_gt_box], axis=0)

		if not sampled:
			return None

		sampled_gt_bboxes = np.concatenate(sampled_gt_bboxes, axis=0)
		sampled_points_list = []
		for info in sampled:
			file_path = (
				os.path.join(self.dataset_root, info['path'])
				if self.dataset_root and not os.path.isabs(info['path'])
				else info['path']
			)
			sampled_points = self.points_loader(dict(lidar_path=file_path))['points']
			sampled_points.translate(info['box3d_lidar'][:3])
			sampled_points_list.append(sampled_points)

		gt_labels_out = np.array(
			[self.cat2label[s['name']] for s in sampled],
			dtype=np.int64,
		)
		return {
			'gt_labels_3d': gt_labels_out,
			'gt_bboxes_3d': sampled_gt_bboxes,
			'points': sampled_points_list[0].cat(sampled_points_list),
			'group_ids': np.arange(gt_bboxes.shape[0], gt_bboxes.shape[0] + len(sampled)),
		}

	def sample_class_v2(self, name, num, gt_bboxes):
		sampled = copy.deepcopy(self.sampler_dict[name].sample(num))
		num_gt = gt_bboxes.shape[0]
		num_sampled = len(sampled)
		gt_bboxes_bv = box_np_ops.center_to_corner_box2d(
			gt_bboxes[:, 0:2], gt_bboxes[:, 3:5], gt_bboxes[:, 6]
		)

		sp_boxes = np.stack([i['box3d_lidar'] for i in sampled], axis=0)
		boxes = np.concatenate([gt_bboxes, sp_boxes], axis=0).copy()
		sp_boxes_new = boxes[num_gt:]
		sp_boxes_bv = box_np_ops.center_to_corner_box2d(
			sp_boxes_new[:, 0:2], sp_boxes_new[:, 3:5], sp_boxes_new[:, 6]
		)

		total_bv = np.concatenate([gt_bboxes_bv, sp_boxes_bv], axis=0)
		coll_mat = box_collision_test(total_bv, total_bv)
		diag = np.arange(total_bv.shape[0])
		coll_mat[diag, diag] = False

		valid_samples = []
		for i in range(num_gt, num_gt + num_sampled):
			if coll_mat[i].any():
				coll_mat[i] = False
				coll_mat[:, i] = False
			else:
				valid_samples.append(sampled[i - num_gt])
		return valid_samples


class ObjectPaste:
	def __init__(self, db_sampler, sample_2d=False, stop_epoch=None) -> None:
		self.sampler_cfg = db_sampler
		self.sample_2d = sample_2d
		self.db_sampler = DataBaseSampler(**db_sampler)
		self.epoch = -1
		self.stop_epoch = stop_epoch

	def set_epoch(self, epoch: int) -> None:
		self.epoch = epoch

	@staticmethod
	def remove_points_in_boxes(points, boxes):
		masks = box_np_ops.points_in_rbbox(points.coord.numpy(), boxes)
		return points[np.logical_not(masks.any(-1))]

	def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
		if self.stop_epoch is not None and self.epoch >= self.stop_epoch:
			return data

		gt_bboxes_3d = data['gt_bboxes_3d']
		gt_labels_3d = data['gt_labels_3d']
		points = data['points']

		sampled_dict = self.db_sampler.sample_all(
			gt_bboxes_3d.tensor.numpy(),
			gt_labels_3d,
			img=data.get('img') if self.sample_2d else None,
		)

		if sampled_dict is None:
			return data

		sampled_gt_bboxes_3d = sampled_dict['gt_bboxes_3d']
		sampled_points = sampled_dict['points']
		sampled_gt_labels = sampled_dict['gt_labels_3d']

		data['gt_labels_3d'] = np.concatenate([gt_labels_3d, sampled_gt_labels], axis=0).astype(
			np.int64
		)
		data['gt_bboxes_3d'] = gt_bboxes_3d.new_box(
			np.concatenate([gt_bboxes_3d.tensor.numpy(), sampled_gt_bboxes_3d], axis=0)
		)
		points = self.remove_points_in_boxes(points, sampled_gt_bboxes_3d)
		data['points'] = points.cat([sampled_points, points])
		return data
