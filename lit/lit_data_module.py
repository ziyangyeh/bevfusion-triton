import lightning as L
import numpy as np
from os import path as osp
from collections.abc import Mapping
import torch
from torch.utils.data import DataLoader

from datasets import NuScenesDataset


def _cfg_get(cfg, path, default=None):
	current = cfg
	for key in path.split('.'):
		if isinstance(current, dict):
			if key not in current:
				return default
			current = current[key]
		else:
			if not hasattr(current, key):
				return default
			current = getattr(current, key)
	return current


def _to_tensor_if_possible(value):
	if torch.is_tensor(value):
		return value
	if isinstance(value, np.ndarray):
		return torch.from_numpy(value)
	if isinstance(value, (int, float, bool, np.number)):
		return torch.tensor(value)
	if isinstance(value, list) and value:
		converted = [_to_tensor_if_possible(item) for item in value]
		if all(torch.is_tensor(item) for item in converted):
			first_shape = converted[0].shape
			if all(item.shape == first_shape for item in converted):
				return torch.stack(converted)
	return value


def _has_nuscenes_map_expansion(dataset_root):
	if not dataset_root:
		return False
	expansion_dir = osp.join(dataset_root, 'maps', 'expansion')
	required = (
		'singapore-onenorth.json',
		'singapore-hollandvillage.json',
		'singapore-queenstown.json',
		'boston-seaport.json',
	)
	return all(osp.exists(osp.join(expansion_dir, name)) for name in required)


def nuscenes_collate_fn(batch):
	batch = [sample for sample in batch if sample is not None]
	if not batch:
		return {}

	collated = {}
	keys = batch[0].keys()
	list_only_keys = {
		'points',
		'radar',
		'gt_bboxes_3d',
		'gt_labels_3d',
		'gt_names',
		'gt_names_3d',
		'ann_info',
		'metas',
		'image_paths',
		'lidar_path',
		'sweeps',
		'token',
		'sample_idx',
		'box_type_3d',
		'box_mode_3d',
	}

	for key in keys:
		values = [sample[key] for sample in batch]
		if key in list_only_keys:
			collated[key] = values
			continue

		converted = [_to_tensor_if_possible(value) for value in values]
		if all(torch.is_tensor(value) for value in converted):
			first_shape = converted[0].shape
			if all(value.shape == first_shape for value in converted):
				collated[key] = torch.stack(converted)
				continue
		collated[key] = values
	return collated


class LitDataModule(L.LightningDataModule):
	def __init__(self, cfg):
		super().__init__()
		self.cfg = cfg
		self.train_dataset = None
		self.val_dataset = None
		self.test_dataset = None

	def _unwrap_dataset_cfg(self, split_cfg):
		if split_cfg is None:
			return None
		if isinstance(split_cfg, Mapping) and isinstance(split_cfg.get('dataset'), Mapping):
			dataset_cfg = dict(split_cfg['dataset'])
			wrapper_type = split_cfg.get('type', None)
			if wrapper_type is not None:
				dataset_cfg.setdefault('_dataset_wrapper_type', wrapper_type)
			return dataset_cfg
		return split_cfg

	def _sanitize_pipeline(self, pipeline, dataset_root):
		if pipeline is None:
			return None

		sanitized = []
		has_map_expansion = _has_nuscenes_map_expansion(dataset_root)
		dbinfo_path = osp.join(dataset_root, 'nuscenes_dbinfos_train.pkl') if dataset_root else None

		for transform in pipeline:
			if not isinstance(transform, Mapping):
				sanitized.append(transform)
				continue

			transform_type = transform.get('type')
			if transform_type == 'LoadBEVSegmentation' and not has_map_expansion:
				continue
			if transform_type == 'ObjectPaste' and (not dbinfo_path or not osp.exists(dbinfo_path)):
				continue
			sanitized.append(transform)

		return sanitized

	def _make_default_pipeline(self, split: str, split_cfg):
		dataset_root = split_cfg.get('dataset_root', _cfg_get(self.cfg, 'dataset_root'))
		object_classes = split_cfg.get('object_classes', _cfg_get(self.cfg, 'object_classes', []))
		map_classes = split_cfg.get('map_classes', _cfg_get(self.cfg, 'map_classes', []))

		load_dim = int(_cfg_get(self.cfg, 'load_dim', 5))
		use_dim = _cfg_get(self.cfg, 'use_dim', 5)
		reduce_beams = _cfg_get(self.cfg, 'reduce_beams', None)
		load_augmented = _cfg_get(self.cfg, 'load_augmented', None)
		point_cloud_range = tuple(
			_cfg_get(self.cfg, 'point_cloud_range', (-51.2, -51.2, -5.0, 51.2, 51.2, 3.0))
		)
		image_size = tuple(_cfg_get(self.cfg, 'image_size', (256, 704)))

		augment2d_resize = _cfg_get(self.cfg, 'augment2d.resize', [[0.48, 0.48], [0.48, 0.48]])
		augment2d_rotate = tuple(_cfg_get(self.cfg, 'augment2d.rotate', [0.0, 0.0]))
		gridmask_prob = float(_cfg_get(self.cfg, 'augment2d.gridmask.prob', 0.0))
		gridmask_fixed_prob = bool(_cfg_get(self.cfg, 'augment2d.gridmask.fixed_prob', True))
		augment3d_scale = tuple(_cfg_get(self.cfg, 'augment3d.scale', [1.0, 1.0]))
		augment3d_rotate = tuple(_cfg_get(self.cfg, 'augment3d.rotate', [0.0, 0.0]))
		augment3d_translate = float(_cfg_get(self.cfg, 'augment3d.translate', 0.0))
		xbound = tuple(_cfg_get(self.cfg, 'map_xbound', (-50.0, 50.0, 0.5)))
		ybound = tuple(_cfg_get(self.cfg, 'map_ybound', (-50.0, 50.0, 0.5)))
		max_epochs = int(_cfg_get(self.cfg, 'max_epochs', 0))
		gt_paste_stop_epoch = _cfg_get(self.cfg, 'gt_paste_stop_epoch', -1)

		is_train = split == 'train'
		pipeline = [
			{'type': 'LoadMultiViewImageFromFiles', 'to_float32': True},
			{
				'type': 'LoadPointsFromFile',
				'coord_type': 'LIDAR',
				'load_dim': load_dim,
				'use_dim': use_dim,
			},
			{
				'type': 'LoadPointsFromMultiSweeps',
				'sweeps_num': 9,
				'load_dim': load_dim,
				'use_dim': use_dim,
				'pad_empty_sweeps': True,
				'remove_close': True,
			},
			{
				'type': 'LoadAnnotations3D',
				'with_bbox_3d': True,
				'with_label_3d': True,
				'with_attr_label': False,
			},
		]

		if reduce_beams is not None:
			pipeline[1]['reduce_beams'] = reduce_beams
			pipeline[2]['reduce_beams'] = reduce_beams
		if load_augmented is not None:
			pipeline[1]['load_augmented'] = load_augmented
			pipeline[2]['load_augmented'] = load_augmented

		if is_train:
			dbinfo_path = (
				osp.join(dataset_root, 'nuscenes_dbinfos_train.pkl') if dataset_root else None
			)
			if dbinfo_path and osp.exists(dbinfo_path):
				pipeline.append(
					{
						'type': 'ObjectPaste',
						'stop_epoch': gt_paste_stop_epoch,
						'db_sampler': {
							'dataset_root': dataset_root,
							'info_path': dbinfo_path,
							'rate': 1.0,
							'prepare': {
								'filter_by_difficulty': [-1],
								'filter_by_min_points': {name: 5 for name in object_classes},
							},
							'classes': object_classes,
							'sample_groups': {name: 2 for name in object_classes},
							'points_loader': {
								'type': 'LoadPointsFromFile',
								'coord_type': 'LIDAR',
								'load_dim': load_dim,
								'use_dim': use_dim,
								**(
									{'reduce_beams': reduce_beams}
									if reduce_beams is not None
									else {}
								),
							},
						},
					}
				)

		if is_train:
			pipeline.extend(
				[
					{
						'type': 'ImageAug3D',
						'final_dim': image_size,
						'resize_lim': tuple(augment2d_resize[0]),
						'bot_pct_lim': (0.0, 0.0),
						'rot_lim': augment2d_rotate,
						'rand_flip': True,
						'is_train': True,
					},
					{
						'type': 'GlobalRotScaleTrans',
						'resize_lim': augment3d_scale,
						'rot_lim': augment3d_rotate,
						'trans_lim': augment3d_translate,
						'is_train': True,
					},
				]
			)
		else:
			pipeline.extend(
				[
					{
						'type': 'ImageAug3D',
						'final_dim': image_size,
						'resize_lim': tuple(augment2d_resize[1]),
						'bot_pct_lim': (0.0, 0.0),
						'rot_lim': (0.0, 0.0),
						'rand_flip': False,
						'is_train': False,
					},
					{
						'type': 'GlobalRotScaleTrans',
						'resize_lim': (1.0, 1.0),
						'rot_lim': (0.0, 0.0),
						'trans_lim': 0.0,
						'is_train': False,
					},
				]
			)

		has_map_expansion = _has_nuscenes_map_expansion(dataset_root)
		if map_classes and has_map_expansion:
			pipeline.append(
				{
					'type': 'LoadBEVSegmentation',
					'dataset_root': dataset_root,
					'xbound': xbound,
					'ybound': ybound,
					'classes': map_classes,
				}
			)

		if is_train:
			pipeline.append({'type': 'RandomFlip3D'})

		pipeline.extend(
			[
				{'type': 'PointsRangeFilter', 'point_cloud_range': point_cloud_range},
			]
		)
		if is_train:
			pipeline.extend(
				[
					{'type': 'ObjectRangeFilter', 'point_cloud_range': point_cloud_range},
					{'type': 'ObjectNameFilter', 'classes': object_classes},
				]
			)

		pipeline.append(
			{'type': 'ImageNormalize', 'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
		)

		if is_train:
			pipeline.extend(
				[
					{
						'type': 'GridMask',
						'use_h': True,
						'use_w': True,
						'max_epoch': max_epochs,
						'rotate': 1,
						'offset': False,
						'ratio': 0.5,
						'mode': 1,
						'prob': gridmask_prob,
						'fixed_prob': gridmask_fixed_prob,
					},
					{'type': 'PointShuffle'},
				]
			)

		collect_keys = ['img', 'points', 'gt_bboxes_3d', 'gt_labels_3d']
		if map_classes and has_map_expansion:
			collect_keys.append('gt_masks_bev')

		pipeline.extend(
			[
				{'type': 'DefaultFormatBundle3D', 'classes': object_classes},
				{
					'type': 'Collect3D',
					'keys': collect_keys,
					'meta_keys': [
						'camera_intrinsics',
						'camera2ego',
						'lidar2ego',
						'lidar2camera',
						'camera2lidar',
						'lidar2image',
						'img_aug_matrix',
						'lidar_aug_matrix',
					],
				},
			]
		)

		if is_train:
			pipeline.append({'type': 'GTDepth', 'keyframe_only': True})
		return pipeline

	def _build_dataset(self, split_cfg):
		if split_cfg is None:
			return None

		split_cfg = self._unwrap_dataset_cfg(split_cfg)

		dataset_type = (
			split_cfg.get('type', 'NuScenesDataset')
			if isinstance(split_cfg, Mapping)
			else _cfg_get(split_cfg, 'type', 'NuScenesDataset')
		)
		if dataset_type != 'NuScenesDataset':
			raise KeyError(f'Unsupported dataset type: {dataset_type}')

		pipeline = split_cfg.get('pipeline')
		if pipeline is None:
			split = 'train'
			if split_cfg.get('test_mode', False):
				split = 'test'
			elif split_cfg is _cfg_get(self.cfg, 'data.val'):
				split = 'val'
			if split in {'val', 'test'}:
				pipeline = _cfg_get(self.cfg, 'evaluation.pipeline', None)
			if pipeline is None:
				pipeline = self._make_default_pipeline(split, split_cfg)
		pipeline = self._sanitize_pipeline(pipeline, split_cfg.get('dataset_root'))

		return NuScenesDataset(
			ann_file=split_cfg['ann_file'],
			dataset_root=split_cfg['dataset_root'],
			pipeline=pipeline,
			object_classes=split_cfg.get('object_classes'),
			map_classes=split_cfg.get('map_classes'),
			load_interval=split_cfg.get('load_interval', 1),
			with_velocity=split_cfg.get('with_velocity', True),
			modality=split_cfg.get('modality'),
			box_type_3d=split_cfg.get('box_type_3d', 'LiDAR'),
			filter_empty_gt=split_cfg.get('filter_empty_gt', True),
			test_mode=split_cfg.get('test_mode', False),
			eval_version=split_cfg.get('eval_version', 'detection_cvpr_2019'),
			use_valid_flag=split_cfg.get('use_valid_flag', False),
		)

	def setup(self, stage=None):
		if stage in (None, 'fit'):
			self.train_dataset = self._build_dataset(_cfg_get(self.cfg, 'data.train'))
			self.val_dataset = self._build_dataset(_cfg_get(self.cfg, 'data.val'))
		if stage in (None, 'test', 'predict'):
			self.test_dataset = self._build_dataset(_cfg_get(self.cfg, 'data.test'))

	def train_dataloader(self):
		return DataLoader(
			self.train_dataset,
			batch_size=_cfg_get(
				self.cfg,
				'dataloader.train.batch_size',
				_cfg_get(self.cfg, 'data.samples_per_gpu', 1),
			),
			shuffle=_cfg_get(self.cfg, 'dataloader.train.shuffle', True),
			num_workers=_cfg_get(
				self.cfg, 'dataloader.num_workers', _cfg_get(self.cfg, 'data.workers_per_gpu', 0)
			),
			collate_fn=nuscenes_collate_fn,
			pin_memory=_cfg_get(self.cfg, 'dataloader.pin_memory', True),
		)

	def val_dataloader(self):
		if self.val_dataset is None:
			return None
		return DataLoader(
			self.val_dataset,
			batch_size=_cfg_get(
				self.cfg,
				'dataloader.val.batch_size',
				_cfg_get(self.cfg, 'data.val.samples_per_gpu', 1),
			),
			shuffle=_cfg_get(self.cfg, 'dataloader.val.shuffle', False),
			num_workers=_cfg_get(
				self.cfg, 'dataloader.num_workers', _cfg_get(self.cfg, 'data.workers_per_gpu', 0)
			),
			collate_fn=nuscenes_collate_fn,
			pin_memory=_cfg_get(self.cfg, 'dataloader.pin_memory', True),
		)

	def test_dataloader(self):
		if self.test_dataset is None:
			return None
		return DataLoader(
			self.test_dataset,
			batch_size=_cfg_get(
				self.cfg,
				'dataloader.test.batch_size',
				_cfg_get(self.cfg, 'data.test.samples_per_gpu', 1),
			),
			shuffle=_cfg_get(self.cfg, 'dataloader.test.shuffle', False),
			num_workers=_cfg_get(
				self.cfg, 'dataloader.num_workers', _cfg_get(self.cfg, 'data.workers_per_gpu', 0)
			),
			collate_fn=nuscenes_collate_fn,
			pin_memory=_cfg_get(self.cfg, 'dataloader.pin_memory', True),
		)
