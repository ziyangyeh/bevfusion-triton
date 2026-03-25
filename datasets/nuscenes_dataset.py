import json
import pickle
import tempfile
from os import path as osp
from typing import Any, Dict

import numpy as np
import pyquaternion
import torch
from nuscenes.utils.data_classes import Box as NuScenesBox
from pyquaternion import Quaternion

from utils import mkdir_or_exist
from core.bbox import LiDARInstance3DBoxes
from .custom_3d import Custom3DDataset


class NuScenesDataset(Custom3DDataset):
	NameMapping = {
		'movable_object.barrier': 'barrier',
		'vehicle.bicycle': 'bicycle',
		'vehicle.bus.bendy': 'bus',
		'vehicle.bus.rigid': 'bus',
		'vehicle.car': 'car',
		'vehicle.construction': 'construction_vehicle',
		'vehicle.motorcycle': 'motorcycle',
		'human.pedestrian.adult': 'pedestrian',
		'human.pedestrian.child': 'pedestrian',
		'human.pedestrian.construction_worker': 'pedestrian',
		'human.pedestrian.police_officer': 'pedestrian',
		'movable_object.trafficcone': 'traffic_cone',
		'vehicle.trailer': 'trailer',
		'vehicle.truck': 'truck',
	}
	DefaultAttribute = {
		'car': 'vehicle.parked',
		'pedestrian': 'pedestrian.moving',
		'trailer': 'vehicle.parked',
		'truck': 'vehicle.parked',
		'bus': 'vehicle.moving',
		'motorcycle': 'cycle.without_rider',
		'construction_vehicle': 'vehicle.parked',
		'bicycle': 'cycle.without_rider',
		'barrier': '',
		'traffic_cone': '',
	}
	AttrMapping = {
		'cycle.with_rider': 0,
		'cycle.without_rider': 1,
		'pedestrian.moving': 2,
		'pedestrian.standing': 3,
		'pedestrian.sitting_lying_down': 4,
		'vehicle.moving': 5,
		'vehicle.parked': 6,
		'vehicle.stopped': 7,
	}
	AttrMapping_rev = [
		'cycle.with_rider',
		'cycle.without_rider',
		'pedestrian.moving',
		'pedestrian.standing',
		'pedestrian.sitting_lying_down',
		'vehicle.moving',
		'vehicle.parked',
		'vehicle.stopped',
	]
	ErrNameMapping = {
		'trans_err': 'mATE',
		'scale_err': 'mASE',
		'orient_err': 'mAOE',
		'vel_err': 'mAVE',
		'attr_err': 'mAAE',
	}
	CLASSES = (
		'car',
		'truck',
		'trailer',
		'bus',
		'construction_vehicle',
		'bicycle',
		'motorcycle',
		'pedestrian',
		'traffic_cone',
		'barrier',
	)

	def __init__(
		self,
		ann_file,
		pipeline=None,
		dataset_root=None,
		object_classes=None,
		map_classes=None,
		load_interval=1,
		with_velocity=True,
		modality=None,
		box_type_3d='LiDAR',
		filter_empty_gt=True,
		test_mode=False,
		eval_version='detection_cvpr_2019',
		use_valid_flag=False,
	) -> None:
		self.load_interval = load_interval
		self.use_valid_flag = use_valid_flag
		super().__init__(
			dataset_root=dataset_root,
			ann_file=ann_file,
			pipeline=pipeline,
			classes=object_classes,
			modality=modality,
			box_type_3d=box_type_3d,
			filter_empty_gt=filter_empty_gt,
			test_mode=test_mode,
		)
		self.map_classes = map_classes
		self.with_velocity = with_velocity
		self.eval_version = eval_version
		from nuscenes.eval.detection.config import config_factory

		self.eval_detection_configs = config_factory(self.eval_version)
		if self.modality is None:
			self.modality = dict(
				use_camera=False,
				use_lidar=True,
				use_radar=False,
				use_map=False,
				use_external=False,
			)

	def get_cat_ids(self, idx):
		info = self.data_infos[idx]
		if self.use_valid_flag:
			mask = info['valid_flag']
			gt_names = set(info['gt_names'][mask])
		else:
			gt_names = set(info['gt_names'])

		cat_ids = []
		for name in gt_names:
			if name in self.CLASSES:
				cat_ids.append(self.cat2id[name])
		return cat_ids

	def load_annotations(self, ann_file):
		with open(ann_file, 'rb') as f:
			data = pickle.load(f)
		data_infos = list(sorted(data['infos'], key=lambda e: e['timestamp']))
		data_infos = data_infos[:: self.load_interval]
		self.metadata = data['metadata']
		self.version = self.metadata['version']
		return data_infos

	def get_data_info(self, index: int) -> Dict[str, Any]:
		info = self.data_infos[index]
		data = dict(
			token=info['token'],
			sample_idx=info['token'],
			lidar_path=info['lidar_path'],
			sweeps=info['sweeps'],
			timestamp=info['timestamp'],
			location=info.get('location', None),
		)
		if self.modality.get('use_radar', False):
			data['radar'] = info.get('radars', None)

		if data['location'] is None:
			data.pop('location')
		if 'radar' in data and data['radar'] is None:
			data.pop('radar')

		ego2global = np.eye(4).astype(np.float32)
		ego2global[:3, :3] = Quaternion(info['ego2global_rotation']).rotation_matrix
		ego2global[:3, 3] = info['ego2global_translation']
		data['ego2global'] = ego2global

		lidar2ego = np.eye(4).astype(np.float32)
		lidar2ego[:3, :3] = Quaternion(info['lidar2ego_rotation']).rotation_matrix
		lidar2ego[:3, 3] = info['lidar2ego_translation']
		data['lidar2ego'] = lidar2ego

		if self.modality['use_camera']:
			data['image_paths'] = []
			data['lidar2camera'] = []
			data['lidar2image'] = []
			data['camera2ego'] = []
			data['camera_intrinsics'] = []
			data['camera2lidar'] = []

			for _, camera_info in info['cams'].items():
				data['image_paths'].append(camera_info['data_path'])

				lidar2camera_r = np.linalg.inv(camera_info['sensor2lidar_rotation'])
				lidar2camera_t = camera_info['sensor2lidar_translation'] @ lidar2camera_r.T
				lidar2camera_rt = np.eye(4).astype(np.float32)
				lidar2camera_rt[:3, :3] = lidar2camera_r.T
				lidar2camera_rt[3, :3] = -lidar2camera_t
				data['lidar2camera'].append(lidar2camera_rt.T)

				camera_intrinsics = np.eye(4).astype(np.float32)
				camera_intrinsics[:3, :3] = camera_info['cam_intrinsic']
				data['camera_intrinsics'].append(camera_intrinsics)

				lidar2image = camera_intrinsics @ lidar2camera_rt.T
				data['lidar2image'].append(lidar2image)

				camera2ego = np.eye(4).astype(np.float32)
				camera2ego[:3, :3] = Quaternion(camera_info['sensor2ego_rotation']).rotation_matrix
				camera2ego[:3, 3] = camera_info['sensor2ego_translation']
				data['camera2ego'].append(camera2ego)

				camera2lidar = np.eye(4).astype(np.float32)
				camera2lidar[:3, :3] = camera_info['sensor2lidar_rotation']
				camera2lidar[:3, 3] = camera_info['sensor2lidar_translation']
				data['camera2lidar'].append(camera2lidar)

		annos = self.get_ann_info(index)
		data['ann_info'] = annos
		return data

	def get_ann_info(self, index):
		info = self.data_infos[index]
		if self.use_valid_flag:
			mask = info['valid_flag']
		else:
			mask = info['num_lidar_pts'] > 0
		gt_bboxes_3d = info['gt_boxes'][mask]
		gt_names_3d = info['gt_names'][mask]
		gt_labels_3d = []
		for cat in gt_names_3d:
			if cat in self.CLASSES:
				gt_labels_3d.append(self.CLASSES.index(cat))
			else:
				gt_labels_3d.append(-1)
		gt_labels_3d = np.array(gt_labels_3d)

		if self.with_velocity:
			gt_velocity = info['gt_velocity'][mask]
			nan_mask = np.isnan(gt_velocity[:, 0])
			gt_velocity[nan_mask] = [0.0, 0.0]
			gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)

		gt_bboxes_3d = LiDARInstance3DBoxes(
			gt_bboxes_3d, box_dim=gt_bboxes_3d.shape[-1], origin=(0.5, 0.5, 0)
		).convert_to(self.box_mode_3d)

		return dict(
			gt_bboxes_3d=gt_bboxes_3d,
			gt_labels_3d=gt_labels_3d,
			gt_names=gt_names_3d,
		)

	def _format_bbox(self, results, jsonfile_prefix=None):
		nusc_annos = {}
		mapped_class_names = self.CLASSES

		print('Start to convert detection format...')
		for sample_id, det in enumerate(results):
			annos = []
			boxes = output_to_nusc_box(det)
			sample_token = self.data_infos[sample_id]['token']
			boxes = lidar_nusc_box_to_global(
				self.data_infos[sample_id],
				boxes,
				mapped_class_names,
				self.eval_detection_configs,
				self.eval_version,
			)
			for box in boxes:
				name = mapped_class_names[box.label]
				if np.sqrt(box.velocity[0] ** 2 + box.velocity[1] ** 2) > 0.2:
					if name in ['car', 'construction_vehicle', 'bus', 'truck', 'trailer']:
						attr = 'vehicle.moving'
					elif name in ['bicycle', 'motorcycle']:
						attr = 'cycle.with_rider'
					else:
						attr = NuScenesDataset.DefaultAttribute[name]
				else:
					if name in ['pedestrian']:
						attr = 'pedestrian.standing'
					elif name in ['bus']:
						attr = 'vehicle.stopped'
					else:
						attr = NuScenesDataset.DefaultAttribute[name]

				nusc_anno = dict(
					sample_token=sample_token,
					translation=box.center.tolist(),
					size=box.wlh.tolist(),
					rotation=box.orientation.elements.tolist(),
					velocity=box.velocity[:2].tolist(),
					detection_name=name,
					detection_score=box.score,
					attribute_name=attr,
				)
				annos.append(nusc_anno)
			nusc_annos[sample_token] = annos
		nusc_submissions = {'meta': self.modality, 'results': nusc_annos}

		mkdir_or_exist(jsonfile_prefix)
		res_path = osp.join(jsonfile_prefix, 'results_nusc.json')
		print('Results writes to', res_path)
		with open(res_path, 'w') as f:
			json.dump(nusc_submissions, f)
		return res_path

	def _evaluate_single(self, result_path, logger=None, metric='bbox', result_name='pts_bbox'):
		del logger, metric, result_name
		from nuscenes import NuScenes
		from nuscenes.eval.detection.evaluate import DetectionEval

		output_dir = osp.join(*osp.split(result_path)[:-1])
		nusc = NuScenes(version=self.version, dataroot=self.dataset_root, verbose=False)
		eval_set_map = {
			'v1.0-mini': 'mini_val',
			'v1.0-trainval': 'val',
		}
		nusc_eval = DetectionEval(
			nusc,
			config=self.eval_detection_configs,
			result_path=result_path,
			eval_set=eval_set_map[self.version],
			output_dir=output_dir,
			verbose=False,
		)
		nusc_eval.main(render_curves=False)

		with open(osp.join(output_dir, 'metrics_detail.json'), 'r') as f:
			metrics = json.load(f)
		detail = {}
		for name in self.CLASSES:
			for k, v in metrics['label_aps'][name].items():
				detail[f'object/{name}_ap_dist_{k}'] = float(f'{v:.4f}')
			for k, v in metrics['label_tp_errors'][name].items():
				detail[f'object/{name}_{k}'] = float(f'{v:.4f}')
			for k, v in metrics['tp_errors'].items():
				detail[f'object/{self.ErrNameMapping[k]}'] = float(f'{v:.4f}')

		detail['object/nds'] = metrics['nd_score']
		detail['object/map'] = metrics['mean_ap']
		return detail

	def format_results(self, results, jsonfile_prefix=None):
		assert isinstance(results, list), 'results must be a list'
		assert len(results) == len(self), (
			'The length of results is not equal to the dataset len: {} != {}'.format(
				len(results), len(self)
			)
		)

		if jsonfile_prefix is None:
			tmp_dir = tempfile.TemporaryDirectory()
			jsonfile_prefix = osp.join(tmp_dir.name, 'results')
		else:
			tmp_dir = None

		result_files = self._format_bbox(results, jsonfile_prefix)
		return result_files, tmp_dir

	def evaluate_map(self, results):
		thresholds = torch.tensor([0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65])

		num_classes = len(self.map_classes)
		num_thresholds = len(thresholds)

		tp = torch.zeros(num_classes, num_thresholds)
		fp = torch.zeros(num_classes, num_thresholds)
		fn = torch.zeros(num_classes, num_thresholds)

		for result in results:
			pred = result['masks_bev']
			label = result['gt_masks_bev']

			pred = pred.detach().reshape(num_classes, -1)
			label = label.detach().bool().reshape(num_classes, -1)

			pred = pred[:, :, None] >= thresholds
			label = label[:, :, None]

			tp += (pred & label).sum(dim=1)
			fp += (pred & ~label).sum(dim=1)
			fn += (~pred & label).sum(dim=1)

		ious = tp / (tp + fp + fn + 1e-7)

		metrics = {}
		for index, name in enumerate(self.map_classes):
			metrics[f'map/{name}/iou@max'] = ious[index].max().item()
			for threshold, iou in zip(thresholds, ious[index]):
				metrics[f'map/{name}/iou@{threshold.item():.2f}'] = iou.item()
		metrics['map/mean/iou@max'] = ious.max(dim=1).values.mean().item()
		return metrics

	def evaluate(
		self, results, metric='bbox', jsonfile_prefix=None, result_names=['pts_bbox'], **kwargs
	):
		del metric, kwargs
		metrics = {}

		if results and 'masks_bev' in results[0]:
			metrics.update(self.evaluate_map(results))

		if results and 'boxes_3d' in results[0]:
			result_files, tmp_dir = self.format_results(results, jsonfile_prefix)
			if isinstance(result_files, dict):
				for name in result_names:
					print('Evaluating bboxes of {}'.format(name))
					ret_dict = self._evaluate_single(result_files[name])
				metrics.update(ret_dict)
			elif isinstance(result_files, str):
				metrics.update(self._evaluate_single(result_files))

			if tmp_dir is not None:
				tmp_dir.cleanup()

		return metrics


def output_to_nusc_box(detection):
	box3d = detection['boxes_3d']
	scores = detection['scores_3d'].numpy()
	labels = detection['labels_3d'].numpy()

	box_gravity_center = box3d.gravity_center.numpy()
	box_dims = box3d.dims.numpy()
	box_yaw = box3d.yaw.numpy()
	box_yaw = -box_yaw - np.pi / 2

	box_list = []
	for i in range(len(box3d)):
		quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box_yaw[i])
		velocity = (*box3d.tensor[i, 7:9], 0.0)
		box = NuScenesBox(
			box_gravity_center[i],
			box_dims[i],
			quat,
			label=labels[i],
			score=scores[i],
			velocity=velocity,
		)
		box_list.append(box)
	return box_list


def lidar_nusc_box_to_global(
	info, boxes, classes, eval_configs, eval_version='detection_cvpr_2019'
):
	del eval_version
	box_list = []
	for box in boxes:
		box.rotate(pyquaternion.Quaternion(info['lidar2ego_rotation']))
		box.translate(np.array(info['lidar2ego_translation']))

		cls_range_map = eval_configs.class_range
		radius = np.linalg.norm(box.center[:2], 2)
		det_range = cls_range_map[classes[box.label]]
		if radius > det_range:
			continue

		box.rotate(pyquaternion.Quaternion(info['ego2global_rotation']))
		box.translate(np.array(info['ego2global_translation']))
		box_list.append(box)
	return box_list
