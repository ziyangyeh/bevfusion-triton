from __future__ import annotations

import atexit
import os
import pickle
import signal
from pathlib import Path
from typing import Any

# Open3D's Wayland support is broken in the CUDA build (GLEW init failure,
# XSendEvent crash). Force X11/XWayland when running in a Wayland session.
if os.environ.get('XDG_SESSION_TYPE') == 'wayland':
    os.environ['XDG_SESSION_TYPE'] = 'x11'
    os.environ['GDK_BACKEND'] = 'x11'

import numpy as np
import open3d as o3d
import torch
from hydra import compose, initialize_config_dir
from PIL import Image, ImageDraw

from core.bbox import LiDARInstance3DBoxes
from lit.lit_data_module import LitDataModule
from lit.lit_module import BEVFusionLitModule


_CLASS_COLORS = {
	'car': (1.0, 0.3, 0.3),
	'truck': (1.0, 0.55, 0.2),
	'bus': (1.0, 0.85, 0.2),
	'trailer': (0.9, 0.6, 0.1),
	'construction_vehicle': (1.0, 0.75, 0.35),
	'pedestrian': (0.2, 1.0, 0.35),
	'motorcycle': (0.2, 0.85, 1.0),
	'bicycle': (0.2, 0.55, 1.0),
	'traffic_cone': (1.0, 0.45, 0.0),
	'barrier': (0.85, 0.2, 1.0),
}

_CAMERA_ORDER = [
	'CAM_FRONT_LEFT',
	'CAM_FRONT',
	'CAM_FRONT_RIGHT',
	'CAM_BACK_LEFT',
	'CAM_BACK',
	'CAM_BACK_RIGHT',
]

_STANDALONE_CONTEXTS: dict[tuple[str, str | None, str | None], dict[str, Any]] = {}
_CLEANUP_REGISTERED = False

_MASK_PALETTE = np.asarray([
	[76, 175, 80],
	[244, 67, 54],
	[255, 214, 102],
	[255, 112, 67],
	[156, 107, 255],
	[66, 165, 245],
	[104, 222, 222],
	[255, 109, 182],
], dtype=np.uint8)

_POINT_MODE_OPTIONS = ('default', 'gt_seg', 'pred_seg', 'rgb')
_BOX_EDGE_INDICES = np.asarray(
	[
		[0, 1], [1, 2], [2, 3], [3, 0],
		[4, 5], [5, 6], [6, 7], [7, 4],
		[0, 4], [1, 5], [2, 6], [3, 7],
	],
	dtype=np.int32,
)
_DEFAULT_BEV_MASK_BOUNDS = {
	'xbound': [-50.0, 50.0, 0.5],
	'ybound': [-50.0, 50.0, 0.5],
}

def cleanup_visualization_resources() -> None:
	global _STANDALONE_CONTEXTS

	for context in _STANDALONE_CONTEXTS.values():
		try:
			context['module'].cpu()
		except Exception:
			pass
	_STANDALONE_CONTEXTS.clear()
	if torch.cuda.is_available():
		try:
			torch.cuda.empty_cache()
		except Exception:
			pass


def _register_cleanup_handlers() -> None:
	global _CLEANUP_REGISTERED
	if _CLEANUP_REGISTERED:
		return
	atexit.register(cleanup_visualization_resources)

	def _make_handler(previous_handler):
		def _handler(signum, frame):
			cleanup_visualization_resources()
			if callable(previous_handler):
				previous_handler(signum, frame)
				return
			if previous_handler == signal.SIG_DFL:
				raise KeyboardInterrupt
		return _handler

	for sig in (signal.SIGINT, signal.SIGTERM):
		try:
			previous = signal.getsignal(sig)
			signal.signal(sig, _make_handler(previous))
		except Exception:
			pass
	_CLEANUP_REGISTERED = True


def _rgb_tuple_to_css(rgb: tuple[float, float, float]) -> str:
	r, g, b = rgb
	return f'rgb({int(round(r * 255))},{int(round(g * 255))},{int(round(b * 255))})'


def _class_color_for_label(label: int | None, class_names: list[str]) -> str:
	if label is None or label < 0 or label >= len(class_names):
		return '#ff3344'
	return _rgb_tuple_to_css(_CLASS_COLORS.get(class_names[int(label)], (1.0, 0.2, 0.2)))


def _mask_palette_color(class_idx: int) -> str:
	color = _MASK_PALETTE[int(class_idx) % len(_MASK_PALETTE)]
	return f'rgb({int(color[0])},{int(color[1])},{int(color[2])})'


def _color_string_to_rgb(color: str) -> np.ndarray:
	if color.startswith('#') and len(color) == 7:
		return np.asarray(
			[
				int(color[1:3], 16),
				int(color[3:5], 16),
				int(color[5:7], 16),
			],
			dtype=np.float64,
		) / 255.0
	if color.startswith('rgb(') and color.endswith(')'):
		values = color[4:-1].split(',')
		return np.asarray([float(v.strip()) for v in values], dtype=np.float64) / 255.0
	return np.asarray([0.81, 0.81, 0.81], dtype=np.float64)


def _point_colors_to_o3d_array(point_colors: np.ndarray | str, num_points: int) -> np.ndarray:
	if num_points == 0:
		return np.zeros((0, 3), dtype=np.float64)
	if isinstance(point_colors, str):
		return np.repeat(_color_string_to_rgb(point_colors)[None, :], num_points, axis=0)
	point_colors = np.asarray(point_colors, dtype=object)
	if point_colors.ndim == 2 and point_colors.shape[1] == 3:
		return point_colors.astype(np.float64)
	return np.stack([_color_string_to_rgb(str(color)) for color in point_colors], axis=0)


def _build_o3d_point_cloud(
	points_xyz: np.ndarray,
	*,
	point_colors: np.ndarray | str = '#cfcfcf',
	max_points: int | None = None,
):
	pcd = o3d.geometry.PointCloud()
	pcd.points = o3d.utility.Vector3dVector(np.asarray(points_xyz, dtype=np.float64))
	pcd.colors = o3d.utility.Vector3dVector(
		_point_colors_to_o3d_array(point_colors, len(points_xyz))
	)
	if max_points is not None and len(points_xyz) > max_points:
		ratio = float(max_points) / float(len(points_xyz))
		pcd = pcd.random_down_sample(max(min(ratio, 1.0), 1e-6))
	return pcd


def _build_pointcloud_data(
	points_xyz: np.ndarray,
	*,
	point_colors: np.ndarray | str = '#cfcfcf',
	seg_class: np.ndarray | None = None,
	max_points: int | None = None,
) -> dict[str, Any]:
	pointcloud = _build_o3d_point_cloud(
		points_xyz,
		point_colors=point_colors,
		max_points=max_points,
	)
	if seg_class is None:
		seg_class = np.full((len(points_xyz),), -1, dtype=np.int32)
	else:
		seg_class = np.asarray(seg_class, dtype=np.int32)
	return {
		'pointcloud': pointcloud,
		'seg_class': seg_class,
	}


def _build_o3d_labeled_linesets(
	boxes_arr: np.ndarray | None,
	labels: np.ndarray | None,
	class_names: list[str],
	default_color: str,
) -> list[dict[str, Any]]:
	if boxes_arr is None or len(boxes_arr) == 0:
		return []
	boxes = _to_box_obj(boxes_arr)
	centers = boxes.gravity_center.detach().cpu().numpy()
	dims = boxes.dims.detach().cpu().numpy()
	yaws = boxes.tensor[:, 6].detach().cpu().numpy()
	corners = boxes.corners.detach().cpu().numpy()
	primitives: list[dict[str, Any]] = []
	for idx, (center, dim, yaw, corners_i) in enumerate(zip(centers, dims, yaws, corners)):
		label = None if labels is None or idx >= len(labels) else int(labels[idx])
		color = _class_color_for_label(label, class_names) if label is not None else default_color
		rotation = o3d.geometry.get_rotation_matrix_from_xyz((0.0, 0.0, float(yaw)))
		obb = o3d.geometry.OrientedBoundingBox(center.astype(np.float64), rotation, dim.astype(np.float64))
		lineset = o3d.geometry.LineSet()
		# Use the box corners generated by LiDARInstance3DBoxes directly so the
		# rendered wireframe follows the same point ordering/rotation convention as
		# the model outputs and targets. The OBB is still kept as a semantic
		# primitive for future Open3D-native ops.
		lineset.points = o3d.utility.Vector3dVector(np.asarray(corners_i, dtype=np.float64))
		lineset.lines = o3d.utility.Vector2iVector(_BOX_EDGE_INDICES)
		line_colors = np.repeat(
			_color_string_to_rgb(color)[None, :],
			len(_BOX_EDGE_INDICES),
			axis=0,
		)
		lineset.colors = o3d.utility.Vector3dVector(line_colors)
		primitives.append(
			{
				'box_index': idx,
				'label': label,
				'color': color,
				'obb': obb,
				'lineset': lineset,
			}
		)
	return primitives


def _build_pointcloud(
	payload: dict[str, Any],
	*,
	max_points: int,
	point_mode: str,
):
	points = np.asarray(payload['points'])[:, :3]
	point_mode = point_mode if point_mode in _POINT_MODE_OPTIONS else 'default'
	if point_mode == 'gt_seg':
		seg_class = _point_seg_classes_from_bev_masks(
			points,
			payload.get('gt_masks_bev', None),
			payload.get('point_cloud_range', np.asarray([-54.0, -54.0, -5.0, 54.0, 54.0, 3.0])),
			payload.get('bev_mask_bounds', None),
		)
		point_colors = _seg_classes_to_color_strings(seg_class)
	elif point_mode == 'rgb':
		point_colors = _point_colors_from_rgb_images(payload, points)
		seg_class = np.full((len(points),), -1, dtype=np.int32)
	else:
		if point_mode == 'pred_seg':
			seg_class = _point_seg_classes_from_bev_masks(
				points,
				payload.get('pred_masks_bev', None),
				payload.get('point_cloud_range', np.asarray([-54.0, -54.0, -5.0, 54.0, 54.0, 3.0])),
				payload.get('bev_mask_bounds', None),
			)
			point_colors = _seg_classes_to_color_strings(seg_class)
		else:
			point_colors = '#cfcfcf'
			seg_class = np.full((len(points),), -1, dtype=np.int32)
	return _build_pointcloud_data(
		points,
		point_colors=point_colors,
		seg_class=seg_class,
		max_points=max_points,
	)


def _build_gt_linesets(
	payload: dict[str, Any],
	*,
	align_for_camera: bool = False,
) -> list[dict[str, Any]]:
	boxes = payload.get('gt_boxes', None)
	if align_for_camera:
		boxes = _visual_box_align_like_original(boxes)
	return _build_o3d_labeled_linesets(
		boxes,
		payload.get('gt_labels', None),
		payload.get('class_names', []),
		'#00ff66',
	)


def _build_pred_linesets(
	payload: dict[str, Any],
	*,
	topk: int | None,
	score_thresh: float | None,
	align_for_camera: bool = False,
) -> tuple[list[dict[str, Any]], np.ndarray | None]:
	pred_boxes, pred_scores, pred_labels = (
		get_detection_predictions_for_display(payload, topk=int(topk), score_thresh=float(score_thresh))
		if topk is not None and score_thresh is not None
		else (payload.get('pred_boxes_vis', None), payload.get('pred_scores_vis', None), payload.get('pred_labels_vis', None))
	)
	if align_for_camera:
		pred_boxes = _visual_box_align_like_original(pred_boxes)
	return (
		_build_o3d_labeled_linesets(
			pred_boxes,
			pred_labels,
			payload.get('class_names', []),
			'#ff3344',
		),
		None if pred_scores is None else np.asarray(pred_scores),
	)


def _to_box_obj(boxes: Any):
	if boxes is None:
		return None
	if hasattr(boxes, 'corners'):
		return boxes
	if torch.is_tensor(boxes):
		boxes = boxes.detach().cpu()
	return LiDARInstance3DBoxes(boxes, box_dim=boxes.shape[-1], origin=(0.5, 0.5, 0))


def _boxes_to_numpy(boxes: Any) -> np.ndarray | None:
	if boxes is None:
		return None
	if hasattr(boxes, 'tensor'):
		return boxes.tensor.detach().cpu().numpy()
	if torch.is_tensor(boxes):
		return boxes.detach().cpu().numpy()
	return np.asarray(boxes)


def _visual_box_align_like_original(boxes: Any):
	boxes_obj = _to_box_obj(boxes)
	if boxes_obj is None:
		return None
	tensor = boxes_obj.tensor.clone()
	if tensor.shape[1] >= 6:
		tensor[:, 2] -= tensor[:, 5] * 0.5
	return LiDARInstance3DBoxes(tensor, box_dim=tensor.shape[-1], origin=(0.5, 0.5, 0))


def _read_image_rgb(path: str | Path) -> np.ndarray | None:
	path = Path(path)
	if not path.exists():
		return None
	with Image.open(path) as image:
		return np.asarray(image.convert('RGB'))
def _colorize_binary_mask_stack(mask_stack: np.ndarray) -> np.ndarray:
	mask_stack = np.asarray(mask_stack)
	num_classes, height, width = mask_stack.shape
	palette = np.asarray(
		[_MASK_PALETTE[idx % len(_MASK_PALETTE)] for idx in range(max(num_classes, 1))],
		dtype=np.uint8,
	)
	canvas = np.zeros((height, width, 3), dtype=np.float32)
	for class_idx in range(num_classes):
		mask = mask_stack[class_idx] > 0
		if np.any(mask):
			canvas[mask] = 0.55 * canvas[mask] + 0.45 * palette[class_idx].astype(np.float32)
	return np.clip(canvas, 0, 255).astype(np.uint8)


def _decode_augmented_images(batch: dict[str, Any], sample_index: int) -> list[np.ndarray]:
	aug_imgs = batch['img'][sample_index].detach().cpu().numpy()
	mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
	std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
	images = []
	for image in aug_imgs:
		image = np.clip((image * std + mean) * 255.0, 0, 255).astype(np.uint8)
		images.append(image.transpose(1, 2, 0))
	return images


def _get_split_dataset(datamodule: LitDataModule, split: str):
	stage = 'fit' if split in {'train', 'val'} else 'test'
	datamodule.setup(stage)
	if split == 'train':
		return datamodule.train_dataset
	if split == 'val':
		return datamodule.val_dataset
	return datamodule.test_dataset


def compose_experiment_config(
	config_path: str | Path,
	*,
	dataset_root: str | None = None,
	checkpoint: str | None = None,
	extra_overrides: list[str] | None = None,
):
	config_path = Path(config_path).resolve()
	project_root = Path(__file__).resolve().parents[1]
	configs_root = project_root / 'configs'
	try:
		config_name = str(config_path.relative_to(configs_root).with_suffix(''))
	except ValueError:
		configs_root = config_path.parent
		config_name = config_path.stem
	overrides = [
		'dataloader.train.batch_size=1',
		'dataloader.val.batch_size=1',
		'dataloader.test.batch_size=1',
		'dataloader.num_workers=0',
	]
	if dataset_root is not None:
		overrides.append(f'dataset_root={dataset_root}')
	if checkpoint is not None:
		overrides.append(f'load_from={checkpoint}')
	if extra_overrides:
		overrides.extend(extra_overrides)
	with initialize_config_dir(version_base='1.3', config_dir=str(configs_root)):
		return compose(config_name=config_name, overrides=overrides)


def _extract_bev_mask_bounds(cfg: Any, split: str) -> dict[str, list[float]] | None:
	pipeline_key = 'train_pipeline' if split == 'train' else 'test_pipeline'
	pipeline = getattr(cfg, pipeline_key, None)
	if pipeline is None:
		return None
	for transform in pipeline:
		transform_type = transform.get('type', None) if hasattr(transform, 'get') else None
		if transform_type == 'LoadBEVSegmentation':
			xbound = list(transform.get('xbound'))
			ybound = list(transform.get('ybound'))
			return {'xbound': xbound, 'ybound': ybound}
	return None


def _get_default_bev_mask_bounds(dataset: Any | None) -> dict[str, list[float]] | None:
	map_classes = getattr(dataset, 'map_classes', None)
	if not map_classes:
		return None
	return {
		'xbound': list(_DEFAULT_BEV_MASK_BOUNDS['xbound']),
		'ybound': list(_DEFAULT_BEV_MASK_BOUNDS['ybound']),
	}


def _build_gt_masks_bev_for_dataset_sample(
	dataset: Any | None,
	dataset_index: int | None,
	*,
	lidar_aug_matrix: np.ndarray | None = None,
	bev_mask_bounds: dict[str, list[float]] | None = None,
) -> np.ndarray | None:
	if dataset is None or dataset_index is None:
		return None
	map_classes = getattr(dataset, 'map_classes', None)
	dataset_root = getattr(dataset, 'dataset_root', None)
	if not map_classes or dataset_root is None:
		return None
	data_info = dataset.get_data_info(int(dataset_index))
	if 'location' not in data_info or 'lidar2ego' not in data_info or 'ego2global' not in data_info:
		return None
	bounds = bev_mask_bounds or _get_default_bev_mask_bounds(dataset)
	if bounds is None:
		return None
	from datasets.pipelines.loading import LoadBEVSegmentation

	transform = LoadBEVSegmentation(
		dataset_root=dataset_root,
		xbound=tuple(bounds['xbound']),
		ybound=tuple(bounds['ybound']),
		classes=tuple(map_classes),
	)
	data = {
		'location': data_info['location'],
		'lidar2ego': np.asarray(data_info['lidar2ego'], dtype=np.float32),
		'ego2global': np.asarray(data_info['ego2global'], dtype=np.float32),
		'lidar_aug_matrix': (
			np.asarray(lidar_aug_matrix, dtype=np.float32)
			if lidar_aug_matrix is not None
			else np.eye(4, dtype=np.float32)
		),
	}
	return np.asarray(transform(data)['gt_masks_bev'])


def _make_standalone_context_key(
	config_path: str | Path,
	*,
	dataset_root: str | None = None,
	checkpoint: str | None = None,
) -> tuple[str, str | None, str | None]:
	return (
		str(Path(config_path).resolve()),
		None if dataset_root is None else str(Path(dataset_root)),
		None if checkpoint is None else str(Path(checkpoint)),
	)


def get_standalone_context(
	config_path: str | Path,
	*,
	dataset_root: str | None = None,
	checkpoint: str | None = None,
) -> dict[str, Any]:
	_register_cleanup_handlers()
	key = _make_standalone_context_key(
		config_path,
		dataset_root=dataset_root,
		checkpoint=checkpoint,
	)
	context = _STANDALONE_CONTEXTS.get(key)
	if context is not None:
		return context

	cfg = compose_experiment_config(
		config_path,
		dataset_root=dataset_root,
		checkpoint=checkpoint,
	)
	datamodule = LitDataModule(cfg)
	datasets = {
		split_name: _get_split_dataset(datamodule, split_name)
		for split_name in ('train', 'val', 'test')
	}

	module = BEVFusionLitModule(cfg)
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	module = module.to(device).eval()

	context = {
		'cfg': cfg,
		'module': module,
		'datasets': datasets,
	}
	_STANDALONE_CONTEXTS[key] = context
	return context


def get_standalone_dataset_length(
	config_path: str | Path,
	*,
	split: str,
	dataset_root: str | None = None,
	checkpoint: str | None = None,
) -> int:
	context = get_standalone_context(
		config_path,
		dataset_root=dataset_root,
		checkpoint=checkpoint,
	)
	dataset = context['datasets'].get(split)
	if dataset is None:
		raise RuntimeError(f'No dataset available for split={split}.')
	return len(dataset)


def _get_batch_for_index(context: dict[str, Any], split: str, index: int):
	dataset = context['datasets'].get(split)
	if dataset is None:
		raise RuntimeError(f'No dataset available for split={split}.')
	if index < 0 or index >= len(dataset):
		raise IndexError(f'index {index} is out of range for split={split} with len={len(dataset)}.')
	sample = dataset[index]
	from lit.lit_data_module import nuscenes_collate_fn
	return dataset, nuscenes_collate_fn([sample])


def build_standalone_payload(
	config_path: str | Path,
	*,
	split: str,
	index: int,
	dataset_root: str | None = None,
	checkpoint: str | None = None,
	topk: int = 30,
	score_thresh: float = 0.3,
) -> dict[str, Any]:
	context = get_standalone_context(
		config_path,
		dataset_root=dataset_root,
		checkpoint=checkpoint,
	)
	dataset, batch = _get_batch_for_index(context, split, index)
	with torch.no_grad():
		batch = context['module']._move_batch_to_device(batch)
		model_inputs = context['module']._prepare_batch(batch)
		outputs = context['module'].model(**model_inputs)
		decoded = context['module']._decode_predictions(outputs, model_inputs)
	return build_payload_from_runtime(
		dataset_cfg=context['cfg'],
		batch=batch,
		model_inputs=model_inputs,
		decoded=decoded,
		dataset=dataset,
		sample_index=0,
		split=split,
		dataset_index=index,
		topk=topk,
		score_thresh=score_thresh,
	)


def _detect_task(pred: dict[str, Any]) -> str:
	if 'boxes_3d' in pred:
		return 'det'
	if 'masks_bev' in pred:
		return 'seg'
	raise ValueError(f'Unsupported prediction payload keys: {list(pred.keys())}')


def build_payload_from_runtime(
	*,
	dataset_cfg: Any | None = None,
	batch: dict[str, Any],
	model_inputs: dict[str, Any],
	decoded: list[dict[str, Any]],
	dataset: Any | None,
	sample_index: int,
	split: str,
	dataset_index: int | None = None,
	topk: int = 30,
	score_thresh: float = 0.3,
) -> dict[str, Any]:
	pred = decoded[sample_index]
	task = _detect_task(pred)
	batch_meta = batch.get('metas', [None])[sample_index] if 'metas' in batch else None
	model_meta = model_inputs.get('metas', [None])[sample_index] if 'metas' in model_inputs else None
	meta = {}
	if isinstance(batch_meta, dict):
		meta.update(batch_meta)
	if isinstance(model_meta, dict):
		meta.update(model_meta)
	token = meta.get('token', f'{split}_{dataset_index if dataset_index is not None else sample_index}')
	bev_mask_bounds = (
		_extract_bev_mask_bounds(dataset_cfg, split)
		if dataset_cfg is not None
		else None
	)
	if bev_mask_bounds is None:
		bev_mask_bounds = _get_default_bev_mask_bounds(dataset)

	payload: dict[str, Any] = {
		'task': task,
		'token': token,
		'split': split,
		'dataset_index': dataset_index,
		'dataset_root': getattr(dataset, 'dataset_root', None),
		'meta': meta,
	}
	if bev_mask_bounds is not None:
		payload['bev_mask_bounds'] = bev_mask_bounds
	payload['point_cloud_range'] = np.asarray(getattr(dataset, 'point_cloud_range', [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]), dtype=np.float32)
	points = model_inputs['points'][sample_index]
	payload['points'] = points.detach().cpu().numpy() if torch.is_tensor(points) else np.asarray(points)
	payload['camera_images_model'] = _decode_augmented_images(batch, sample_index)
	camera_image_paths = meta.get('image_paths', None)
	if not camera_image_paths:
		camera_image_paths = meta.get('filename', [])
	payload['camera_image_paths'] = list(camera_image_paths)
	payload['camera_names'] = [
		Path(path).stem.split('__')[1] if '__' in Path(path).stem else Path(path).stem
		for path in payload['camera_image_paths']
	]
	if task == 'seg':
		payload['map_class_names'] = list(getattr(dataset, 'map_classes', [])) if dataset is not None else []
		payload['class_names'] = list(getattr(dataset, 'CLASSES', [])) if dataset is not None else []
		payload['gt_boxes'] = _boxes_to_numpy(model_inputs.get('gt_bboxes_3d', [None])[sample_index])
		gt_labels = model_inputs.get('gt_labels_3d', [None])[sample_index]
		payload['gt_labels'] = None if gt_labels is None else (
			gt_labels.detach().cpu().numpy() if torch.is_tensor(gt_labels) else np.asarray(gt_labels)
		)
		payload['pred_masks_bev'] = pred['masks_bev'].detach().cpu().numpy() if torch.is_tensor(pred['masks_bev']) else np.asarray(pred['masks_bev'])
		gt_masks = pred.get('gt_masks_bev', None)
		payload['gt_masks_bev'] = None if gt_masks is None else (
			gt_masks.detach().cpu().numpy() if torch.is_tensor(gt_masks) else np.asarray(gt_masks)
		)
		payload['lidar2image'] = model_inputs['lidar2image'][sample_index].detach().cpu().numpy()
		payload['img_aug_matrix'] = model_inputs['img_aug_matrix'][sample_index].detach().cpu().numpy()
		payload['lidar_aug_matrix'] = model_inputs['lidar_aug_matrix'][sample_index].detach().cpu().numpy()
		return payload

	payload['class_names'] = list(getattr(dataset, 'CLASSES', [])) if dataset is not None else []
	payload['gt_boxes'] = _boxes_to_numpy(model_inputs.get('gt_bboxes_3d', [None])[sample_index])
	gt_labels = model_inputs.get('gt_labels_3d', [None])[sample_index]
	payload['gt_labels'] = None if gt_labels is None else (
		gt_labels.detach().cpu().numpy() if torch.is_tensor(gt_labels) else np.asarray(gt_labels)
	)
	payload['pred_boxes'] = _boxes_to_numpy(pred['boxes_3d'])
	pred_scores = pred['scores_3d'].detach().cpu() if torch.is_tensor(pred['scores_3d']) else torch.as_tensor(pred['scores_3d'])
	pred_labels = pred['labels_3d'].detach().cpu() if torch.is_tensor(pred['labels_3d']) else torch.as_tensor(pred['labels_3d'])
	payload['pred_scores'] = pred_scores.numpy()
	payload['pred_labels'] = pred_labels.numpy()
	payload['gt_masks_bev'] = _build_gt_masks_bev_for_dataset_sample(
		dataset,
		dataset_index,
		lidar_aug_matrix=(
			model_inputs['lidar_aug_matrix'][sample_index].detach().cpu().numpy()
			if 'lidar_aug_matrix' in model_inputs
			else None
		),
		bev_mask_bounds=bev_mask_bounds,
	)
	if len(pred_scores) > 0:
		keep = pred_scores >= score_thresh
		if int(keep.sum().item()) == 0:
			topk_scores, topk_idx = torch.topk(pred_scores, k=min(topk, len(pred_scores)))
			payload['pred_boxes_vis'] = _boxes_to_numpy(pred['boxes_3d'][topk_idx])
			payload['pred_scores_vis'] = topk_scores.numpy()
			payload['pred_labels_vis'] = pred_labels[topk_idx].numpy()
		else:
			selected_scores = pred_scores[keep]
			selected_boxes = pred['boxes_3d'][keep]
			selected_labels = pred_labels[keep]
			if len(selected_scores) > topk:
				topk_scores, topk_idx = torch.topk(selected_scores, k=topk)
				payload['pred_boxes_vis'] = _boxes_to_numpy(selected_boxes[topk_idx])
				payload['pred_scores_vis'] = topk_scores.numpy()
				payload['pred_labels_vis'] = selected_labels[topk_idx].numpy()
			else:
				payload['pred_boxes_vis'] = _boxes_to_numpy(selected_boxes)
				payload['pred_scores_vis'] = selected_scores.numpy()
				payload['pred_labels_vis'] = selected_labels.numpy()
	else:
		payload['pred_boxes_vis'] = _boxes_to_numpy(pred['boxes_3d'])
		payload['pred_scores_vis'] = pred_scores.numpy()
		payload['pred_labels_vis'] = pred_labels.numpy()
	payload['lidar2image'] = model_inputs['lidar2image'][sample_index].detach().cpu().numpy()
	payload['img_aug_matrix'] = model_inputs['img_aug_matrix'][sample_index].detach().cpu().numpy()
	payload['lidar_aug_matrix'] = model_inputs['lidar_aug_matrix'][sample_index].detach().cpu().numpy()
	return payload


def save_runtime_payload(payload: dict[str, Any], output_dir: str | Path, *, stage: str) -> Path:
	output_dir = Path(output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)
	file_path = output_dir / f'{stage}_{payload["token"]}.pkl'
	with file_path.open('wb') as f:
		pickle.dump(payload, f)
	return file_path


def load_runtime_payload(path: str | Path) -> dict[str, Any]:
	with Path(path).open('rb') as f:
		return pickle.load(f)


def get_detection_predictions_for_display(
	payload: dict[str, Any],
	*,
	topk: int,
	score_thresh: float,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
	boxes = payload.get('pred_boxes', None)
	scores = payload.get('pred_scores', None)
	labels = payload.get('pred_labels', None)
	if boxes is None or scores is None or labels is None:
		return payload.get('pred_boxes_vis', None), payload.get('pred_scores_vis', None), payload.get('pred_labels_vis', None)
	boxes = np.asarray(boxes)
	scores = np.asarray(scores)
	labels = np.asarray(labels)
	if len(scores) == 0:
		return boxes, scores, labels
	keep = scores >= float(score_thresh)
	if int(np.sum(keep)) == 0:
		order = np.argsort(-scores)[: min(int(topk), len(scores))]
		return boxes[order], scores[order], labels[order]
	selected_boxes = boxes[keep]
	selected_scores = scores[keep]
	selected_labels = labels[keep]
	if len(selected_scores) > int(topk):
		order = np.argsort(-selected_scores)[: int(topk)]
		selected_boxes = selected_boxes[order]
		selected_scores = selected_scores[order]
		selected_labels = selected_labels[order]
	return selected_boxes, selected_scores, selected_labels


def _seg_classes_to_color_strings(seg_class: np.ndarray) -> np.ndarray:
	colors = np.full((len(seg_class),), '#cfcfcf', dtype=object)
	valid = seg_class >= 0
	if np.any(valid):
		colors[valid] = [_mask_palette_color(int(idx)) for idx in seg_class[valid]]
	return colors


def _point_seg_classes_from_bev_masks(
	points_xyz: np.ndarray,
	mask_stack: np.ndarray | None,
	point_cloud_range: np.ndarray,
	bev_mask_bounds: dict[str, list[float]] | None = None,
) -> np.ndarray:
	if mask_stack is None or len(points_xyz) == 0:
		return np.full((len(points_xyz),), -1, dtype=np.int32)
	mask_stack = np.asarray(mask_stack)
	if mask_stack.ndim != 3:
		return np.full((len(points_xyz),), -1, dtype=np.int32)
	num_classes, height, width = mask_stack.shape
	if bev_mask_bounds is not None:
		xbound = np.asarray(bev_mask_bounds['xbound'], dtype=np.float32)
		ybound = np.asarray(bev_mask_bounds['ybound'], dtype=np.float32)
		xmin, xmax = float(xbound[0]), float(xbound[1])
		ymin, ymax = float(ybound[0]), float(ybound[1])
	else:
		xmin, ymin, _, xmax, ymax, _ = np.asarray(point_cloud_range, dtype=np.float32)
	xspan = max(float(xmax - xmin), 1e-6)
	yspan = max(float(ymax - ymin), 1e-6)
	x = points_xyz[:, 0]
	y = points_xyz[:, 1]
	x_idx = np.floor((x - xmin) / xspan * width).astype(np.int64)
	y_idx = np.floor((y - ymin) / yspan * height).astype(np.int64)
	seg_class = np.full((len(points_xyz),), -1, dtype=np.int32)

	def _map_bev_indices(
		x_idx: np.ndarray,
		y_idx: np.ndarray,
		*,
		height: int,
		width: int,
		rotation: str,
	) -> tuple[np.ndarray, np.ndarray]:
		if rotation == 'cw90':
			# For a display-space clockwise 90-degree rotation where
			# display[row, col] = mask[height - 1 - col, row],
			# sampling needs the inverse mapping:
			# mask[row, col] = display[col, width - 1 - row].
			return x_idx, width - 1 - y_idx
		if rotation == 'ccw90':
			# For a display-space counter-clockwise 90-degree rotation where
			# display[row, col] = mask[col, width - 1 - row],
			# sampling needs the inverse mapping:
			# mask[row, col] = display[height - 1 - col, row].
			return height - 1 - x_idx, y_idx
		if rotation == 'none':
			return y_idx, x_idx
		raise ValueError(f'Unsupported BEV mask rotation: {rotation}')

	def _evaluate_mapping(row_idx: np.ndarray, col_idx: np.ndarray):
		valid = (col_idx >= 0) & (col_idx < width) & (row_idx >= 0) & (row_idx < height)
		if not np.any(valid):
			return -1, None
		valid_rows = row_idx[valid]
		valid_cols = col_idx[valid]
		class_scores = mask_stack[:, valid_rows, valid_cols]
		class_idx = np.argmax(class_scores, axis=0)
		has_class = np.any(class_scores > 0, axis=0)
		return int(has_class.sum()), (valid, class_idx, has_class)

	# `gt_masks_bev` is rasterized on a row/column grid, while LiDAR points are
	# indexed in x/y space. For the current NuScenes map pipeline, a 90-degree
	# clockwise lookup aligns the BEV mask best with the point cloud view.
	row_idx, col_idx = _map_bev_indices(
		x_idx,
		y_idx,
		height=height,
		width=width,
		rotation='cw90',
	)
	mapped = _evaluate_mapping(row_idx, col_idx)[1]
	if mapped is None:
		return seg_class
	valid, class_idx, has_class = mapped
	valid_indices = np.flatnonzero(valid)
	ground_like = (points_xyz[:, 2] >= -3.0) & (points_xyz[:, 2] <= 0.5)
	for idx, mask_ok in enumerate(has_class):
		point_idx = int(valid_indices[idx])
		if mask_ok and ground_like[point_idx]:
			seg_class[point_idx] = int(class_idx[idx] % max(num_classes, 1))
	return seg_class


def _point_colors_from_rgb_images(payload: dict[str, Any], points_xyz: np.ndarray) -> np.ndarray | str:
	images = payload.get('camera_images_model', [])
	if len(points_xyz) == 0 or not images:
		return '#cfcfcf'
	lidar2image = np.asarray(payload.get('lidar2image'))
	img_aug_matrix = np.asarray(payload.get('img_aug_matrix'))
	lidar_aug = np.asarray(payload.get('lidar_aug_matrix'))
	inv_lidar_aug = np.linalg.inv(lidar_aug[:3, :3])
	translated = points_xyz - lidar_aug[:3, 3]
	raw_points = translated @ inv_lidar_aug.T
	# Use a darker fallback for points that are not visible in any camera so
	# RGB-projected colors stand out more clearly in the viewer.
	colors = np.full((len(points_xyz), 3), 64, dtype=np.uint8)
	assigned = np.zeros((len(points_xyz),), dtype=bool)
	for cam_idx, image in enumerate(images):
		if cam_idx >= len(lidar2image):
			continue
		transform = np.asarray(lidar2image[cam_idx], dtype=np.float64).reshape(4, 4)
		homo = np.concatenate([raw_points, np.ones((len(raw_points), 1), dtype=np.float64)], axis=1)
		proj = homo @ transform.T
		depth = proj[:, 2]
		valid = depth > 1e-5
		if not np.any(valid):
			continue
		pts = proj[:, :2] / depth[:, None]
		pts_h = np.concatenate([pts, np.ones((len(pts), 1), dtype=np.float64)], axis=1)
		aug = np.asarray(img_aug_matrix[cam_idx], dtype=np.float64)
		pts_aug = pts_h @ aug[:3, :3].T
		pts_img = pts_aug[:, :2] + aug[:2, 3]
		h, w = image.shape[:2]
		cols = np.round(pts_img[:, 0]).astype(np.int64)
		rows = np.round(pts_img[:, 1]).astype(np.int64)
		in_frame = valid & (cols >= 0) & (cols < w) & (rows >= 0) & (rows < h)
		new_points = in_frame & (~assigned)
		if not np.any(new_points):
			continue
		colors[new_points] = image[rows[new_points], cols[new_points]]
		assigned[new_points] = True
		if np.all(assigned):
			break

	if np.any(assigned):
		# Mildly boost contrast/saturation for RGB-projected points. Raw camera
		# colors in overcast urban scenes are often low-saturation and can look
		# almost grayscale against a dark background.
		assigned_colors = colors[assigned].astype(np.float32) / 255.0
		luminance = assigned_colors.mean(axis=1, keepdims=True)
		enhanced = luminance + 1.35 * (assigned_colors - luminance)
		enhanced = np.clip(enhanced * 1.08, 0.0, 1.0)
		colors[assigned] = np.round(enhanced * 255.0).astype(np.uint8)

	return np.asarray([f'rgb({int(c[0])},{int(c[1])},{int(c[2])})' for c in colors], dtype=object)


def _project_lineset_edges_2d(
	primitives: list[dict[str, Any]],
	lidar2image: np.ndarray,
	*,
	lidar_aug_matrix: np.ndarray | None = None,
	img_aug_matrix: np.ndarray | None = None,
	scores: np.ndarray | None = None,
) -> list[dict[str, Any]]:
	if not primitives:
		return []
	transform = np.asarray(lidar2image, dtype=np.float64).reshape(4, 4)
	lidar_aug = None if lidar_aug_matrix is None else np.asarray(lidar_aug_matrix, dtype=np.float64)
	img_aug = None if img_aug_matrix is None else np.asarray(img_aug_matrix, dtype=np.float64)
	segments: list[dict[str, Any]] = []
	for i, primitive in enumerate(primitives):
		lineset = primitive['lineset']
		xyz = np.asarray(lineset.points, dtype=np.float64)
		if lidar_aug is not None:
			xyz = xyz - lidar_aug[:3, 3]
			xyz = xyz @ np.linalg.inv(lidar_aug[:3, :3]).T
		homo = np.concatenate([xyz, np.ones((xyz.shape[0], 1), dtype=np.float64)], axis=1)
		proj = homo @ transform.T
		depth = proj[:, 2:3]
		if np.any(depth <= 1e-5):
			continue
		pts = proj[:, :2] / depth
		if img_aug is not None:
			pts_h = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float64)], axis=1)
			pts_aug = pts_h @ img_aug[:3, :3].T
			pts = pts_aug[:, :2] + img_aug[:2, 3]
		if not np.isfinite(pts).all():
			continue
		for start, end in np.asarray(lineset.lines):
			segments.append({
				'box_index': i,
				'color': primitive['color'],
				'x': [float(pts[start, 0]), float(pts[end, 0]), None],
				'y': [float(pts[start, 1]), float(pts[end, 1]), None],
				'score_x': float(pts[0, 0]),
				'score_y': float(pts[0, 1]),
				'score': None if scores is None or i >= len(scores) else float(scores[i]),
			})
	return segments


def _draw_segments_on_image(
	image: np.ndarray,
	segments: list[dict[str, Any]],
	*,
	line_width: float,
) -> np.ndarray:
	canvas = Image.fromarray(np.asarray(image, dtype=np.uint8)).convert('RGB')
	draw = ImageDraw.Draw(canvas)
	for segment in segments:
		x0, x1 = segment['x'][0], segment['x'][1]
		y0, y1 = segment['y'][0], segment['y'][1]
		draw.line(
			[(float(x0), float(y0)), (float(x1), float(y1))],
			fill=segment['color'],
			width=max(1, int(round(line_width))),
		)
		if segment['score'] is not None:
			draw.text(
				(float(segment['score_x']), float(segment['score_y'])),
				f'{float(segment["score"]):.2f}',
				fill=segment['color'],
			)
	return np.asarray(canvas, dtype=np.uint8)


def _compose_camera_mosaic(name_to_image: dict[str, np.ndarray], *, title_height: int = 28) -> np.ndarray | None:
	images = [name_to_image.get(name) for name in _CAMERA_ORDER]
	valid = [img for img in images if img is not None]
	if not valid:
		return None
	cell_w = max(img.shape[1] for img in valid)
	cell_h = max(img.shape[0] for img in valid)
	canvas = Image.new('RGB', (cell_w * 3, (cell_h + title_height) * 2), '#05070b')
	draw = ImageDraw.Draw(canvas)
	for idx, cam_name in enumerate(_CAMERA_ORDER):
		img = name_to_image.get(cam_name)
		row = idx // 3
		col = idx % 3
		x0 = col * cell_w
		y0 = row * (cell_h + title_height)
		draw.text((x0 + 8, y0 + 4), cam_name, fill='white')
		if img is None:
			continue
		pil = Image.fromarray(np.asarray(img, dtype=np.uint8)).convert('RGB')
		if pil.size != (cell_w, cell_h):
			pil = pil.resize((cell_w, cell_h))
		canvas.paste(pil, (x0, y0 + title_height))
	return np.asarray(canvas, dtype=np.uint8)


def render_detection_camera_image(
	payload: dict[str, Any],
	*,
	view: str,
	show_gt: bool = True,
	show_pred: bool = True,
	box_line_width: float = 2.0,
	topk: int | None = None,
	score_thresh: float | None = None,
) -> np.ndarray | None:
	camera_names = payload.get('camera_names', [])
	if view == 'model':
		images = [np.asarray(image) for image in payload.get('camera_images_model', [])]
	else:
		images = [_read_image_rgb(path) for path in payload.get('camera_image_paths', [])]
	if not images:
		return None
	gt_linesets = _build_gt_linesets(payload, align_for_camera=True) if show_gt else []
	pred_linesets, pred_scores = _build_pred_linesets(
		payload,
		topk=topk,
		score_thresh=score_thresh,
		align_for_camera=True,
	) if show_pred else ([], None)
	name_to_image = {
		camera_names[idx]: np.asarray(image, dtype=np.uint8)
		for idx, image in enumerate(images)
		if idx < len(camera_names) and image is not None
	}
	name_to_index = {name: idx for idx, name in enumerate(camera_names)}
	rendered: dict[str, np.ndarray] = {}
	for cam_name in _CAMERA_ORDER:
		image = name_to_image.get(cam_name)
		if image is None:
			continue
		cam_idx = name_to_index[cam_name]
		segments: list[dict[str, Any]] = []
		if show_gt:
			segments.extend(
				_project_lineset_edges_2d(
					gt_linesets,
					payload['lidar2image'][cam_idx],
					lidar_aug_matrix=payload['lidar_aug_matrix'],
					img_aug_matrix=(payload['img_aug_matrix'][cam_idx] if view == 'model' else None),
				)
			)
		if show_pred:
			segments.extend(
				_project_lineset_edges_2d(
					pred_linesets,
					payload['lidar2image'][cam_idx],
					lidar_aug_matrix=payload['lidar_aug_matrix'],
					img_aug_matrix=(payload['img_aug_matrix'][cam_idx] if view == 'model' else None),
					scores=pred_scores,
				)
			)
		rendered[cam_name] = _draw_segments_on_image(image, segments, line_width=box_line_width)
	return _compose_camera_mosaic(rendered)


def render_segmentation_masks_image(payload: dict[str, Any]) -> np.ndarray | None:
	masks = render_segmentation_masks(payload)
	if not masks:
		return None
	names = list(masks.keys())
	images = [Image.fromarray(masks[name]).convert('RGB') for name in names]
	cell_w = max(img.size[0] for img in images)
	cell_h = max(img.size[1] for img in images)
	title_h = 28
	canvas = Image.new('RGB', (cell_w * len(images), cell_h + title_h), '#05070b')
	draw = ImageDraw.Draw(canvas)
	for idx, (name, img) in enumerate(zip(names, images)):
		x0 = idx * cell_w
		draw.text((x0 + 8, 4), name.upper(), fill='white')
		if img.size != (cell_w, cell_h):
			img = img.resize((cell_w, cell_h))
		canvas.paste(img, (x0, title_h))
	return np.asarray(canvas, dtype=np.uint8)


def render_segmentation_context_image(
	payload: dict[str, Any],
	*,
	view: str,
	show_gt: bool = True,
	box_line_width: float = 2.0,
) -> np.ndarray | None:
	return render_detection_camera_image(
		payload,
		view=view,
		show_gt=show_gt,
		show_pred=False,
		box_line_width=box_line_width,
	)


def _combine_3d_bounds(
	pointcloud: o3d.geometry.PointCloud,
	linesets: list[dict[str, Any]],
) -> np.ndarray:
	points = [np.asarray(pointcloud.points, dtype=np.float64)]
	for primitive in linesets:
		points.append(np.asarray(primitive['lineset'].points, dtype=np.float64))
	valid = [pts for pts in points if pts.size > 0]
	if not valid:
		return np.asarray([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]], dtype=np.float64)
	stacked = np.concatenate(valid, axis=0)
	return np.stack([stacked.min(axis=0), stacked.max(axis=0)], axis=0)


def _render_o3d_detection_image(
	pointcloud: o3d.geometry.PointCloud,
	gt_linesets: list[dict[str, Any]],
	pred_linesets: list[dict[str, Any]],
	*,
	point_size: float,
	line_width: float,
	width: int = 1600,
	height: int = 900,
) -> np.ndarray:
	renderer = o3d.visualization.rendering.OffscreenRenderer(int(width), int(height))
	scene = renderer.scene
	scene.set_background(np.asarray([0, 0, 0, 255], dtype=np.uint8))

	point_material = o3d.visualization.rendering.MaterialRecord()
	point_material.shader = 'defaultUnlit'
	point_material.point_size = max(float(point_size) * 3.0, 1.0)
	scene.add_geometry('pointcloud', pointcloud, point_material)

	line_material = o3d.visualization.rendering.MaterialRecord()
	line_material.shader = 'unlitLine'
	line_material.line_width = max(float(line_width) * 2.0, 1.0)

	for idx, primitive in enumerate(gt_linesets):
		scene.add_geometry(f'gt_{idx}', primitive['lineset'], line_material)
	for idx, primitive in enumerate(pred_linesets):
		scene.add_geometry(f'pred_{idx}', primitive['lineset'], line_material)

	bounds = _combine_3d_bounds(pointcloud, [*gt_linesets, *pred_linesets])
	center = np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
	extent = float(np.max(bounds[1] - bounds[0]))
	eye = np.asarray([0.0, 0.0, max(extent * 1.5, 60.0)], dtype=np.float32)
	up = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
	renderer.setup_camera(60.0, center, eye, up)
	image = np.asarray(renderer.render_to_image())
	scene.clear_geometry()
	del renderer
	return image


def _show_o3d_detection_window(
	pointcloud: o3d.geometry.PointCloud,
	gt_linesets: list[dict[str, Any]],
	pred_linesets: list[dict[str, Any]],
	*,
	point_size: float,
	line_width: float,
	window_name: str,
) -> None:
	geometries: list[Any] = [pointcloud]
	geometries.extend(primitive['lineset'] for primitive in gt_linesets)
	geometries.extend(primitive['lineset'] for primitive in pred_linesets)

	vis = o3d.visualization.Visualizer()
	vis.create_window(window_name=window_name, width=1600, height=900)
	try:
		for geometry in geometries:
			vis.add_geometry(geometry)
		render_option = vis.get_render_option()
		render_option.background_color = np.asarray([5 / 255.0, 7 / 255.0, 11 / 255.0], dtype=np.float64)
		render_option.point_size = max(float(point_size) * 3.0, 1.0)
		render_option.line_width = max(float(line_width) * 2.0, 1.0)

		bounds = _combine_3d_bounds(pointcloud, [*gt_linesets, *pred_linesets])
		center = bounds.mean(axis=0)
		extent = float(np.max(bounds[1] - bounds[0]))
		eye = center + np.asarray([0.0, 0.0, max(extent * 1.5, 60.0)], dtype=np.float64)
		up = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)

		view_control = vis.get_view_control()
		view_control.set_lookat(center.astype(np.float64))
		view_control.set_front((eye - center).astype(np.float64))
		view_control.set_up(up)
		view_control.set_zoom(0.35)

		vis.run()
	finally:
		vis.destroy_window()


def render_detection_pointcloud_image(
	payload: dict[str, Any],
	*,
	show_gt: bool = True,
	show_pred: bool = True,
	max_points: int = 50000,
	point_mode: str = 'default',
	point_size: float = 1.0,
	box_line_width: float = 1.0,
	topk: int | None = None,
	score_thresh: float | None = None,
	width: int = 1600,
	height: int = 900,
) -> np.ndarray:
	pointcloud_data = _build_pointcloud(payload, max_points=max_points, point_mode=point_mode)
	gt_linesets = _build_gt_linesets(payload) if show_gt else []
	pred_linesets, _ = _build_pred_linesets(
		payload,
		topk=topk,
		score_thresh=score_thresh,
	) if show_pred else ([], None)
	return _render_o3d_detection_image(
		pointcloud_data['pointcloud'],
		gt_linesets,
		pred_linesets,
		point_size=point_size,
		line_width=box_line_width,
		width=width,
		height=height,
	)


def show_detection_pointcloud_window(
	payload: dict[str, Any],
	*,
	show_gt: bool = True,
	show_pred: bool = True,
	max_points: int = 50000,
	point_mode: str = 'default',
	point_size: float = 1.0,
	box_line_width: float = 1.0,
	topk: int | None = None,
	score_thresh: float | None = None,
) -> None:
	pointcloud_data = _build_pointcloud(payload, max_points=max_points, point_mode=point_mode)
	gt_linesets = _build_gt_linesets(payload) if show_gt else []
	pred_linesets, _ = _build_pred_linesets(
		payload,
		topk=topk,
		score_thresh=score_thresh,
	) if show_pred else ([], None)
	_show_o3d_detection_window(
		pointcloud_data['pointcloud'],
		gt_linesets,
		pred_linesets,
		point_size=point_size,
		line_width=box_line_width,
		window_name=f"BEVFusion DET {payload['token']}",
	)


def render_segmentation_pointcloud_image(
	payload: dict[str, Any],
	*,
	show_gt: bool = True,
	max_points: int = 50000,
	point_mode: str = 'default',
	point_size: float = 1.0,
	box_line_width: float = 1.0,
	width: int = 1600,
	height: int = 900,
) -> np.ndarray:
	if show_gt:
		pointcloud_data = _build_pointcloud(payload, max_points=max_points, point_mode=point_mode)
		gt_linesets = _build_gt_linesets(payload)
	else:
		pointcloud_data = _build_pointcloud(payload, max_points=max_points, point_mode=point_mode)
		gt_linesets = []
	return _render_o3d_detection_image(
		pointcloud_data['pointcloud'],
		gt_linesets,
		[],
		point_size=point_size,
		line_width=box_line_width,
		width=width,
		height=height,
	)


def show_segmentation_pointcloud_window(
	payload: dict[str, Any],
	*,
	show_gt: bool = True,
	max_points: int = 50000,
	point_mode: str = 'default',
	point_size: float = 1.0,
	box_line_width: float = 1.0,
) -> None:
	pointcloud_data = _build_pointcloud(payload, max_points=max_points, point_mode=point_mode)
	gt_linesets = _build_gt_linesets(payload) if show_gt else []
	_show_o3d_detection_window(
		pointcloud_data['pointcloud'],
		gt_linesets,
		[],
		point_size=point_size,
		line_width=box_line_width,
		window_name=f"BEVFusion SEG {payload['token']}",
	)


def render_segmentation_masks(payload: dict[str, Any]) -> dict[str, np.ndarray]:
	out: dict[str, np.ndarray] = {}
	pred_masks = np.asarray(payload['pred_masks_bev'], dtype=np.float32)
	out['pred'] = _colorize_binary_mask_stack(pred_masks >= 0.5)
	gt_masks = payload.get('gt_masks_bev', None)
	if gt_masks is not None:
		gt_bin = np.asarray(gt_masks) > 0
		out['gt'] = _colorize_binary_mask_stack(gt_bin)
		pred_any = (pred_masks >= 0.5).any(axis=0)
		gt_any = gt_bin.any(axis=0)
		overlay = np.zeros_like(out['pred'], dtype=np.uint8)
		overlay[np.logical_and(gt_any, ~pred_any)] = np.array([0, 255, 0], dtype=np.uint8)
		overlay[np.logical_and(pred_any, ~gt_any)] = np.array([255, 0, 0], dtype=np.uint8)
		overlay[np.logical_and(pred_any, gt_any)] = np.array([255, 255, 0], dtype=np.uint8)
		out['overlay'] = overlay
	return out


def _save_image(image: np.ndarray | None, path: Path) -> Path | None:
	if image is None:
		return None
	Image.fromarray(image).save(path)
	return path


def save_detection_visuals(
	payload: dict[str, Any],
	output_dir: str | Path,
	*,
	show_gt: bool = True,
	show_pred: bool = True,
	point_mode: str = 'default',
	point_size: float = 1.0,
	box_line_width: float = 1.0,
	topk: int | None = 30,
	score_thresh: float | None = 0.3,
) -> dict[str, Path]:
	output_dir = Path(output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)
	token = payload['token']
	result: dict[str, Path] = {}
	pointcloud_path = output_dir / f'{token}_pointcloud.png'
	_save_image(
		render_detection_pointcloud_image(
			payload,
			show_gt=show_gt,
			show_pred=show_pred,
			point_mode=point_mode,
			point_size=point_size,
			box_line_width=box_line_width,
			topk=topk,
			score_thresh=score_thresh,
		),
		pointcloud_path,
	)
	result['pointcloud'] = pointcloud_path
	for view in ('raw', 'model'):
		camera_image = render_detection_camera_image(
			payload,
			view=view,
			show_gt=show_gt,
			show_pred=show_pred,
			box_line_width=box_line_width,
			topk=topk,
			score_thresh=score_thresh,
		)
		camera_path = _save_image(camera_image, output_dir / f'{token}_cameras_{view}.png')
		if camera_path is not None:
			result[f'cameras_{view}'] = camera_path
	return result


def save_segmentation_visuals(
	payload: dict[str, Any],
	output_dir: str | Path,
	*,
	show_gt: bool = True,
	point_mode: str = 'default',
	point_size: float = 1.0,
	box_line_width: float = 1.0,
) -> dict[str, Path]:
	output_dir = Path(output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)
	token = payload['token']
	result: dict[str, Path] = {}
	pointcloud_path = output_dir / f'{token}_pointcloud.png'
	_save_image(
		render_segmentation_pointcloud_image(
			payload,
			show_gt=show_gt,
			point_mode=point_mode,
			point_size=point_size,
			box_line_width=box_line_width,
		),
		pointcloud_path,
	)
	result['pointcloud'] = pointcloud_path
	mask_image = render_segmentation_masks_image(payload)
	mask_path = _save_image(mask_image, output_dir / f'{token}_maps.png')
	if mask_path is not None:
		result['maps'] = mask_path
	context_image = render_segmentation_context_image(
		payload,
		view='model',
		show_gt=show_gt,
		box_line_width=box_line_width,
	)
	context_path = _save_image(context_image, output_dir / f'{token}_cameras_model.png')
	if context_path is not None:
		result['cameras_model'] = context_path
	return result
