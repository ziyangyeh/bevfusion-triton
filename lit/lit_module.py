from __future__ import annotations

from typing import Any

import lightning as L
import torch
from torch import nn
from omegaconf import OmegaConf

from lit.batch_adapter import prepare_bevfusion_batch
from models.fusion_models.bevfusion import BEVFusion
from utils.config import register_omegaconf_resolvers


register_omegaconf_resolvers()


class BEVFusionLitModule(L.LightningModule):
	"""Lightning wrapper for BEVFusion.

	Validation/test result aggregation currently assumes single-GPU evaluation.
	"""

	def __init__(self, cfg: Any, loss_fn: nn.Module | None = None) -> None:
		super().__init__()
		self.cfg = cfg
		self.save_hyperparameters(ignore=['loss_fn'])

		model_cfg = self._cfg_get('model', {}) or {}
		if OmegaConf.is_config(model_cfg):
			model_cfg = OmegaConf.to_container(model_cfg, resolve=True)
		model_cfg = dict(model_cfg or {})
		self.model = BEVFusion(**model_cfg) if isinstance(model_cfg, dict) else BEVFusion()
		self.loss_fn = self._build_loss_fn(loss_fn)
		self._val_results: list[dict[str, Any]] = []
		self._test_results: list[dict[str, Any]] = []
		self._load_model_initialization_checkpoint()

	def _cfg_get(self, path: str, default: Any = None) -> Any:
		current = self.cfg
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

	def _build_loss_fn(self, loss_fn: Any) -> nn.Module | None:
		if isinstance(loss_fn, nn.Module):
			return loss_fn
		return None

	def _load_model_initialization_checkpoint(self) -> None:
		ckpt_path = self._cfg_get('load_from', None) or self._cfg_get('model_pretrained', None)
		if not ckpt_path:
			return

		state = torch.load(ckpt_path, map_location='cpu')
		state_dict = state.get('state_dict', state)
		missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
		print(
			f'Loaded model initialization checkpoint from {ckpt_path} '
			f'(missing={len(missing)}, unexpected={len(unexpected)})'
		)
		if missing:
			print(f'Initialization missing keys: {missing[:10]}')
		if unexpected:
			print(f'Initialization unexpected keys: {unexpected[:10]}')

	def _set_dataset_epoch(self, dataset: Any, epoch: int) -> None:
		if dataset is None:
			return
		pipeline = getattr(dataset, 'pipeline', None)
		transforms = getattr(pipeline, 'transforms', None)
		if transforms is None:
			return
		for transform in transforms:
			set_epoch = getattr(transform, 'set_epoch', None)
			if callable(set_epoch):
				set_epoch(epoch)

	def on_train_epoch_start(self) -> None:
		datamodule = getattr(self.trainer, 'datamodule', None)
		if datamodule is None:
			return
		self._set_dataset_epoch(getattr(datamodule, 'train_dataset', None), int(self.current_epoch))

	def on_validation_epoch_start(self) -> None:
		self._val_results = []

	def on_test_epoch_start(self) -> None:
		self._test_results = []

	def _prepare_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
		return prepare_bevfusion_batch(batch)

	def _move_batch_to_device(self, value: Any) -> Any:
		if torch.is_tensor(value):
			return value.to(self.device)
		if isinstance(value, list):
			return [self._move_batch_to_device(item) for item in value]
		if isinstance(value, tuple):
			return tuple(self._move_batch_to_device(item) for item in value)
		if isinstance(value, dict):
			return {key: self._move_batch_to_device(item) for key, item in value.items()}
		if hasattr(value, 'tensor') and torch.is_tensor(value.tensor):
			value = value.clone()
			value.tensor = value.tensor.to(self.device)
			return value
		return value

	def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
		batch = self._move_batch_to_device(batch)
		model_inputs = self._prepare_batch(batch)
		return self.model(**model_inputs)

	def _compute_loss(
		self,
		outputs: dict[str, Any],
		batch: dict[str, Any],
		model_inputs: dict[str, Any],
	) -> torch.Tensor | dict[str, torch.Tensor]:
		if self.loss_fn is not None:
			try:
				return self.loss_fn(outputs=outputs, batch=batch, model_inputs=model_inputs)
			except TypeError:
				try:
					return self.loss_fn(outputs, model_inputs)
				except TypeError:
					return self.loss_fn(outputs, batch)

		object_head = self.model.heads['object'] if 'object' in self.model.heads else None
		preds_dict = outputs.get('object', outputs)
		if object_head is None or not hasattr(object_head, 'loss'):
			if 'map' in outputs and isinstance(outputs['map'], dict):
				return outputs['map']
			raise RuntimeError(
				'BEVFusionLitModule requires either an external loss_fn or a supported head loss path.'
			)

		losses = object_head.loss(
			preds_dict,
			gt_bboxes_3d=model_inputs['gt_bboxes_3d'],
			gt_labels_3d=model_inputs['gt_labels_3d'],
		)
		if 'map' in outputs and isinstance(outputs['map'], dict):
			map_losses = outputs['map']
			total = losses['loss']
			for name, value in map_losses.items():
				losses[f'map/{name}'] = value
				if name == 'loss':
					total = total + value
			losses['loss'] = total
		return losses

	def _decode_predictions(
		self,
		outputs: dict[str, Any],
		model_inputs: dict[str, Any],
	) -> list[dict[str, Any]]:
		object_head = self.model.heads['object'] if 'object' in self.model.heads else None
		gt_masks_bev = model_inputs.get('gt_masks_bev', None)
		if object_head is None:
			map_outputs = outputs.get('map', None)
			if torch.is_tensor(map_outputs):
				results = []
				for batch_idx in range(map_outputs.shape[0]):
					result = {'masks_bev': map_outputs[batch_idx].detach().cpu()}
					if gt_masks_bev is not None:
						result['gt_masks_bev'] = gt_masks_bev[batch_idx].detach().cpu()
					results.append(result)
				return results
			return []
		preds_dict = outputs.get('object', outputs)
		proposal_slice = slice(-object_head.num_proposals, None)
		vel = preds_dict.get('vel', None)
		decoded = object_head.bbox_coder.decode(
			preds_dict['heatmap'][..., proposal_slice].sigmoid(),
			preds_dict['rot'][..., proposal_slice],
			preds_dict['dim'][..., proposal_slice],
			preds_dict['center'][..., proposal_slice],
			preds_dict['height'][..., proposal_slice],
			vel[..., proposal_slice] if vel is not None else None,
			filter=True,
		)

		metas = model_inputs.get('metas', [])
		results = []
		for batch_idx, pred in enumerate(decoded):
			meta = metas[batch_idx] if batch_idx < len(metas) else {}
			box_type_3d = meta.get('box_type_3d', None)
			boxes = pred['bboxes'].detach().cpu()
			if box_type_3d is not None:
				boxes = box_type_3d(boxes, box_dim=boxes.shape[-1])
			result = {
				'boxes_3d': boxes,
				'scores_3d': pred['scores'].detach().cpu(),
				'labels_3d': pred['labels'].detach().cpu(),
			}
			if gt_masks_bev is not None:
				result['gt_masks_bev'] = gt_masks_bev[batch_idx].detach().cpu()
			map_outputs = outputs.get('map', None)
			if torch.is_tensor(map_outputs):
				result['masks_bev'] = map_outputs[batch_idx].detach().cpu()
			results.append(result)
		return results

	def _evaluate_epoch_results(self, stage: str, results: list[dict[str, Any]]) -> None:
		if not results:
			return
		trainer = getattr(self, 'trainer', None)
		world_size = int(getattr(trainer, 'world_size', 1) or 1) if trainer is not None else 1
		if world_size > 1:
			return
		datamodule = getattr(trainer, 'datamodule', None) if trainer is not None else None
		dataset = None
		if stage == 'val':
			dataset = getattr(datamodule, 'val_dataset', None)
		elif stage == 'test':
			dataset = getattr(datamodule, 'test_dataset', None)
		if dataset is None or not hasattr(dataset, 'evaluate'):
			return

		metrics = dataset.evaluate(results)
		for name, value in metrics.items():
			self.log(
				f'{stage}/{name}',
				float(value),
				prog_bar=name.endswith('/map') or name.endswith('/nds'),
				on_step=False,
				on_epoch=True,
				sync_dist=False,
			)

	def _log_loss_dict(
		self, prefix: str, loss_dict: dict[str, Any], batch_size: int
	) -> torch.Tensor:
		if 'loss' not in loss_dict:
			raise KeyError('External loss dict must contain a "loss" key.')

		total_loss = loss_dict['loss']
		for name, value in loss_dict.items():
			if torch.is_tensor(value):
				log_value = value.detach()
			else:
				log_value = value
			self.log(
				f'{prefix}/{name}',
				log_value,
				prog_bar=(name == 'loss'),
				on_step=(prefix == 'train'),
				on_epoch=True,
				batch_size=batch_size,
			)
		return total_loss

	def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
		del batch_idx
		batch = self._move_batch_to_device(batch)
		model_inputs = self._prepare_batch(batch)
		outputs = self.model(**model_inputs)
		loss_output = self._compute_loss(outputs, batch, model_inputs)
		batch_size = batch['img'].shape[0]

		if isinstance(loss_output, dict):
			return self._log_loss_dict('train', loss_output, batch_size)

		self.log(
			'train/loss',
			loss_output.detach(),
			prog_bar=True,
			on_step=True,
			on_epoch=True,
			batch_size=batch_size,
		)
		return loss_output

	def validation_step(self, batch: dict[str, Any], batch_idx: int) -> Any:
		del batch_idx
		batch = self._move_batch_to_device(batch)
		model_inputs = self._prepare_batch(batch)
		outputs = self.model(**model_inputs)
		trainer = getattr(self, 'trainer', None)
		if not bool(getattr(trainer, 'sanity_checking', False)):
			self._val_results.extend(self._decode_predictions(outputs, model_inputs))
		loss_output = self._compute_loss(outputs, batch, model_inputs)
		batch_size = batch['img'].shape[0]
		if isinstance(loss_output, dict):
			self._log_loss_dict('val', loss_output, batch_size)
			return loss_output

		self.log(
			'val/loss',
			loss_output.detach(),
			prog_bar=True,
			on_step=False,
			on_epoch=True,
			batch_size=batch_size,
		)
		return loss_output

	def test_step(self, batch: dict[str, Any], batch_idx: int) -> Any:
		del batch_idx
		batch = self._move_batch_to_device(batch)
		model_inputs = self._prepare_batch(batch)
		outputs = self.model(**model_inputs)
		self._test_results.extend(self._decode_predictions(outputs, model_inputs))
		return outputs

	def predict_step(self, batch: dict[str, Any], batch_idx: int, dataloader_idx: int = 0) -> Any:
		del batch_idx, dataloader_idx
		return self.forward(batch)

	def on_validation_epoch_end(self) -> None:
		trainer = getattr(self, 'trainer', None)
		if bool(getattr(trainer, 'sanity_checking', False)):
			self._val_results = []
			return
		self._evaluate_epoch_results('val', self._val_results)
		self._val_results = []

	def on_test_epoch_end(self) -> None:
		self._evaluate_epoch_results('test', self._test_results)
		self._test_results = []

	def configure_optimizers(self):
		optimizer_cfg = dict(self._cfg_get('optimizer', {}) or {})
		optimizer_type = optimizer_cfg.pop('type', 'AdamW')
		lr = optimizer_cfg.pop('lr', 2e-4)
		weight_decay = optimizer_cfg.pop('weight_decay', 0.01)
		paramwise_cfg = dict(optimizer_cfg.pop('paramwise_cfg', {}) or {})

		parameters: Any = self.parameters()
		if paramwise_cfg:
			parameters = self._build_param_groups(
				lr=lr, weight_decay=weight_decay, paramwise_cfg=paramwise_cfg
			)

		if optimizer_type == 'AdamW':
			optimizer = torch.optim.AdamW(
				parameters, lr=lr, weight_decay=weight_decay, **optimizer_cfg
			)
		elif optimizer_type == 'Adam':
			optimizer = torch.optim.Adam(
				parameters, lr=lr, weight_decay=weight_decay, **optimizer_cfg
			)
		elif optimizer_type == 'SGD':
			optimizer = torch.optim.SGD(
				parameters, lr=lr, weight_decay=weight_decay, **optimizer_cfg
			)
		else:
			raise KeyError(f'Unsupported optimizer type: {optimizer_type}')

		scheduler_bundle = self._build_scheduler_bundle(optimizer, lr)
		if scheduler_bundle is None:
			return optimizer

		return {
			'optimizer': optimizer,
			'lr_scheduler': {
				'scheduler': scheduler_bundle['scheduler'],
				'interval': scheduler_bundle['interval'],
				'frequency': scheduler_bundle['frequency'],
				'monitor': scheduler_bundle['monitor'],
			},
		}

	def _build_scheduler_bundle(self, optimizer: torch.optim.Optimizer, base_lr: float):
		scheduler_cfg = self._cfg_get('lr_scheduler', None)
		if scheduler_cfg:
			scheduler_cfg = dict(scheduler_cfg)
			scheduler_type = scheduler_cfg.pop('type', 'MultiStepLR')
			interval = scheduler_cfg.pop('interval', 'epoch')
			frequency = scheduler_cfg.pop('frequency', 1)
			monitor = scheduler_cfg.pop('monitor', 'val/loss')

			if scheduler_type == 'MultiStepLR':
				scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, **scheduler_cfg)
			elif scheduler_type == 'CosineAnnealingLR':
				scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **scheduler_cfg)
			elif scheduler_type == 'StepLR':
				scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **scheduler_cfg)
			elif scheduler_type == 'OneCycleLR':
				scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, **scheduler_cfg)
			else:
				raise KeyError(f'Unsupported lr_scheduler type: {scheduler_type}')

			return {
				'scheduler': scheduler,
				'interval': interval,
				'frequency': frequency,
				'monitor': monitor,
			}

		lr_cfg = dict(self._cfg_get('lr_config', {}) or {})
		if not lr_cfg:
			return None

		policy = str(lr_cfg.get('policy', '')).lower()
		if policy == 'cosineannealing':
			return self._build_cosine_from_lr_config(optimizer, base_lr, lr_cfg)
		if policy == 'cyclic':
			return self._build_cyclic_from_lr_config(optimizer, base_lr, lr_cfg)
		raise KeyError(f'Unsupported lr_config policy: {lr_cfg.get("policy")}')

	def _build_param_groups(
		self, lr: float, weight_decay: float, paramwise_cfg: dict[str, Any]
	) -> list[dict[str, Any]]:
		custom_keys = dict(paramwise_cfg.get('custom_keys', {}) or {})
		sorted_keys = sorted(custom_keys.keys(), key=len, reverse=True)
		param_groups: list[dict[str, Any]] = []
		for name, param in self.named_parameters():
			if not param.requires_grad:
				continue

			group_lr = lr
			group_weight_decay = weight_decay
			for key in sorted_keys:
				if key in name:
					options = dict(custom_keys.get(key, {}) or {})
					if 'lr_mult' in options:
						group_lr = lr * float(options['lr_mult'])
					if 'decay_mult' in options:
						group_weight_decay = weight_decay * float(options['decay_mult'])
					break

			param_groups.append(
				{
					'params': [param],
					'lr': group_lr,
					'weight_decay': group_weight_decay,
				}
			)
		return param_groups

	def _build_cosine_from_lr_config(self, optimizer, base_lr: float, lr_cfg: dict[str, Any]):
		warmup = str(lr_cfg.get('warmup', '')).lower()
		warmup_iters = int(lr_cfg.get('warmup_iters', 0) or 0)
		warmup_ratio = float(lr_cfg.get('warmup_ratio', 1.0))
		min_lr_ratio = float(lr_cfg.get('min_lr_ratio', 0.0))
		trainer = getattr(self, '_trainer', None)
		total_steps = int(getattr(trainer, 'estimated_stepping_batches', 0) or 0)
		if total_steps <= 0:
			total_steps = max(int(self._cfg_get('max_epochs', 1)), 1)

		def lr_lambda(step: int) -> float:
			if warmup == 'linear' and warmup_iters > 0 and step < warmup_iters:
				alpha = step / max(warmup_iters, 1)
				return warmup_ratio + alpha * (1.0 - warmup_ratio)

			cosine_total = max(total_steps - warmup_iters, 1)
			cosine_step = max(step - warmup_iters, 0)
			progress = min(cosine_step / cosine_total, 1.0)
			cosine = 0.5 * (1.0 + torch.cos(torch.tensor(progress * torch.pi))).item()
			return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

		scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
		return {
			'scheduler': scheduler,
			'interval': 'step',
			'frequency': 1,
			'monitor': 'val/loss',
		}

	def _build_cyclic_from_lr_config(self, optimizer, base_lr: float, lr_cfg: dict[str, Any]):
		trainer = getattr(self, '_trainer', None)
		total_steps = int(getattr(trainer, 'estimated_stepping_batches', 0) or 0)
		if total_steps <= 0:
			total_steps = max(int(self._cfg_get('max_epochs', 1)), 1)

		momentum_cfg = dict(self._cfg_get('momentum_config', {}) or {})
		cycle_momentum = str(momentum_cfg.get('policy', '')).lower() == 'cyclic'
		if isinstance(optimizer, (torch.optim.Adam, torch.optim.AdamW)):
			cycle_momentum = False

		scheduler = torch.optim.lr_scheduler.OneCycleLR(
			optimizer,
			max_lr=base_lr,
			total_steps=total_steps,
			pct_start=0.4,
			anneal_strategy='cos',
			cycle_momentum=cycle_momentum,
		)
		return {
			'scheduler': scheduler,
			'interval': 'step',
			'frequency': 1,
			'monitor': 'val/loss',
		}


if __name__ == '__main__':
	from lit.lit_data_module import LitDataModule

	cfg = {
		'data': {
			'samples_per_gpu': 1,
			'workers_per_gpu': 0,
			'train': {
				'type': 'NuScenesDataset',
				'dataset_root': 'data/nuscenes-mini',
				'ann_file': 'data/nuscenes-mini/nuscenes_infos_train.pkl',
				'object_classes': [
					'car',
					'truck',
					'construction_vehicle',
					'bus',
					'trailer',
					'barrier',
					'motorcycle',
					'bicycle',
					'pedestrian',
					'traffic_cone',
				],
				'map_classes': [
					'drivable_area',
					'ped_crossing',
					'walkway',
					'stop_line',
					'carpark_area',
					'divider',
				],
				'modality': {
					'use_lidar': True,
					'use_camera': True,
					'use_radar': False,
					'use_map': False,
					'use_external': False,
				},
				'use_valid_flag': True,
				'pipeline': [
					{'type': 'LoadMultiViewImageFromFiles', 'to_float32': True},
					{
						'type': 'LoadPointsFromFile',
						'coord_type': 'LIDAR',
						'load_dim': 5,
						'use_dim': 5,
					},
					{
						'type': 'LoadPointsFromMultiSweeps',
						'sweeps_num': 1,
						'load_dim': 5,
						'use_dim': 5,
						'pad_empty_sweeps': True,
						'remove_close': True,
					},
					{
						'type': 'LoadAnnotations3D',
						'with_bbox_3d': True,
						'with_label_3d': True,
						'with_attr_label': False,
					},
					{
						'type': 'ImageAug3D',
						'final_dim': (256, 704),
						'resize_lim': (0.48, 0.48),
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
					{
						'type': 'PointsRangeFilter',
						'point_cloud_range': (-54.0, -54.0, -5.0, 54.0, 54.0, 3.0),
					},
					{
						'type': 'ObjectRangeFilter',
						'point_cloud_range': (-54.0, -54.0, -5.0, 54.0, 54.0, 3.0),
					},
					{
						'type': 'ObjectNameFilter',
						'classes': [
							'car',
							'truck',
							'construction_vehicle',
							'bus',
							'trailer',
							'barrier',
							'motorcycle',
							'bicycle',
							'pedestrian',
							'traffic_cone',
						],
					},
					{
						'type': 'ImageNormalize',
						'mean': [0.485, 0.456, 0.406],
						'std': [0.229, 0.224, 0.225],
					},
					{'type': 'PointShuffle'},
					{'type': 'GTDepth', 'keyframe_only': True},
				],
			},
		},
		'optimizer': {'type': 'AdamW', 'lr': 2e-4, 'weight_decay': 0.01},
	}

	datamodule = LitDataModule(cfg)
	datamodule.setup('fit')
	batch = next(iter(datamodule.train_dataloader()))

	module = BEVFusionLitModule(cfg)
	outputs = module(batch)

	print(sorted(outputs.keys()))
	print(outputs['object']['heatmap'].shape)
	print(outputs['features']['camera_bev'].shape)
	print(type(module.configure_optimizers()).__name__)
