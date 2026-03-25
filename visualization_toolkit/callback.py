from __future__ import annotations

from pathlib import Path
from typing import Any

import lightning as L
import torch

from .core import build_payload_from_runtime, save_runtime_payload


class VisualizationDumpCallback(L.Callback):
	"""Dump shared visualization payloads for runtime CLI inspection."""

	def __init__(
		self,
		output_dir: str = 'outputs/visualize_runtime_payloads',
		*,
		enable_train: bool = False,
		enable_val: bool = True,
		enable_test: bool = True,
		save_results: bool = True,
		max_samples_per_stage: int = 1,
		every_n_epochs: int = 1,
	) -> None:
		super().__init__()
		self.output_dir = Path(output_dir)
		self.enable_train = enable_train
		self.enable_val = enable_val
		self.enable_test = enable_test
		self.save_results = save_results
		self.max_samples_per_stage = max(1, int(max_samples_per_stage))
		self.every_n_epochs = max(1, int(every_n_epochs))
		self._stage_counts = {'train': 0, 'val': 0, 'test': 0}

	def _reset_counts(self, stage: str) -> None:
		self._stage_counts[stage] = 0

	def on_train_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
		del trainer, pl_module
		self._reset_counts('train')

	def on_validation_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
		del trainer, pl_module
		self._reset_counts('val')

	def on_test_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
		del trainer, pl_module
		self._reset_counts('test')

	def _should_dump(self, trainer: L.Trainer, stage: str) -> bool:
		if not self.save_results:
			return False
		if stage == 'train' and not self.enable_train:
			return False
		if stage == 'val' and not self.enable_val:
			return False
		if stage == 'test' and not self.enable_test:
			return False
		if stage in {'train', 'val'} and (trainer.current_epoch % self.every_n_epochs) != 0:
			return False
		return self._stage_counts[stage] < self.max_samples_per_stage

	def _dump_batch(
		self,
		trainer: L.Trainer,
		pl_module: Any,
		batch: dict[str, Any],
		stage: str,
		batch_idx: int,
	) -> None:
		if not self._should_dump(trainer, stage):
			return
		with torch.no_grad():
			batch = pl_module._move_batch_to_device(batch)
			model_inputs = pl_module._prepare_batch(batch)
			outputs = pl_module.model(**model_inputs)
			decoded = pl_module._decode_predictions(outputs, model_inputs)
		datamodule = getattr(trainer, 'datamodule', None)
		dataset = None
		if stage == 'train':
			dataset = getattr(datamodule, 'train_dataset', None)
		elif stage == 'val':
			dataset = getattr(datamodule, 'val_dataset', None)
		elif stage == 'test':
			dataset = getattr(datamodule, 'test_dataset', None)
		for sample_index in range(min(len(decoded), self.max_samples_per_stage - self._stage_counts[stage])):
			payload = build_payload_from_runtime(
				batch=batch,
				model_inputs=model_inputs,
				decoded=decoded,
				dataset=dataset,
				sample_index=sample_index,
				split=stage,
				dataset_index=batch_idx * max(len(decoded), 1) + sample_index,
			)
			save_runtime_payload(payload, self.output_dir / stage, stage=stage)
			self._stage_counts[stage] += 1
			if self._stage_counts[stage] >= self.max_samples_per_stage:
				break

	def on_train_batch_end(
		self,
		trainer: L.Trainer,
		pl_module: L.LightningModule,
		outputs: Any,
		batch: dict[str, Any],
		batch_idx: int,
	) -> None:
		del outputs
		self._dump_batch(trainer, pl_module, batch, 'train', batch_idx)

	def on_validation_batch_end(
		self,
		trainer: L.Trainer,
		pl_module: L.LightningModule,
		outputs: Any,
		batch: dict[str, Any],
		batch_idx: int,
		dataloader_idx: int = 0,
	) -> None:
		del outputs, dataloader_idx
		self._dump_batch(trainer, pl_module, batch, 'val', batch_idx)

	def on_test_batch_end(
		self,
		trainer: L.Trainer,
		pl_module: L.LightningModule,
		outputs: Any,
		batch: dict[str, Any],
		batch_idx: int,
		dataloader_idx: int = 0,
	) -> None:
		del outputs, dataloader_idx
		self._dump_batch(trainer, pl_module, batch, 'test', batch_idx)
