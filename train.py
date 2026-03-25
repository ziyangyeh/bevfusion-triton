from __future__ import annotations

from typing import Any

import hydra
import lightning as L
import torch
from hydra.core.hydra_config import HydraConfig
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from omegaconf import DictConfig, OmegaConf

from lit.lit_data_module import LitDataModule
from lit.lit_module import BEVFusionLitModule
from utils.config import register_omegaconf_resolvers
from visualization_toolkit.callback import VisualizationDumpCallback


register_omegaconf_resolvers()


def _to_plain_config(cfg: Any, *, resolve: bool = True) -> dict[str, Any]:
	if cfg is None:
		return {}
	if isinstance(cfg, dict):
		return cfg
	return OmegaConf.to_container(cfg, resolve=resolve, throw_on_missing=False)


def _cfg_get(cfg: dict[str, Any], path: str, default: Any = None) -> Any:
	current: Any = cfg
	for key in path.split('.'):
		if not isinstance(current, dict) or key not in current:
			return default
		current = current[key]
	return current


def build_callbacks(cfg: dict[str, Any]) -> list[Any]:
	callbacks: list[Any] = []
	trainer_cfg = dict(cfg.get('trainer', {}) or {})
	if trainer_cfg.get('enable_checkpointing', True):
		checkpoint_cfg = dict(cfg.get('checkpoint', {}) or {})
		callbacks.append(
			ModelCheckpoint(
				dirpath=checkpoint_cfg.get('dirpath'),
				filename=checkpoint_cfg.get('filename', '{epoch}-{step}-{val/loss:.4f}'),
				monitor=checkpoint_cfg.get('monitor', 'val/loss'),
				mode=checkpoint_cfg.get('mode', 'min'),
				save_top_k=checkpoint_cfg.get('save_top_k', 1),
				save_last=checkpoint_cfg.get('save_last', True),
				every_n_epochs=checkpoint_cfg.get('every_n_epochs', 1),
			)
		)

	if cfg.get('lr_monitor', True):
		callbacks.append(
			LearningRateMonitor(logging_interval=_cfg_get(cfg, 'lr_monitor_interval', 'epoch'))
		)

	vis_cfg = dict(cfg.get('visualization_runtime', {}) or {})
	if vis_cfg.get('enabled', False):
		callbacks.append(
			VisualizationDumpCallback(
				output_dir=vis_cfg.get('output_dir', 'outputs/visualize_runtime_payloads'),
				enable_train=vis_cfg.get('enable_train', False),
				enable_val=vis_cfg.get('enable_val', True),
				enable_test=vis_cfg.get('enable_test', True),
				save_results=vis_cfg.get('save_results', True),
				max_samples_per_stage=vis_cfg.get('max_samples_per_stage', 1),
				every_n_epochs=vis_cfg.get('every_n_epochs', 1),
			)
		)
	return callbacks


def build_logger(cfg: dict[str, Any]):
	logger_cfg = cfg.get('logger', None)
	if logger_cfg is False:
		return False
	if isinstance(logger_cfg, dict):
		return CSVLogger(
			save_dir=logger_cfg.get('save_dir', 'outputs'),
			name=logger_cfg.get('name', 'bevfusion'),
			version=logger_cfg.get('version'),
		)
	return CSVLogger(save_dir='outputs', name='bevfusion')


def build_trainer(cfg: dict[str, Any]) -> L.Trainer:
	trainer_cfg = dict(cfg.get('trainer', {}) or {})
	accelerator = trainer_cfg.pop('accelerator', 'gpu' if torch.cuda.is_available() else 'cpu')
	devices = trainer_cfg.pop('devices', 1)
	strategy = trainer_cfg.pop('strategy', 'auto')
	precision = trainer_cfg.pop('precision', '32-true')
	max_epochs = trainer_cfg.pop('max_epochs', cfg.get('max_epochs', 1))
	default_root_dir = trainer_cfg.pop('default_root_dir', cfg.get('default_root_dir', 'outputs'))
	if 'check_val_every_n_epoch' not in trainer_cfg:
		trainer_cfg['check_val_every_n_epoch'] = _cfg_get(cfg, 'evaluation.interval', 1)

	grad_clip_cfg = dict(cfg.get('optimizer_config', {}).get('grad_clip', {}) or {})
	if 'gradient_clip_val' not in trainer_cfg and 'max_norm' in grad_clip_cfg:
		trainer_cfg['gradient_clip_val'] = float(grad_clip_cfg['max_norm'])
	if 'gradient_clip_algorithm' not in trainer_cfg and 'norm_type' in grad_clip_cfg:
		trainer_cfg['gradient_clip_algorithm'] = 'norm'

	callbacks = build_callbacks(cfg)
	logger = build_logger(cfg)
	return L.Trainer(
		accelerator=accelerator,
		devices=devices,
		strategy=strategy,
		precision=precision,
		max_epochs=max_epochs,
		default_root_dir=default_root_dir,
		callbacks=callbacks,
		logger=logger,
		**trainer_cfg,
	)


@hydra.main(version_base='1.3', config_path='configs', config_name='config')
def main(cfg: DictConfig) -> None:
	plain_cfg = _to_plain_config(cfg)

	seed = plain_cfg.get('seed', None)
	if seed is not None:
		L.seed_everything(int(seed), workers=True)

	hydra_cfg = HydraConfig.get()
	plain_cfg.setdefault('hydra', _to_plain_config(hydra_cfg, resolve=False))

	trainer = build_trainer(plain_cfg)
	datamodule = LitDataModule(plain_cfg)
	module = BEVFusionLitModule(plain_cfg)

	ckpt_path = plain_cfg.get('ckpt_path', None)
	trainer.fit(module, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == '__main__':
	main()
