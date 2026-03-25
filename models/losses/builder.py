from .focal_loss import FocalLoss
from .gaussian_focal_loss import GaussianFocalLoss
from .l1_loss import L1Loss
from .smooth_l1_loss import SmoothL1Loss
from .varifocal_loss import VarifocalLoss


LOSS_REGISTRY = {
	'FocalLoss': FocalLoss,
	'GaussianFocalLoss': GaussianFocalLoss,
	'L1Loss': L1Loss,
	'SmoothL1Loss': SmoothL1Loss,
	'VarifocalLoss': VarifocalLoss,
}


def build_loss(cfg: dict):
	if not isinstance(cfg, dict):
		raise TypeError('cfg must be a dict')
	if 'type' not in cfg:
		raise KeyError('the cfg dict must contain the key "type"')
	cfg = dict(cfg)
	loss_type = cfg.pop('type')
	if loss_type not in LOSS_REGISTRY:
		raise KeyError(f'Unsupported loss type: {loss_type}')
	return LOSS_REGISTRY[loss_type](**cfg)
