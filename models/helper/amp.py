import functools
import inspect

import torch


def _find_tensor(obj):
	if isinstance(obj, torch.Tensor):
		return obj
	if isinstance(obj, (list, tuple)):
		for item in obj:
			tensor = _find_tensor(item)
			if tensor is not None:
				return tensor
	if isinstance(obj, dict):
		for item in obj.values():
			tensor = _find_tensor(item)
			if tensor is not None:
				return tensor
	return None


def _cast_fp32(obj):
	if isinstance(obj, torch.Tensor) and torch.is_floating_point(obj):
		return obj.float()
	if isinstance(obj, tuple):
		return tuple(_cast_fp32(item) for item in obj)
	if isinstance(obj, list):
		return [_cast_fp32(item) for item in obj]
	if isinstance(obj, dict):
		return {key: _cast_fp32(value) for key, value in obj.items()}
	return obj


def force_fp32(apply_to=None):
	# Minimal replacement for mmcv.runner.force_fp32 used by geometry / BEV ops.
	# It casts selected floating-point inputs to fp32 and runs the function with
	# autocast disabled so numerically sensitive code stays in full precision.
	def decorator(func):
		signature = inspect.signature(func)

		@functools.wraps(func)
		def wrapper(*args, **kwargs):
			bound = signature.bind(*args, **kwargs)
			bound.apply_defaults()

			target_names = apply_to
			if target_names is None:
				target_names = tuple(name for name in bound.arguments.keys() if name != 'self')

			for name in target_names:
				if name in bound.arguments:
					bound.arguments[name] = _cast_fp32(bound.arguments[name])

			tensor = _find_tensor(tuple(bound.arguments.values()))
			device_type = (
				tensor.device.type
				if tensor is not None
				else ('cuda' if torch.cuda.is_available() else 'cpu')
			)

			with torch.amp.autocast(device_type=device_type, enabled=False):
				return func(*bound.args, **bound.kwargs)

		return wrapper

	return decorator
