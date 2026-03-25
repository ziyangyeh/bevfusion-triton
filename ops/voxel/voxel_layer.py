from __future__ import annotations

import importlib

try:
	from . import src as voxel_triton
except Exception:  # pragma: no cover
	voxel_triton = None

# Keep the historical extension available as a pytest reference backend.
# Runtime voxelization should use the local Triton implementation, which
# already includes its own CPU fallback path.
_legacy_module = None


def _get_legacy_module():
	global _legacy_module
	if _legacy_module is None:
		_legacy_module = importlib.import_module('.voxel_layer_legacy', package=__package__)
	return _legacy_module


# Expose the legacy module for parity tests against the Triton/CPU-fallback path.
voxel_layer_ext = _get_legacy_module()


def hard_voxelize(*args, **kwargs):
	if voxel_triton is not None:
		return voxel_triton.hard_voxelize(*args, **kwargs)
	return voxel_layer_ext.hard_voxelize(*args, **kwargs)


def dynamic_voxelize(*args, **kwargs):
	if voxel_triton is not None:
		return voxel_triton.dynamic_voxelize(*args, **kwargs)
	return voxel_layer_ext.dynamic_voxelize(*args, **kwargs)


def dynamic_point_to_voxel_forward(*args, **kwargs):
	if voxel_triton is not None:
		return voxel_triton.dynamic_point_to_voxel_forward(*args, **kwargs)
	return voxel_layer_ext.dynamic_point_to_voxel_forward(*args, **kwargs)


def dynamic_point_to_voxel_backward(*args, **kwargs):
	if voxel_triton is not None:
		return voxel_triton.dynamic_point_to_voxel_backward(*args, **kwargs)
	return voxel_layer_ext.dynamic_point_to_voxel_backward(*args, **kwargs)
