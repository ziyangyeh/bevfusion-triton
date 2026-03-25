from __future__ import annotations

from pathlib import Path

from torch.utils.cpp_extension import load

from utils.lazy_inline_extension import _DEFAULT_CFLAGS, _default_nvcc_flags

_CSRC_DIR = Path(__file__).resolve().parent / 'csrc'
_EXPORTED_NAMES = [
	'hard_voxelize',
	'dynamic_voxelize',
	'dynamic_point_to_voxel_forward',
	'dynamic_point_to_voxel_backward',
]


def _source_paths() -> list[str]:
	return [
		str(_CSRC_DIR / 'voxelization.cpp'),
		str(_CSRC_DIR / 'voxelization_cpu.cpp'),
		str(_CSRC_DIR / 'scatter_points_cpu.cpp'),
		str(_CSRC_DIR / 'voxelization_cuda.cu'),
		str(_CSRC_DIR / 'scatter_points_cuda.cu'),
	]


def _load_module():
	extra_cflags = [*_DEFAULT_CFLAGS, '-DWITH_CUDA']
	extra_cuda_cflags = [*_default_nvcc_flags(), '-DWITH_CUDA']
	return load(
		name='voxel_layer_ext',
		sources=_source_paths(),
		extra_include_paths=[str(_CSRC_DIR)],
		extra_cflags=extra_cflags,
		extra_cuda_cflags=extra_cuda_cflags,
		with_cuda=True,
		verbose=False,
	)


_module = None


def _get_module():
	global _module
	if _module is None:
		_module = _load_module()
	return _module


class _VoxelLayerExtProxy:
	def __getattr__(self, name: str):
		return getattr(_get_module(), name)

	def __dir__(self):
		return sorted(set(type(self).__dict__.keys()) | set(_EXPORTED_NAMES))


voxel_layer_ext = _VoxelLayerExtProxy()


def hard_voxelize(*args, **kwargs):
	return voxel_layer_ext.hard_voxelize(*args, **kwargs)


def dynamic_voxelize(*args, **kwargs):
	return voxel_layer_ext.dynamic_voxelize(*args, **kwargs)


def dynamic_point_to_voxel_forward(*args, **kwargs):
	return voxel_layer_ext.dynamic_point_to_voxel_forward(*args, **kwargs)


def dynamic_point_to_voxel_backward(*args, **kwargs):
	return voxel_layer_ext.dynamic_point_to_voxel_backward(*args, **kwargs)


def __dir__():
	return sorted(set(globals().keys()) | set(_EXPORTED_NAMES))


__all__ = [
	'dynamic_point_to_voxel_backward',
	'dynamic_point_to_voxel_forward',
	'dynamic_voxelize',
	'hard_voxelize',
	'voxel_layer_ext',
]
