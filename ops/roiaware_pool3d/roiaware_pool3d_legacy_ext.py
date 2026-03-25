from __future__ import annotations

from pathlib import Path

from torch.utils.cpp_extension import load

from utils.lazy_inline_extension import _DEFAULT_CFLAGS, _default_nvcc_flags

_CSRC_DIR = Path(__file__).resolve().parent / 'csrc'
_EXPORTED_NAMES = [
	'forward',
	'backward',
	'points_in_boxes_gpu',
	'points_in_boxes_batch',
	'points_in_boxes_cpu',
]


def _source_paths() -> list[str]:
	return [
		str(_CSRC_DIR / 'roiaware_pool3d.cpp'),
		str(_CSRC_DIR / 'points_in_boxes_cpu.cpp'),
		str(_CSRC_DIR / 'roiaware_pool3d_kernel.cu'),
		str(_CSRC_DIR / 'points_in_boxes_cuda.cu'),
	]


def _load_module():
	return load(
		name='roiaware_pool3d_ext',
		sources=_source_paths(),
		extra_include_paths=[str(_CSRC_DIR)],
		extra_cflags=list(_DEFAULT_CFLAGS),
		extra_cuda_cflags=_default_nvcc_flags(),
		with_cuda=True,
		verbose=False,
	)


_module = None


def __getattr__(name: str):
	global _module
	if _module is None:
		_module = _load_module()
	return getattr(_module, name)


def __dir__():
	return sorted(set(globals().keys()) | set(_EXPORTED_NAMES))
