from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Iterable, Sequence

from torch.utils.cpp_extension import load_inline

__all__ = ['load_inline_from_filenames', 'LazyInlineExtension']

_DEFAULT_CFLAGS = ['-w', '-std=c++17']
_DEFAULT_NVCC_FLAGS = [
	'-D__CUDA_NO_HALF_OPERATORS__',
	'-D__CUDA_NO_HALF_CONVERSIONS__',
	'-D__CUDA_NO_HALF2_OPERATORS__',
	'-gencode=arch=compute_70,code=sm_70',
	'-gencode=arch=compute_75,code=sm_75',
	'-gencode=arch=compute_80,code=sm_80',
	'-gencode=arch=compute_86,code=sm_86',
	'-gencode=arch=compute_120,code=sm_120',
]


def _read_sources(base: Path, filenames: Sequence[str], kind: str) -> list[str]:
	sources: list[str] = []
	for filename in filenames:
		path = base / filename
		if not path.exists():
			raise FileNotFoundError(f'Missing {kind} source file: {path}')
		sources.append(path.read_text(encoding='utf-8'))
	return sources


def _default_nvcc_flags() -> list[str]:
	try:
		result = subprocess.run(
			['nvcc', '--list-gpu-code'],
			check=True,
			capture_output=True,
			text=True,
		)
		supported = {
			line.strip() for line in result.stdout.splitlines() if line.strip().startswith('sm_')
		}
	except Exception:
		supported = None

	flags = list(_DEFAULT_CFLAGS)
	for flag in _DEFAULT_NVCC_FLAGS:
		if not flag.startswith('-gencode=') or supported is None:
			flags.append(flag)
			continue
		code = flag.split('code=', 1)[1]
		if code in supported:
			flags.append(flag)
	return flags


def load_inline_from_filenames(
	*,
	name: str,
	source_dir: str | Path,
	cpp_filenames: Sequence[str] = (),
	cuda_filenames: Sequence[str] = (),
	functions: Sequence[str] | dict[str, str] | None = None,
	extra_include_paths: Iterable[str | Path] = (),
	with_cuda: bool = False,
	extra_cflags: Sequence[str] | None = None,
	extra_cuda_cflags: Sequence[str] | None = None,
	verbose: bool = False,
):
	base = Path(source_dir)
	cflags = list(dict.fromkeys([*_DEFAULT_CFLAGS, *(extra_cflags or ())]))
	cuda_cflags = list(dict.fromkeys([*_default_nvcc_flags(), *(extra_cuda_cflags or ())]))

	return load_inline(
		name=name,
		cpp_sources=_read_sources(base, cpp_filenames, 'C++'),
		cuda_sources=_read_sources(base, cuda_filenames, 'CUDA'),
		functions=functions,
		extra_include_paths=[str(path) for path in extra_include_paths],
		with_cuda=with_cuda,
		extra_cflags=cflags,
		extra_cuda_cflags=cuda_cflags,
		verbose=verbose,
	)


class LazyInlineExtension:
	def __init__(self, *, exported_names: Iterable[str] = (), **inline_kwargs):
		self._exported_names = tuple(exported_names)
		self._inline_kwargs = inline_kwargs
		self._module = None

	def _get_module(self):
		if self._module is None:
			self._module = load_inline_from_filenames(**self._inline_kwargs)
		return self._module

	def __getattr__(self, name: str):
		return getattr(self._get_module(), name)

	def __dir__(self):
		return sorted(set(type(self).__dict__.keys()) | set(self._exported_names))
