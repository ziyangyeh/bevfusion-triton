from __future__ import annotations

from typing import Any

from omegaconf import OmegaConf


def register_omegaconf_resolvers() -> None:
	def _slice(value: Any, start: int | str | None = None, end: int | str | None = None):
		if value is None:
			return value
		sequence = list(value)
		start_idx = None if start in (None, '', 'null', 'None') else int(start)
		end_idx = None if end in (None, '', 'null', 'None') else int(end)
		return sequence[start_idx:end_idx]

	OmegaConf.register_new_resolver('slice', _slice, replace=True)
