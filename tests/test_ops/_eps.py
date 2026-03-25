from __future__ import annotations

import torch

DTYPE_EPS = {
	torch.float32: (1e-4, 1e-5),
	torch.float16: (6e-2, 2e-2),
	torch.bfloat16: (2.0, 4e-1),
}


def dtype_tiny(dtype: torch.dtype) -> float:
	return float(torch.finfo(dtype).tiny)
