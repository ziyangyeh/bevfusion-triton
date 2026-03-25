"""
3-D bounding box utilities.

Placeholder functions whose names match the original repo:
  - encode_bbox  (BBoxCoder.encode – mmdet3d/core/bbox/coders/)
  - decode_bbox  (BBoxCoder.decode)
  - xywhr2xyxyr  (mmdet3d/core/utils/array_converter.py)

TODO: implement box encoding/decoding, IoU, NMS.
"""

from __future__ import annotations

import torch


def encode_bbox(boxes: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
	"""Encode ground-truth boxes relative to anchors.

	Placeholder: BBoxCoder.encode from mmdet3d/core/bbox/coders/
	"""
	raise NotImplementedError('encode_bbox not yet implemented.')


def decode_bbox(preds: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
	"""Decode predicted deltas back to absolute 3-D boxes.

	Placeholder: BBoxCoder.decode from mmdet3d/core/bbox/coders/
	"""
	raise NotImplementedError('decode_bbox not yet implemented.')


def xywhr2xyxyr(boxes: torch.Tensor) -> torch.Tensor:
	"""Convert (cx, cy, w, h, r) to (x1, y1, x2, y2, r).

	Mirrors: mmdet3d/core/utils/array_converter.py – xywhr2xyxyr
	"""
	x1 = boxes[:, 0] - boxes[:, 2] / 2
	y1 = boxes[:, 1] - boxes[:, 3] / 2
	x2 = boxes[:, 0] + boxes[:, 2] / 2
	y2 = boxes[:, 1] + boxes[:, 3] / 2
	return torch.stack([x1, y1, x2, y2, boxes[:, 4]], dim=-1)
