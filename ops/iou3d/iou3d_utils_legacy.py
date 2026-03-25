from pathlib import Path

import torch

from utils.lazy_inline_extension import LazyInlineExtension

_CSRC_DIR = Path(__file__).resolve().parent / 'csrc'
_EXT_EXPORTED_FUNCTIONS = ['boxes_iou_bev_gpu', 'nms_gpu', 'nms_normal_gpu']

ext_module = LazyInlineExtension(
	exported_names=_EXT_EXPORTED_FUNCTIONS,
	name='iou3d_utils_ext',
	source_dir=_CSRC_DIR,
	cpp_filenames=['iou3d.cpp'],
	cuda_filenames=['iou3d_kernel.cu'],
	extra_include_paths=[_CSRC_DIR],
	with_cuda=True,
	verbose=False,
)


def boxes_iou_bev(boxes_a, boxes_b):
	"""Calculate boxes IoU in the bird view.

	Args:
	    boxes_a (torch.Tensor): Input boxes a with shape (M, 5).
	    boxes_b (torch.Tensor): Input boxes b with shape (N, 5).

	Returns:
	    ans_iou (torch.Tensor): IoU result with shape (M, N).
	"""
	ans_iou = boxes_a.new_zeros(torch.Size((boxes_a.shape[0], boxes_b.shape[0])))

	ext_module.boxes_iou_bev_gpu(boxes_a.contiguous(), boxes_b.contiguous(), ans_iou)

	return ans_iou


def nms_gpu(boxes, scores, thresh, pre_maxsize=None, post_max_size=None):
	"""Nms function with gpu implementation.

	Args:
	    boxes (torch.Tensor): Input boxes with the shape of [N, 5]
	        ([x1, y1, x2, y2, ry]).
	    scores (torch.Tensor): Scores of boxes with the shape of [N].
	    thresh (int): Threshold.
	    pre_maxsize (int): Max size of boxes before nms. Default: None.
	    post_maxsize (int): Max size of boxes after nms. Default: None.

	Returns:
	    torch.Tensor: Indexes after nms.
	"""
	order = scores.sort(0, descending=True)[1]

	if pre_maxsize is not None:
		order = order[:pre_maxsize]
	boxes = boxes[order].contiguous()

	keep = torch.zeros(boxes.size(0), dtype=torch.long)
	num_out = ext_module.nms_gpu(boxes, keep, thresh, boxes.device.index)
	keep = order[keep[:num_out].cuda(boxes.device)].contiguous()
	if post_max_size is not None:
		keep = keep[:post_max_size]
	return keep


def nms_normal_gpu(boxes, scores, thresh):
	"""Normal non maximum suppression on GPU.

	Args:
	    boxes (torch.Tensor): Input boxes with shape (N, 5).
	    scores (torch.Tensor): Scores of predicted boxes with shape (N).
	    thresh (torch.Tensor): Threshold of non maximum suppression.

	Returns:
	    torch.Tensor: Remaining indices with scores in descending order.
	"""
	order = scores.sort(0, descending=True)[1]

	boxes = boxes[order].contiguous()

	keep = torch.zeros(boxes.size(0), dtype=torch.long)
	num_out = ext_module.nms_normal_gpu(boxes, keep, thresh, boxes.device.index)
	return order[keep[:num_out].cuda(boxes.device)].contiguous()
