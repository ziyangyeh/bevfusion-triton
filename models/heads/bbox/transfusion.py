from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from core.bbox.structures.lidar_box3d import LiDARInstance3DBoxes
from models.helper import ConvModule, build_conv_layer
from models.losses.focal_loss import FocalLoss
from models.losses.gaussian_focal_loss import GaussianFocalLoss
from models.losses.l1_loss import L1Loss
from models.losses.varifocal_loss import VarifocalLoss

__all__ = ['TransFusionBBoxCoder', 'TransFusionHead']

try:
	from scipy.optimize import linear_sum_assignment
except Exception:
	linear_sum_assignment = None


def _resolve_bias(bias: bool | str, with_norm: bool = False) -> bool:
	if bias == 'auto':
		return not with_norm
	return bool(bias)


class PositionEmbeddingLearned(nn.Module):
	"""Learned absolute position embedding for BEV query/key coordinates."""

	def __init__(self, input_channel: int, num_pos_feats: int) -> None:
		super().__init__()
		self.position_embedding_head = nn.Sequential(
			nn.Conv1d(input_channel, num_pos_feats, kernel_size=1),
			nn.BatchNorm1d(num_pos_feats),
			nn.ReLU(inplace=True),
			nn.Conv1d(num_pos_feats, num_pos_feats, kernel_size=1),
		)

	def forward(self, xyz: torch.Tensor) -> torch.Tensor:
		return self.position_embedding_head(xyz.transpose(1, 2).contiguous())


class TransformerDecoderLayer(nn.Module):
	"""Minimal decoder layer used by TransFusion query refinement."""

	def __init__(
		self,
		d_model: int,
		nhead: int,
		dim_feedforward: int = 2048,
		dropout: float = 0.1,
		activation: str = 'relu',
		self_posembed: nn.Module | None = None,
		cross_posembed: nn.Module | None = None,
		cross_only: bool = False,
	) -> None:
		super().__init__()
		self.cross_only = cross_only
		if not self.cross_only:
			self.self_attn = nn.MultiheadAttention(
				d_model,
				nhead,
				dropout=dropout,
				batch_first=True,
			)
		self.multihead_attn = nn.MultiheadAttention(
			d_model,
			nhead,
			dropout=dropout,
			batch_first=True,
		)
		self.linear1 = nn.Linear(d_model, dim_feedforward)
		self.linear2 = nn.Linear(dim_feedforward, d_model)
		self.norm1 = nn.LayerNorm(d_model)
		self.norm2 = nn.LayerNorm(d_model)
		self.norm3 = nn.LayerNorm(d_model)
		self.dropout = nn.Dropout(dropout)
		self.dropout1 = nn.Dropout(dropout)
		self.dropout2 = nn.Dropout(dropout)
		self.dropout3 = nn.Dropout(dropout)
		self.self_posembed = self_posembed
		self.cross_posembed = cross_posembed

		if activation == 'relu':
			self.activation = F.relu
		elif activation == 'gelu':
			self.activation = F.gelu
		else:
			raise ValueError(f'Unsupported activation: {activation}')

	def forward(
		self,
		query: torch.Tensor,
		key: torch.Tensor,
		query_pos: torch.Tensor,
		key_pos: torch.Tensor,
		attn_mask: torch.Tensor | None = None,
	) -> torch.Tensor:
		query_seq = query.transpose(1, 2).contiguous()
		key_seq = key.transpose(1, 2).contiguous()

		query_pos_embed = None
		if self.self_posembed is not None:
			query_pos_embed = self.self_posembed(query_pos).transpose(1, 2).contiguous()

		key_pos_embed = None
		if self.cross_posembed is not None:
			key_pos_embed = self.cross_posembed(key_pos).transpose(1, 2).contiguous()

		if not self.cross_only:
			self_q = query_seq if query_pos_embed is None else query_seq + query_pos_embed
			query2, _ = self.self_attn(self_q, self_q, self_q, need_weights=False)
			query_seq = self.norm1(query_seq + self.dropout1(query2))

		cross_q = query_seq if query_pos_embed is None else query_seq + query_pos_embed
		cross_k = key_seq if key_pos_embed is None else key_seq + key_pos_embed
		query2, _ = self.multihead_attn(
			cross_q,
			cross_k,
			cross_k,
			attn_mask=attn_mask,
			need_weights=False,
		)
		query_seq = self.norm2(query_seq + self.dropout2(query2))

		query2 = self.linear2(self.dropout(self.activation(self.linear1(query_seq))))
		query_seq = self.norm3(query_seq + self.dropout3(query2))
		return query_seq.transpose(1, 2).contiguous()


class FFN(nn.Module):
	"""Head-specific Conv1d branches following the original TransFusion layout."""

	def __init__(
		self,
		in_channels: int,
		heads: dict[str, tuple[int, int]],
		head_conv: int = 64,
		final_kernel: int = 1,
		init_bias: float = -2.19,
		conv_cfg: dict[str, Any] | None = None,
		norm_cfg: dict[str, Any] | None = None,
		bias: bool | str = 'auto',
	) -> None:
		super().__init__()
		self.heads = heads
		self.init_bias = init_bias
		for name, (out_channels, num_convs) in heads.items():
			layers: list[nn.Module] = []
			c_in = in_channels
			for _ in range(max(num_convs - 1, 0)):
				layers.append(
					ConvModule(
						c_in,
						head_conv,
						kernel_size=final_kernel,
						stride=1,
						padding=final_kernel // 2,
						bias=bias,
						conv_cfg=conv_cfg,
						norm_cfg=norm_cfg,
					)
				)
				c_in = head_conv
			layers.append(
				build_conv_layer(
					conv_cfg,
					c_in,
					out_channels,
					kernel_size=final_kernel,
					stride=1,
					padding=final_kernel // 2,
					bias=True,
				)
			)
			self.__setattr__(name, nn.Sequential(*layers))

	def init_weights(self) -> None:
		for name in self.heads:
			if name == 'heatmap':
				self.__getattr__(name)[-1].bias.data.fill_(self.init_bias)

	def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
		return {name: self.__getattr__(name)(x) for name in self.heads}


class TransFusionBBoxCoder:
	"""Local bbox coder used by the Lightning-friendly TransFusion head."""

	def __init__(
		self,
		pc_range: tuple[float, float] | list[float] = (0.0, 0.0),
		out_size_factor: int = 8,
		voxel_size: tuple[float, float] | list[float] = (0.1, 0.1),
		post_center_range: tuple[float, ...] | list[float] | None = None,
		score_threshold: float | None = None,
		code_size: int = 10,
	) -> None:
		self.pc_range = tuple(pc_range)
		self.out_size_factor = int(out_size_factor)
		self.voxel_size = tuple(voxel_size)
		self.post_center_range = post_center_range
		self.score_threshold = score_threshold
		self.code_size = int(code_size)

	def encode(self, dst_boxes: torch.Tensor) -> torch.Tensor:
		targets = torch.zeros((dst_boxes.shape[0], self.code_size), device=dst_boxes.device)
		targets[:, 0] = (dst_boxes[:, 0] - self.pc_range[0]) / (
			self.out_size_factor * self.voxel_size[0]
		)
		targets[:, 1] = (dst_boxes[:, 1] - self.pc_range[1]) / (
			self.out_size_factor * self.voxel_size[1]
		)
		targets[:, 2] = dst_boxes[:, 2] + dst_boxes[:, 5] * 0.5
		targets[:, 3:6] = dst_boxes[:, 3:6].log()
		targets[:, 6] = torch.sin(dst_boxes[:, 6])
		targets[:, 7] = torch.cos(dst_boxes[:, 6])
		if self.code_size == 10 and dst_boxes.shape[1] >= 9:
			targets[:, 8:10] = dst_boxes[:, 7:9]
		return targets

	def decode(
		self,
		heatmap: torch.Tensor,
		rot: torch.Tensor,
		dim: torch.Tensor,
		center: torch.Tensor,
		height: torch.Tensor,
		vel: torch.Tensor | None,
		filter: bool = False,
	) -> list[dict[str, torch.Tensor]]:
		final_preds = heatmap.max(dim=1).indices
		final_scores = heatmap.max(dim=1).values

		center = center.clone()
		dim = dim.clone()
		height = height.clone()
		center[:, 0, :] = (
			center[:, 0, :] * self.out_size_factor * self.voxel_size[0] + self.pc_range[0]
		)
		center[:, 1, :] = (
			center[:, 1, :] * self.out_size_factor * self.voxel_size[1] + self.pc_range[1]
		)
		dim = dim.exp()
		height = height - dim[:, 2:3, :] * 0.5
		rot = torch.atan2(rot[:, 0:1, :], rot[:, 1:2, :])

		if vel is None:
			final_box_preds = torch.cat([center, height, dim, rot], dim=1).permute(0, 2, 1)
		else:
			final_box_preds = torch.cat([center, height, dim, rot, vel], dim=1).permute(0, 2, 1)

		predictions = []
		for batch_idx in range(heatmap.shape[0]):
			boxes = final_box_preds[batch_idx]
			scores = final_scores[batch_idx]
			labels = final_preds[batch_idx]

			if filter:
				keep = torch.ones_like(scores, dtype=torch.bool)
				if self.score_threshold is not None:
					keep &= scores > self.score_threshold
				if self.post_center_range is not None:
					center_range = boxes.new_tensor(self.post_center_range)
					keep &= (boxes[:, :3] >= center_range[:3]).all(dim=1)
					keep &= (boxes[:, :3] <= center_range[3:]).all(dim=1)
				boxes = boxes[keep]
				scores = scores[keep]
				labels = labels[keep]

			predictions.append({'bboxes': boxes, 'scores': scores, 'labels': labels})
		return predictions


class AssignResult:
	def __init__(
		self,
		num_gts: int,
		gt_inds: torch.Tensor,
		max_overlaps: torch.Tensor | None,
		labels: torch.Tensor | None = None,
	) -> None:
		self.num_gts = num_gts
		self.gt_inds = gt_inds
		self.max_overlaps = max_overlaps
		self.labels = labels


class SamplingResult:
	def __init__(
		self,
		pos_inds: torch.Tensor,
		neg_inds: torch.Tensor,
		bboxes: torch.Tensor,
		gt_bboxes: torch.Tensor,
		assign_result: AssignResult,
		gt_flags: torch.Tensor,
	) -> None:
		self.pos_inds = pos_inds
		self.neg_inds = neg_inds
		self.pos_bboxes = bboxes[pos_inds]
		self.neg_bboxes = bboxes[neg_inds]
		self.pos_is_gt = gt_flags[pos_inds]
		self.num_gts = gt_bboxes.shape[0]
		self.pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1
		if gt_bboxes.numel() == 0:
			box_dim = gt_bboxes.shape[-1] if gt_bboxes.ndim > 1 else 0
			self.pos_gt_bboxes = torch.empty(
				(0, box_dim), device=gt_bboxes.device, dtype=gt_bboxes.dtype
			)
		else:
			self.pos_gt_bboxes = gt_bboxes[self.pos_assigned_gt_inds, :]
		self.pos_gt_labels = (
			assign_result.labels[pos_inds] if assign_result.labels is not None else None
		)


class PseudoSampler:
	def sample(
		self,
		assign_result: AssignResult,
		bboxes: torch.Tensor,
		gt_bboxes: torch.Tensor,
	) -> SamplingResult:
		pos_inds = torch.nonzero(assign_result.gt_inds > 0, as_tuple=False).squeeze(-1).unique()
		neg_inds = torch.nonzero(assign_result.gt_inds == 0, as_tuple=False).squeeze(-1).unique()
		gt_flags = bboxes.new_zeros(bboxes.shape[0], dtype=torch.uint8)
		return SamplingResult(pos_inds, neg_inds, bboxes, gt_bboxes, assign_result, gt_flags)


def clip_sigmoid(x: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
	return x.sigmoid().clamp(min=eps, max=1 - eps)


def gaussian_2d(shape: tuple[int, int], sigma: float = 1.0) -> np.ndarray:
	m, n = [(ss - 1.0) / 2.0 for ss in shape]
	y, x = np.ogrid[-m : m + 1, -n : n + 1]
	h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
	h[h < np.finfo(h.dtype).eps * h.max()] = 0
	return h


def draw_heatmap_gaussian(
	heatmap: torch.Tensor,
	center: torch.Tensor,
	radius: int,
	k: float = 1.0,
) -> torch.Tensor:
	diameter = 2 * radius + 1
	gaussian = gaussian_2d((diameter, diameter), sigma=diameter / 6)

	x, y = int(center[0]), int(center[1])
	height, width = heatmap.shape[:2]

	left, right = min(x, radius), min(width - x, radius + 1)
	top, bottom = min(y, radius), min(height - y, radius + 1)

	masked_heatmap = heatmap[y - top : y + bottom, x - left : x + right]
	masked_gaussian = torch.from_numpy(
		gaussian[radius - top : radius + bottom, radius - left : radius + right]
	).to(heatmap.device, torch.float32)

	if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
		torch.max(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
	return heatmap


def gaussian_radius(
	det_size: tuple[torch.Tensor, torch.Tensor], min_overlap: float = 0.5
) -> torch.Tensor:
	height, width = det_size

	a1 = 1
	b1 = height + width
	c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
	sq1 = torch.sqrt(b1**2 - 4 * a1 * c1)
	r1 = (b1 + sq1) / 2

	a2 = 4
	b2 = 2 * (height + width)
	c2 = (1 - min_overlap) * width * height
	sq2 = torch.sqrt(b2**2 - 4 * a2 * c2)
	r2 = (b2 + sq2) / 2

	a3 = 4 * min_overlap
	b3 = -2 * min_overlap * (height + width)
	c3 = (min_overlap - 1) * width * height
	sq3 = torch.sqrt(b3**2 - 4 * a3 * c3)
	r3 = (b3 + sq3) / 2
	return min(r1, r2, r3)


def focal_loss_cost(
	cls_pred: torch.Tensor,
	gt_labels: torch.Tensor,
	alpha: float = 0.25,
	gamma: float = 2.0,
	weight: float = 1.0,
	eps: float = 1e-12,
) -> torch.Tensor:
	cls_pred = cls_pred.sigmoid()
	neg_cost = -(1 - cls_pred + eps).log() * (1 - alpha) * cls_pred.pow(gamma)
	pos_cost = -(cls_pred + eps).log() * alpha * (1 - cls_pred).pow(gamma)
	return (pos_cost[:, gt_labels] - neg_cost[:, gt_labels]) * weight


def bbox_bev_l1_cost(
	bboxes: torch.Tensor,
	gt_bboxes: torch.Tensor,
	train_cfg: dict[str, Any],
	weight: float = 1.0,
) -> torch.Tensor:
	pc_start = bboxes.new_tensor(train_cfg['point_cloud_range'][0:2])
	pc_end = bboxes.new_tensor(train_cfg['point_cloud_range'][3:5])
	pc_range = pc_end - pc_start
	normalized_bboxes_xy = (bboxes[:, :2] - pc_start) / pc_range
	normalized_gt_bboxes_xy = (gt_bboxes[:, :2] - pc_start) / pc_range
	return torch.cdist(normalized_bboxes_xy, normalized_gt_bboxes_xy, p=1) * weight


def axis_aligned_bev_iou(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
	if boxes_a.numel() == 0 or boxes_b.numel() == 0:
		return boxes_a.new_zeros((boxes_a.shape[0], boxes_b.shape[0]))

	xa1 = boxes_a[:, 0] - boxes_a[:, 3] * 0.5
	ya1 = boxes_a[:, 1] - boxes_a[:, 4] * 0.5
	xa2 = boxes_a[:, 0] + boxes_a[:, 3] * 0.5
	ya2 = boxes_a[:, 1] + boxes_a[:, 4] * 0.5

	xb1 = boxes_b[:, 0] - boxes_b[:, 3] * 0.5
	yb1 = boxes_b[:, 1] - boxes_b[:, 4] * 0.5
	xb2 = boxes_b[:, 0] + boxes_b[:, 3] * 0.5
	yb2 = boxes_b[:, 1] + boxes_b[:, 4] * 0.5

	inter_x1 = torch.maximum(xa1[:, None], xb1[None, :])
	inter_y1 = torch.maximum(ya1[:, None], yb1[None, :])
	inter_x2 = torch.minimum(xa2[:, None], xb2[None, :])
	inter_y2 = torch.minimum(ya2[:, None], yb2[None, :])

	inter_w = (inter_x2 - inter_x1).clamp(min=0)
	inter_h = (inter_y2 - inter_y1).clamp(min=0)
	inter_area = inter_w * inter_h

	area_a = (xa2 - xa1).clamp(min=0) * (ya2 - ya1).clamp(min=0)
	area_b = (xb2 - xb1).clamp(min=0) * (yb2 - yb1).clamp(min=0)
	union = area_a[:, None] + area_b[None, :] - inter_area
	return inter_area / union.clamp(min=1e-6)


def lidar_box_iou3d(pred_boxes: torch.Tensor, gt_boxes: torch.Tensor) -> torch.Tensor:
	if pred_boxes.numel() == 0 or gt_boxes.numel() == 0:
		return pred_boxes.new_zeros((pred_boxes.shape[0], gt_boxes.shape[0]))
	if pred_boxes.is_cuda and gt_boxes.is_cuda:
		pred_box_obj = LiDARInstance3DBoxes(pred_boxes[:, :7], box_dim=7)
		gt_box_obj = LiDARInstance3DBoxes(gt_boxes[:, :7], box_dim=7)
		return LiDARInstance3DBoxes.overlaps(pred_box_obj, gt_box_obj, mode='iou')
	return axis_aligned_bev_iou(pred_boxes, gt_boxes)


class HungarianAssigner3D:
	def __init__(
		self, cls_cost: dict[str, Any], reg_cost: dict[str, Any], iou_cost: dict[str, Any]
	) -> None:
		self.cls_cost_cfg = dict(cls_cost)
		self.reg_cost_cfg = dict(reg_cost)
		self.iou_cost_cfg = dict(iou_cost)

	def assign(
		self,
		bboxes: torch.Tensor,
		gt_bboxes: torch.Tensor,
		gt_labels: torch.Tensor,
		cls_pred: torch.Tensor,
		train_cfg: dict[str, Any],
	) -> AssignResult:
		num_gts, num_bboxes = gt_bboxes.size(0), bboxes.size(0)
		assigned_gt_inds = bboxes.new_full((num_bboxes,), -1, dtype=torch.long)
		assigned_labels = bboxes.new_full((num_bboxes,), -1, dtype=torch.long)
		if num_gts == 0 or num_bboxes == 0:
			if num_gts == 0:
				assigned_gt_inds[:] = 0
			return AssignResult(num_gts, assigned_gt_inds, None, labels=assigned_labels)

		cls_cost = focal_loss_cost(
			cls_pred,
			gt_labels,
			alpha=float(self.cls_cost_cfg.get('alpha', 0.25)),
			gamma=float(self.cls_cost_cfg.get('gamma', 2.0)),
			weight=float(self.cls_cost_cfg.get('weight', 0.15)),
		)
		reg_cost = bbox_bev_l1_cost(
			bboxes,
			gt_bboxes,
			train_cfg,
			weight=float(self.reg_cost_cfg.get('weight', 0.25)),
		)
		iou = lidar_box_iou3d(bboxes, gt_bboxes)
		iou_cost = -iou * float(self.iou_cost_cfg.get('weight', 0.25))
		cost = cls_cost + reg_cost + iou_cost

		if linear_sum_assignment is None:
			raise ImportError('Please install scipy for Hungarian assignment.')
		matched_row_inds, matched_col_inds = linear_sum_assignment(cost.detach().cpu().numpy())
		matched_row_inds = bboxes.new_tensor(matched_row_inds, dtype=torch.long)
		matched_col_inds = bboxes.new_tensor(matched_col_inds, dtype=torch.long)

		assigned_gt_inds[:] = 0
		assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
		assigned_labels[matched_row_inds] = gt_labels[matched_col_inds]

		max_overlaps = torch.zeros_like(iou.max(1).values)
		max_overlaps[matched_row_inds] = iou[matched_row_inds, matched_col_inds]
		return AssignResult(num_gts, assigned_gt_inds, max_overlaps, labels=assigned_labels)


class HeuristicAssigner3D:
	def __init__(self, dist_thre: float = 100.0) -> None:
		self.dist_thre = float(dist_thre)

	def assign(
		self,
		bboxes: torch.Tensor,
		gt_bboxes: torch.Tensor,
		gt_labels: torch.Tensor | None,
		query_labels: torch.Tensor | None = None,
	) -> AssignResult:
		num_gts, num_bboxes = len(gt_bboxes), len(bboxes)
		assigned_gt_inds = bboxes.new_zeros((num_bboxes,), dtype=torch.long)
		assigned_gt_labels = bboxes.new_full((num_bboxes,), -1, dtype=torch.long)
		if num_gts == 0 or num_bboxes == 0:
			return AssignResult(
				num_gts,
				assigned_gt_inds,
				bboxes.new_zeros((num_bboxes,)),
				labels=assigned_gt_labels,
			)

		bev_dist = torch.norm(bboxes[:, 0:2][None, :, :] - gt_bboxes[:, 0:2][:, None, :], dim=-1)
		if query_labels is not None and gt_labels is not None:
			bev_dist = bev_dist + (query_labels[None] != gt_labels[:, None]) * self.dist_thre

		_, nearest_indices = bev_dist.min(1)
		assigned_gt_vals = bboxes.new_full((num_bboxes,), 10000.0)
		for idx_gts in range(num_gts):
			idx_pred = nearest_indices[idx_gts]
			if (
				bev_dist[idx_gts, idx_pred] <= self.dist_thre
				and bev_dist[idx_gts, idx_pred] < assigned_gt_vals[idx_pred]
			):
				assigned_gt_vals[idx_pred] = bev_dist[idx_gts, idx_pred]
				assigned_gt_inds[idx_pred] = idx_gts + 1
				if gt_labels is not None:
					assigned_gt_labels[idx_pred] = gt_labels[idx_gts]

		max_overlaps = bboxes.new_zeros((num_bboxes,))
		matched_indices = torch.where(assigned_gt_inds > 0)[0]
		if matched_indices.numel() > 0:
			matched_gt = assigned_gt_inds[matched_indices] - 1
			overlaps = lidar_box_iou3d(gt_bboxes[matched_gt], bboxes[matched_indices])
			max_overlaps[matched_indices] = overlaps.diag()
		return AssignResult(num_gts, assigned_gt_inds, max_overlaps, labels=assigned_gt_labels)


class TransFusionHead(nn.Module):
	"""Lightning-friendly TransFusion head with original task loss logic."""

	def __init__(
		self,
		num_proposals: int = 200,
		auxiliary: bool = True,
		in_channels: int | None = None,
		hidden_channel: int = 128,
		num_classes: int = 10,
		num_decoder_layers: int = 1,
		num_heads: int = 8,
		nms_kernel_size: int = 3,
		ffn_channel: int = 256,
		dropout: float = 0.1,
		bn_momentum: float = 0.1,
		activation: str = 'relu',
		common_heads: dict[str, tuple[int, int]] | None = None,
		num_heatmap_convs: int = 2,
		conv_cfg: dict[str, Any] | None = None,
		norm_cfg: dict[str, Any] | None = None,
		bias: bool | str = 'auto',
		train_cfg: dict[str, Any] | None = None,
		test_cfg: dict[str, Any] | None = None,
		bbox_coder: dict[str, Any] | None = None,
		loss_cls: dict[str, Any] | None = None,
		loss_heatmap: dict[str, Any] | None = None,
		loss_bbox: dict[str, Any] | None = None,
		loss_iou: dict[str, Any] | None = None,
	) -> None:
		super().__init__()
		self.num_proposals = num_proposals
		self.auxiliary = auxiliary
		self.in_channels = in_channels
		self.hidden_channel = hidden_channel
		self.num_classes = num_classes
		self.num_heads = num_heads
		self.num_decoder_layers = num_decoder_layers
		self.nms_kernel_size = nms_kernel_size
		self.bn_momentum = bn_momentum
		self.train_cfg = train_cfg or {
			'dataset': 'nuScenes',
			'point_cloud_range': (-54.0, -54.0, -5.0, 54.0, 54.0, 3.0),
			'grid_size': (1024, 1024, 1),
			'voxel_size': (0.075, 0.075, 0.2),
			'out_size_factor': 8,
		}
		self.test_cfg = test_cfg or {
			'dataset': 'nuScenes',
			'grid_size': (1024, 1024, 1),
			'out_size_factor': 8,
			'voxel_size': (0.075, 0.075),
			'pc_range': (-54.0, -54.0),
			'nms_type': None,
		}
		self.common_heads = copy.deepcopy(
			common_heads
			or {
				'center': (2, 2),
				'height': (1, 2),
				'dim': (3, 2),
				'rot': (2, 2),
				'vel': (2, 2),
			}
		)
		self.loss_cfg = {
			'loss_cls': loss_cls,
			'loss_heatmap': loss_heatmap,
			'loss_bbox': loss_bbox,
			'loss_iou': loss_iou,
		}

		bbox_coder_cfg = dict(bbox_coder or {})
		if not bbox_coder_cfg:
			bbox_coder_cfg = {
				'pc_range': tuple(self.test_cfg.get('pc_range', (-54.0, -54.0))),
				'post_center_range': (-61.2, -61.2, -10.0, 61.2, 61.2, 10.0),
				'score_threshold': 0.0,
				'out_size_factor': self.test_cfg.get('out_size_factor', 8),
				'voxel_size': tuple(self.test_cfg.get('voxel_size', (0.075, 0.075))),
				'code_size': 10,
			}
		self.bbox_coder = TransFusionBBoxCoder(**bbox_coder_cfg)
		assigner_cfg = dict(self.train_cfg.get('assigner', {}))
		self.bbox_assigner = HungarianAssigner3D(
			cls_cost=assigner_cfg.get('cls_cost', {}),
			reg_cost=assigner_cfg.get('reg_cost', {}),
			iou_cost=assigner_cfg.get('iou_cost', {}),
		)
		self.bbox_sampler = PseudoSampler()
		loss_cls_cfg = dict(loss_cls or {})
		loss_heatmap_cfg = dict(loss_heatmap or {})
		loss_bbox_cfg = dict(loss_bbox or {})
		loss_iou_cfg = dict(loss_iou or {}) if loss_iou is not None else None

		self.loss_cls = FocalLoss(
			**{
				'use_sigmoid': True,
				'gamma': 2.0,
				'alpha': 0.25,
				'reduction': 'mean',
				'loss_weight': 1.0,
				**loss_cls_cfg,
			}
		)
		self.loss_heatmap = GaussianFocalLoss(
			**{
				'reduction': 'mean',
				'loss_weight': 1.0,
				**loss_heatmap_cfg,
			}
		)
		self.loss_bbox = L1Loss(
			**{
				'reduction': 'mean',
				'loss_weight': 0.25,
				**loss_bbox_cfg,
			}
		)
		self.loss_iou = VarifocalLoss(**loss_iou_cfg) if loss_iou_cfg is not None else None

		shared_bias = _resolve_bias(bias, with_norm=False)
		if in_channels is None:
			self.shared_conv = nn.LazyConv2d(
				hidden_channel, kernel_size=3, padding=1, bias=shared_bias
			)
		else:
			self.shared_conv = build_conv_layer(
				{'type': 'Conv2d'},
				in_channels,
				hidden_channel,
				kernel_size=3,
				padding=1,
				bias=shared_bias,
			)

		heatmap_layers: list[nn.Module] = [
			ConvModule(
				hidden_channel,
				hidden_channel,
				kernel_size=3,
				padding=1,
				bias=bias,
				conv_cfg={'type': 'Conv2d'},
				norm_cfg={'type': 'BN2d'},
			)
		]
		for _ in range(max(num_heatmap_convs - 2, 0)):
			heatmap_layers.append(
				ConvModule(
					hidden_channel,
					hidden_channel,
					kernel_size=3,
					padding=1,
					bias=bias,
					conv_cfg={'type': 'Conv2d'},
					norm_cfg={'type': 'BN2d'},
				)
			)
		heatmap_layers.append(
			build_conv_layer(
				{'type': 'Conv2d'},
				hidden_channel,
				num_classes,
				kernel_size=3,
				padding=1,
				bias=_resolve_bias(bias, with_norm=False),
			)
		)
		self.heatmap_head = nn.Sequential(*heatmap_layers)
		self.class_encoding = nn.Conv1d(num_classes, hidden_channel, kernel_size=1)

		decoder_conv_cfg = conv_cfg or {'type': 'Conv1d'}
		decoder_norm_cfg = norm_cfg or {'type': 'BN1d'}
		self.decoder = nn.ModuleList(
			[
				TransformerDecoderLayer(
					hidden_channel,
					num_heads,
					dim_feedforward=ffn_channel,
					dropout=dropout,
					activation=activation,
					self_posembed=PositionEmbeddingLearned(2, hidden_channel),
					cross_posembed=PositionEmbeddingLearned(2, hidden_channel),
				)
				for _ in range(num_decoder_layers)
			]
		)

		self.prediction_heads = nn.ModuleList()
		for _ in range(num_decoder_layers):
			heads = copy.deepcopy(self.common_heads)
			heads['heatmap'] = (self.num_classes, num_heatmap_convs)
			self.prediction_heads.append(
				FFN(
					hidden_channel,
					heads,
					conv_cfg=decoder_conv_cfg,
					norm_cfg=decoder_norm_cfg,
					bias=bias,
				)
			)

		x_size = int(self.test_cfg['grid_size'][0] // self.test_cfg['out_size_factor'])
		y_size = int(self.test_cfg['grid_size'][1] // self.test_cfg['out_size_factor'])
		self.register_buffer('bev_pos', self.create_2d_grid(x_size, y_size), persistent=False)
		self.query_labels: torch.Tensor | None = None
		self.init_weights()

	def create_2d_grid(self, x_size: int, y_size: int) -> torch.Tensor:
		grid_x = torch.linspace(0, x_size - 1, x_size)
		grid_y = torch.linspace(0, y_size - 1, y_size)
		batch_x, batch_y = torch.meshgrid(grid_x, grid_y, indexing='ij')
		coord_base = torch.stack((batch_x + 0.5, batch_y + 0.5), dim=0).unsqueeze(0)
		return coord_base.flatten(2).transpose(1, 2).contiguous()

	def init_weights(self) -> None:
		for module in self.decoder.modules():
			if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
				nn.init.xavier_uniform_(module.weight)
				if module.bias is not None:
					nn.init.constant_(module.bias, 0)
		for prediction_head in self.prediction_heads:
			prediction_head.init_weights()
		self.init_bn_momentum()

	def init_bn_momentum(self) -> None:
		for module in self.modules():
			if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
				module.momentum = self.bn_momentum

	def forward(
		self,
		feats: torch.Tensor | list[torch.Tensor],
		metas: list[dict[str, Any]] | None = None,
	) -> dict[str, torch.Tensor]:
		if isinstance(feats, (list, tuple)):
			if len(feats) != 1:
				raise ValueError('TransFusionHead currently supports exactly one feature level.')
			feats = feats[0]

		batch_size = feats.shape[0]
		lidar_feat = self.shared_conv(feats)
		lidar_feat_flatten = lidar_feat.flatten(2)
		bev_pos = self.bev_pos.to(lidar_feat.device).repeat(batch_size, 1, 1)

		dense_heatmap = self.heatmap_head(lidar_feat)
		heatmap = dense_heatmap.detach().sigmoid()

		padding = self.nms_kernel_size // 2
		local_max = torch.zeros_like(heatmap)
		local_max_inner = F.max_pool2d(
			heatmap,
			kernel_size=self.nms_kernel_size,
			stride=1,
			padding=padding,
		)
		if padding > 0:
			local_max.copy_(local_max_inner)
		else:
			local_max = local_max_inner

		dataset = self.test_cfg.get('dataset')
		if dataset == 'nuScenes' and self.num_classes >= 10:
			local_max[:, 8] = F.max_pool2d(heatmap[:, 8], kernel_size=1, stride=1, padding=0)
			local_max[:, 9] = F.max_pool2d(heatmap[:, 9], kernel_size=1, stride=1, padding=0)
		elif dataset == 'Waymo' and self.num_classes >= 3:
			local_max[:, 1] = F.max_pool2d(heatmap[:, 1], kernel_size=1, stride=1, padding=0)
			local_max[:, 2] = F.max_pool2d(heatmap[:, 2], kernel_size=1, stride=1, padding=0)

		heatmap = heatmap * (heatmap == local_max)
		heatmap = heatmap.flatten(2)

		top_proposals = heatmap.flatten(1).topk(self.num_proposals, dim=-1).indices
		top_proposals_class = top_proposals // heatmap.shape[-1]
		top_proposals_index = top_proposals % heatmap.shape[-1]
		query_feat = lidar_feat_flatten.gather(
			dim=-1,
			index=top_proposals_index[:, None, :].expand(-1, lidar_feat_flatten.shape[1], -1),
		)
		self.query_labels = top_proposals_class

		one_hot = (
			F.one_hot(top_proposals_class, num_classes=self.num_classes).permute(0, 2, 1).float()
		)
		query_feat = query_feat + self.class_encoding(one_hot)
		query_pos = bev_pos.gather(
			dim=1,
			index=top_proposals_index[:, :, None].expand(-1, -1, bev_pos.shape[-1]),
		)

		ret_dicts: list[dict[str, torch.Tensor]] = []
		for layer_idx, (decoder, prediction_head) in enumerate(
			zip(self.decoder, self.prediction_heads)
		):
			query_feat = decoder(query_feat, lidar_feat_flatten, query_pos, bev_pos)
			res_layer = prediction_head(query_feat)
			res_layer['center'] = res_layer['center'] + query_pos.transpose(1, 2)
			ret_dicts.append(res_layer)
			if layer_idx != len(self.decoder) - 1:
				query_pos = res_layer['center'].detach().transpose(1, 2).contiguous()

		ret_dicts[0]['query_heatmap_score'] = heatmap.gather(
			dim=-1,
			index=top_proposals_index[:, None, :].expand(-1, self.num_classes, -1),
		)
		ret_dicts[0]['dense_heatmap'] = dense_heatmap

		if not self.auxiliary:
			return ret_dicts[-1]

		merged: dict[str, torch.Tensor] = {}
		for key in ret_dicts[0]:
			if key in {'dense_heatmap', 'query_heatmap_score'}:
				merged[key] = ret_dicts[0][key]
			else:
				merged[key] = torch.cat([ret[key] for ret in ret_dicts], dim=-1)
		return merged

	def _get_targets_single(
		self,
		gt_bboxes_3d: Any,
		gt_labels_3d: torch.Tensor,
		preds_dict: dict[str, torch.Tensor],
		batch_idx: int,
	):
		num_proposals = preds_dict['center'].shape[-1]
		score = preds_dict['heatmap'].detach().clone()
		center = preds_dict['center'].detach().clone()
		height = preds_dict['height'].detach().clone()
		dim = preds_dict['dim'].detach().clone()
		rot = preds_dict['rot'].detach().clone()
		vel = preds_dict.get('vel', None)
		if vel is not None:
			vel = vel.detach().clone()

		boxes_dict = self.bbox_coder.decode(score, rot, dim, center, height, vel)
		bboxes_tensor = boxes_dict[0]['bboxes']
		gt_bboxes_tensor = gt_bboxes_3d.tensor.to(score.device)
		gt_labels_3d = gt_labels_3d.to(score.device)

		num_layer = self.num_decoder_layers if self.auxiliary else 1
		assigned_gt_inds_all = []
		assigned_labels_all = []
		overlaps_all = []

		for idx_layer in range(num_layer):
			start = self.num_proposals * idx_layer
			end = self.num_proposals * (idx_layer + 1)
			layer_boxes = bboxes_tensor[start:end, :]
			layer_scores = score[..., start:end][0].T
			if isinstance(self.bbox_assigner, HungarianAssigner3D):
				assign_result = self.bbox_assigner.assign(
					layer_boxes,
					gt_bboxes_tensor,
					gt_labels_3d,
					layer_scores,
					self.train_cfg,
				)
			else:
				assign_result = self.bbox_assigner.assign(
					layer_boxes,
					gt_bboxes_tensor,
					gt_labels_3d,
					self.query_labels[batch_idx, start:end]
					if self.query_labels is not None
					else None,
				)
			assigned_gt_inds_all.append(assign_result.gt_inds)
			assigned_labels_all.append(assign_result.labels)
			overlaps_all.append(
				assign_result.max_overlaps
				if assign_result.max_overlaps is not None
				else layer_boxes.new_zeros((layer_boxes.shape[0],))
			)

		assign_result_ensemble = AssignResult(
			num_gts=gt_bboxes_tensor.size(0),
			gt_inds=torch.cat(assigned_gt_inds_all),
			max_overlaps=torch.cat(overlaps_all).clamp_(min=0.0, max=1.0),
			labels=torch.cat(assigned_labels_all),
		)
		sampling_result = self.bbox_sampler.sample(
			assign_result_ensemble, bboxes_tensor, gt_bboxes_tensor
		)
		pos_inds = sampling_result.pos_inds
		neg_inds = sampling_result.neg_inds
		ious = assign_result_ensemble.max_overlaps

		bbox_targets = torch.zeros((num_proposals, self.bbox_coder.code_size), device=score.device)
		bbox_weights = torch.zeros_like(bbox_targets)
		labels = bboxes_tensor.new_full((num_proposals,), self.num_classes, dtype=torch.long)
		label_weights = bboxes_tensor.new_zeros((num_proposals,), dtype=torch.float32)

		if pos_inds.numel() > 0:
			pos_bbox_targets = self.bbox_coder.encode(sampling_result.pos_gt_bboxes)
			bbox_targets[pos_inds, :] = pos_bbox_targets
			bbox_weights[pos_inds, :] = 1.0
			labels[pos_inds] = gt_labels_3d[sampling_result.pos_assigned_gt_inds]
			pos_weight = float(self.train_cfg.get('pos_weight', -1))
			label_weights[pos_inds] = 1.0 if pos_weight <= 0 else pos_weight

		if neg_inds.numel() > 0:
			label_weights[neg_inds] = 1.0

		gt_boxes_gravity = torch.cat(
			[gt_bboxes_3d.gravity_center, gt_bboxes_3d.tensor[:, 3:]], dim=1
		).to(score.device)
		grid_size = torch.as_tensor(self.train_cfg['grid_size'], device=score.device)
		pc_range = torch.as_tensor(self.train_cfg['point_cloud_range'], device=score.device)
		voxel_size = torch.as_tensor(self.train_cfg['voxel_size'], device=score.device)
		feature_map_size = grid_size[:2] // int(self.train_cfg['out_size_factor'])
		heatmap = gt_boxes_gravity.new_zeros(
			self.num_classes,
			int(feature_map_size[1].item()),
			int(feature_map_size[0].item()),
		)

		for idx in range(len(gt_boxes_gravity)):
			cls = int(gt_labels_3d[idx].item())
			if cls < 0 or cls >= self.num_classes:
				continue
			width = gt_boxes_gravity[idx][3] / voxel_size[0] / self.train_cfg['out_size_factor']
			length = gt_boxes_gravity[idx][4] / voxel_size[1] / self.train_cfg['out_size_factor']
			if width <= 0 or length <= 0:
				continue
			radius = gaussian_radius(
				(length, width),
				min_overlap=float(self.train_cfg.get('gaussian_overlap', 0.1)),
			)
			radius = max(int(self.train_cfg.get('min_radius', 2)), int(radius))

			x, y = gt_boxes_gravity[idx][0], gt_boxes_gravity[idx][1]
			coor_x = (x - pc_range[0]) / voxel_size[0] / self.train_cfg['out_size_factor']
			coor_y = (y - pc_range[1]) / voxel_size[1] / self.train_cfg['out_size_factor']
			center = torch.tensor([coor_x, coor_y], dtype=torch.float32, device=score.device)
			center_int = center.to(torch.int32)
			draw_heatmap_gaussian(heatmap[cls], center_int[[1, 0]], radius)

		mean_iou = ious[pos_inds].sum() / max(int(pos_inds.numel()), 1)
		return (
			labels[None],
			label_weights[None],
			bbox_targets[None],
			bbox_weights[None],
			ious[None],
			int(pos_inds.numel()),
			float(mean_iou),
			heatmap[None],
		)

	def get_targets(
		self,
		gt_bboxes_3d: list[Any],
		gt_labels_3d: list[torch.Tensor],
		preds_dict: dict[str, torch.Tensor],
	):
		list_of_pred_dict = []
		for batch_idx in range(len(gt_bboxes_3d)):
			pred_dict = {
				key: preds_dict[key][batch_idx : batch_idx + 1] for key in preds_dict.keys()
			}
			list_of_pred_dict.append(pred_dict)

		results = [
			self._get_targets_single(gt_boxes, gt_labels, pred, batch_idx)
			for batch_idx, (gt_boxes, gt_labels, pred) in enumerate(
				zip(gt_bboxes_3d, gt_labels_3d, list_of_pred_dict)
			)
		]
		labels = torch.cat([res[0] for res in results], dim=0)
		label_weights = torch.cat([res[1] for res in results], dim=0)
		bbox_targets = torch.cat([res[2] for res in results], dim=0)
		bbox_weights = torch.cat([res[3] for res in results], dim=0)
		ious = torch.cat([res[4] for res in results], dim=0)
		num_pos = sum(res[5] for res in results)
		matched_ious = float(np.mean([res[6] for res in results])) if results else 0.0
		heatmap = torch.cat([res[7] for res in results], dim=0)
		return (
			labels,
			label_weights,
			bbox_targets,
			bbox_weights,
			ious,
			num_pos,
			matched_ious,
			heatmap,
		)

	def loss(
		self,
		preds_dict: dict[str, torch.Tensor],
		gt_bboxes_3d: list[Any],
		gt_labels_3d: list[torch.Tensor],
	) -> dict[str, torch.Tensor]:
		(
			labels,
			label_weights,
			bbox_targets,
			bbox_weights,
			ious,
			num_pos,
			matched_ious,
			heatmap,
		) = self.get_targets(gt_bboxes_3d, gt_labels_3d, preds_dict)

		loss_dict: dict[str, torch.Tensor] = {}
		dense_heatmap = preds_dict.get('dense_heatmap')
		if dense_heatmap is None:
			raise KeyError('TransFusionHead.loss expects dense_heatmap in predictions.')

		loss_heatmap = self.loss_heatmap(
			clip_sigmoid(dense_heatmap),
			heatmap,
			avg_factor=max(float(heatmap.eq(1).float().sum().item()), 1.0),
		)
		loss_dict['loss_heatmap'] = loss_heatmap

		num_layers = self.num_decoder_layers if self.auxiliary else 1
		total_loss = loss_heatmap

		for idx_layer in range(num_layers):
			prefix = (
				'layer_-1'
				if idx_layer == num_layers - 1 or (idx_layer == 0 and not self.auxiliary)
				else f'layer_{idx_layer}'
			)
			start = idx_layer * self.num_proposals
			end = (idx_layer + 1) * self.num_proposals

			layer_labels = labels[..., start:end].reshape(-1)
			layer_label_weights = label_weights[..., start:end].reshape(-1)
			layer_cls_score = (
				preds_dict['heatmap'][..., start:end].permute(0, 2, 1).reshape(-1, self.num_classes)
			)
			layer_loss_cls = self.loss_cls(
				layer_cls_score,
				layer_labels,
				layer_label_weights,
				avg_factor=max(num_pos, 1),
			)

			layer_center = preds_dict['center'][..., start:end]
			layer_height = preds_dict['height'][..., start:end]
			layer_dim = preds_dict['dim'][..., start:end]
			layer_rot = preds_dict['rot'][..., start:end]
			preds = torch.cat([layer_center, layer_height, layer_dim, layer_rot], dim=1).permute(
				0, 2, 1
			)
			if 'vel' in preds_dict:
				layer_vel = preds_dict['vel'][..., start:end]
				preds = torch.cat(
					[layer_center, layer_height, layer_dim, layer_rot, layer_vel], dim=1
				).permute(0, 2, 1)

			code_weights = self.train_cfg.get('code_weights', [1.0] * self.bbox_coder.code_size)
			layer_bbox_weights = bbox_weights[:, start:end, :]
			layer_reg_weights = layer_bbox_weights * layer_bbox_weights.new_tensor(code_weights)
			layer_bbox_targets = bbox_targets[:, start:end, :]
			layer_loss_bbox = self.loss_bbox(
				preds,
				layer_bbox_targets,
				layer_reg_weights,
				avg_factor=max(num_pos, 1),
			)

			loss_dict[f'{prefix}_loss_cls'] = layer_loss_cls
			loss_dict[f'{prefix}_loss_bbox'] = layer_loss_bbox
			total_loss = total_loss + layer_loss_cls + layer_loss_bbox

			if self.loss_iou is not None and 'iou' in preds_dict:
				layer_iou = preds_dict['iou'][..., start:end].squeeze(1)
				layer_iou_target = ious[..., start:end]
				layer_iou_weight = layer_bbox_weights.max(-1).values
				layer_loss_iou = self.loss_iou(
					layer_iou,
					layer_iou_target,
					layer_iou_weight,
					avg_factor=max(num_pos, 1),
				)
				loss_dict[f'{prefix}_loss_iou'] = layer_loss_iou
				total_loss = total_loss + layer_loss_iou

		loss_dict['matched_ious'] = loss_heatmap.new_tensor(matched_ious)
		loss_dict['loss'] = total_loss
		return loss_dict


if __name__ == '__main__':
	head = TransFusionHead()
	features = torch.randn(2, 512, 128, 128)
	preds = head(features)
	print(preds['heatmap'].shape)
	decoded = head.bbox_coder.decode(
		preds['heatmap'][..., -head.num_proposals :].sigmoid(),
		preds['rot'][..., -head.num_proposals :],
		preds['dim'][..., -head.num_proposals :],
		preds['center'][..., -head.num_proposals :],
		preds['height'][..., -head.num_proposals :],
		preds.get('vel', None)[..., -head.num_proposals :]
		if preds.get('vel', None) is not None
		else None,
		filter=True,
	)
	print(decoded[0]['bboxes'].shape, decoded[0]['scores'].shape, decoded[0]['labels'].shape)
