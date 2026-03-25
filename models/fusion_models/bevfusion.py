from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from models.backbones.second import SECOND
from models.backbones.sparse_encoder import SparseEncoder
from models.backbones.swin_transformer import SwinTransformer
from models.fusers.conv import ConvFuser
from models.heads.bbox.transfusion import TransFusionHead
from models.heads.segm.vanilla import BEVSegmentationHead
from models.necks.generalized_lss import GeneralizedLSSFPN
from models.necks.second import SECONDFPN
from models.vtransforms.depth_lss import DepthLSSTransform
from ops.voxel import Voxelization

__all__ = ['BEVFusion']


class BEVFusion(nn.Module):
	"""Minimal training-oriented BEVFusion for the convfuser TransFusion config.

	This local version covers the local Hydra-style experiment configs under
	`configs/experiment/`.

	It intentionally avoids the full MMDet/MMCV builder stack. Anything outside
	this config family is treated as unsupported or placeholder behavior.
	"""

	def __init__(
		self,
		encoders: dict[str, Any] | None = None,
		fuser: dict[str, Any] | None = None,
		decoder: dict[str, Any] | None = None,
		heads: dict[str, Any] | None = None,
		image_size: tuple[int, int] = (256, 704),
		point_cloud_range: tuple[float, ...] = (-54.0, -54.0, -5.0, 54.0, 54.0, 3.0),
		voxel_size: tuple[float, float, float] = (0.075, 0.075, 0.2),
	) -> None:
		super().__init__()
		self.image_size = tuple(image_size)
		self.point_cloud_range = tuple(point_cloud_range)
		self.voxel_size = tuple(voxel_size)

		encoders_cfg = self._default_encoders()
		if encoders is not None:
			encoders_cfg = self._merge_nested(encoders_cfg, encoders)

		decoder_cfg = self._default_decoder()
		if decoder is not None:
			decoder_cfg = self._merge_nested(decoder_cfg, decoder)

		fuser_cfg = self._default_fuser()
		if fuser is not None:
			fuser_cfg = self._merge_nested(fuser_cfg, fuser)

		heads_cfg = self._default_heads()
		if heads is not None:
			heads_cfg = self._merge_nested(heads_cfg, heads)

		self.encoders = nn.ModuleDict(
			{
				'camera': nn.ModuleDict(
					{
						'backbone': self._build_camera_backbone(encoders_cfg['camera']['backbone']),
						'neck': GeneralizedLSSFPN(**dict(encoders_cfg['camera']['neck'])),
						'vtransform': DepthLSSTransform(
							**dict(encoders_cfg['camera']['vtransform'])
						),
					}
				),
				'lidar': nn.ModuleDict(
					{
						'voxelize': Voxelization(**dict(encoders_cfg['lidar']['voxelize'])),
						'backbone': SparseEncoder(**dict(encoders_cfg['lidar']['backbone'])),
					}
				),
			}
		)
		self.voxelize_reduce = True

		self.fuser = ConvFuser(**dict(fuser_cfg))
		self.decoder = nn.ModuleDict(
			{
				'backbone': SECOND(**dict(decoder_cfg['backbone'])),
				'neck': SECONDFPN(**dict(decoder_cfg['neck'])),
			}
		)
		self.heads = nn.ModuleDict({})
		if heads_cfg.get('object') is not None:
			self.heads['object'] = TransFusionHead(**dict(heads_cfg['object']))
		if heads_cfg.get('map') is not None:
			self.heads['map'] = BEVSegmentationHead(**dict(heads_cfg['map']))

	def _default_encoders(self) -> dict[str, Any]:
		return {
			'camera': {
				'backbone': {
					'pretrain_img_size': 224,
					'embed_dims': 96,
					'depths': [2, 2, 6, 2],
					'num_heads': [3, 6, 12, 24],
					'window_size': 7,
					'mlp_ratio': 4,
					'qkv_bias': True,
					'qk_scale': None,
					'drop_rate': 0.0,
					'attn_drop_rate': 0.0,
					'drop_path_rate': 0.2,
					'patch_norm': True,
					'out_indices': [1, 2, 3],
					'with_cp': False,
					'convert_weights': True,
					'pretrained': 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth',
				},
				'neck': {
					'in_channels': [192, 384, 768],
					'out_channels': 256,
					'start_level': 0,
					'num_outs': 3,
					'norm_cfg': {'type': 'BN2d', 'requires_grad': True},
					'act_cfg': {'type': 'ReLU', 'inplace': True},
					'upsample_cfg': {'mode': 'bilinear', 'align_corners': False},
				},
				'vtransform': {
					'in_channels': 256,
					'out_channels': 80,
					'image_size': self.image_size,
					'feature_size': (self.image_size[0] // 8, self.image_size[1] // 8),
					'xbound': (-54.0, 54.0, 0.3),
					'ybound': (-54.0, 54.0, 0.3),
					'zbound': (-10.0, 10.0, 20.0),
					'dbound': (1.0, 60.0, 0.5),
					'downsample': 2,
				},
			},
			'lidar': {
				'voxelize': {
					'max_num_points': 10,
					'point_cloud_range': self.point_cloud_range,
					'voxel_size': self.voxel_size,
					'max_voxels': (120000, 160000),
				},
				'backbone': {
					'in_channels': 5,
					'sparse_shape': (1440, 1440, 41),
					'output_channels': 128,
					'order': ('conv', 'norm', 'act'),
					'encoder_channels': ((16, 16, 32), (32, 32, 64), (64, 64, 128), (128, 128)),
					'encoder_paddings': ((0, 0, 1), (0, 0, 1), (0, 0, (1, 1, 0)), (0, 0)),
					'block_type': 'basicblock',
				},
			},
		}

	def _default_fuser(self) -> dict[str, Any]:
		return {
			'in_channels': (80, 256),
			'out_channels': 256,
		}

	def _default_decoder(self) -> dict[str, Any]:
		return {
			'backbone': {
				'in_channels': 256,
				'out_channels': (128, 256),
				'layer_nums': (5, 5),
				'layer_strides': (1, 2),
				'norm_cfg': {'type': 'BN', 'eps': 1e-3, 'momentum': 0.01},
				'conv_cfg': {'type': 'Conv2d', 'bias': False},
			},
			'neck': {
				'in_channels': (128, 256),
				'out_channels': (256, 256),
				'upsample_strides': (1, 2),
				'norm_cfg': {'type': 'BN', 'eps': 1e-3, 'momentum': 0.01},
				'upsample_cfg': {'type': 'deconv', 'bias': False},
				'use_conv_for_no_stride': True,
			},
		}

	def _default_heads(self) -> dict[str, Any]:
		return {
			'object': {
				'num_proposals': 200,
				'auxiliary': True,
				'in_channels': None,
				'hidden_channel': 128,
				'num_classes': 10,
				'num_decoder_layers': 1,
				'num_heads': 8,
				'nms_kernel_size': 3,
				'ffn_channel': 256,
				'dropout': 0.1,
				'bn_momentum': 0.1,
				'activation': 'relu',
				'train_cfg': {
					'dataset': 'nuScenes',
					'point_cloud_range': self.point_cloud_range,
					'grid_size': (1440, 1440, 41),
					'voxel_size': self.voxel_size,
					'out_size_factor': 8,
					'gaussian_overlap': 0.1,
					'min_radius': 2,
					'pos_weight': -1,
					'code_weights': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
					'assigner': {},
				},
				'test_cfg': {
					'dataset': 'nuScenes',
					'grid_size': (1440, 1440, 41),
					'out_size_factor': 8,
					'voxel_size': self.voxel_size[:2],
					'pc_range': self.point_cloud_range[:2],
					'nms_type': None,
				},
				'common_heads': {
					'center': (2, 2),
					'height': (1, 2),
					'dim': (3, 2),
					'rot': (2, 2),
					'vel': (2, 2),
				},
				'bbox_coder': {
					'pc_range': self.point_cloud_range[:2],
					'post_center_range': (-61.2, -61.2, -10.0, 61.2, 61.2, 10.0),
					'score_threshold': 0.0,
					'out_size_factor': 8,
					'voxel_size': self.voxel_size[:2],
					'code_size': 10,
				},
			},
		}

	def _merge_nested(self, base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
		merged = copy.deepcopy(base)
		for key, value in override.items():
			if isinstance(value, dict) and isinstance(merged.get(key), dict):
				merged[key] = self._merge_nested(merged[key], value)
			else:
				merged[key] = value
		return merged

	def _build_camera_backbone(self, backbone_cfg: dict[str, Any]) -> nn.Module:
		cfg = dict(backbone_cfg)
		return SwinTransformer(
			pretrain_img_size=cfg.get('pretrain_img_size', 224),
			embed_dims=cfg.get('embed_dims', 96),
			depths=tuple(cfg.get('depths', (2, 2, 6, 2))),
			num_heads=tuple(cfg.get('num_heads', (3, 6, 12, 24))),
			window_size=cfg.get('window_size', 7),
			mlp_ratio=cfg.get('mlp_ratio', 4),
			qkv_bias=cfg.get('qkv_bias', True),
			qk_scale=cfg.get('qk_scale', None),
			patch_norm=cfg.get('patch_norm', True),
			drop_rate=cfg.get('drop_rate', 0.0),
			attn_drop_rate=cfg.get('attn_drop_rate', 0.0),
			drop_path_rate=cfg.get('drop_path_rate', 0.2),
			out_indices=tuple(cfg.get('out_indices', (1, 2, 3))),
			with_cp=cfg.get('with_cp', False),
			pretrained=cfg.get('pretrained'),
			convert_weights=cfg.get('convert_weights', False),
		)

	@torch.no_grad()
	def voxelize(
		self, points: list[torch.Tensor], sensor: str = 'lidar'
	) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | list]:
		feats, coords, sizes = [], [], []
		for batch_idx, sample_points in enumerate(points):
			if hasattr(sample_points, 'tensor'):
				sample_points = sample_points.tensor
			voxelized = self.encoders[sensor]['voxelize'](sample_points)
			if len(voxelized) == 3:
				voxel_feats, voxel_coords, num_points = voxelized
			else:
				voxel_feats, voxel_coords = voxelized
				num_points = None
			feats.append(voxel_feats)
			coords.append(F.pad(voxel_coords, (1, 0), mode='constant', value=batch_idx))
			if num_points is not None:
				sizes.append(num_points)

		feats = torch.cat(feats, dim=0)
		coords = torch.cat(coords, dim=0)
		if sizes:
			sizes_tensor = torch.cat(sizes, dim=0)
			if self.voxelize_reduce:
				feats = feats.sum(dim=1) / sizes_tensor.to(feats.dtype).unsqueeze(1)
				feats = feats.contiguous()
			return feats, coords, sizes_tensor
		return feats, coords, sizes

	def extract_camera_features(
		self,
		img: torch.Tensor,
		points: list[torch.Tensor],
		radar: list[torch.Tensor] | None,
		camera2ego: torch.Tensor,
		lidar2ego: torch.Tensor,
		lidar2camera: torch.Tensor,
		lidar2image: torch.Tensor,
		camera_intrinsics: torch.Tensor,
		camera2lidar: torch.Tensor,
		img_aug_matrix: torch.Tensor,
		lidar_aug_matrix: torch.Tensor,
		metas: list[dict[str, Any]] | None,
	) -> torch.Tensor:
		points = [point.tensor if hasattr(point, 'tensor') else point for point in points]
		if radar is not None:
			radar = [point.tensor if hasattr(point, 'tensor') else point for point in radar]
		batch_size, num_cams, channels, height, width = img.shape
		img = img.view(batch_size * num_cams, channels, height, width)

		camera_feats = self.encoders['camera']['backbone'](img)
		camera_feats = self.encoders['camera']['neck'](camera_feats)
		if not isinstance(camera_feats, torch.Tensor):
			camera_feats = camera_feats[0]

		_, feat_channels, feat_h, feat_w = camera_feats.shape
		camera_feats = camera_feats.view(batch_size, num_cams, feat_channels, feat_h, feat_w)

		# Placeholder behavior: if radar is absent, pass lidar points through the radar slot
		# because the local DepthLSSTransform still expects that positional argument.
		radar_points = radar if radar is not None else points

		return self.encoders['camera']['vtransform'](
			camera_feats,
			points,
			radar_points,
			camera2ego,
			lidar2ego,
			lidar2camera,
			lidar2image,
			camera_intrinsics,
			camera2lidar,
			img_aug_matrix,
			lidar_aug_matrix,
			metas or [{} for _ in range(batch_size)],
		)

	def extract_lidar_features(self, points: list[torch.Tensor]) -> torch.Tensor:
		points = [point.tensor if hasattr(point, 'tensor') else point for point in points]
		voxel_features, coords, sizes = self.voxelize(points, sensor='lidar')
		del sizes
		batch_size = len(points)
		return self.encoders['lidar']['backbone'](voxel_features, coords, batch_size)

	def forward(
		self,
		img: torch.Tensor,
		points: list[torch.Tensor],
		camera2ego: torch.Tensor,
		lidar2ego: torch.Tensor,
		lidar2camera: torch.Tensor,
		lidar2image: torch.Tensor,
		camera_intrinsics: torch.Tensor,
		camera2lidar: torch.Tensor,
		img_aug_matrix: torch.Tensor,
		lidar_aug_matrix: torch.Tensor,
		metas: list[dict[str, Any]] | None = None,
		depths: torch.Tensor | None = None,
		radar: list[torch.Tensor] | None = None,
		gt_masks_bev: torch.Tensor | None = None,
		gt_bboxes_3d: list[Any] | None = None,
		gt_labels_3d: list[torch.Tensor] | None = None,
		**kwargs,
	) -> dict[str, Any]:
		del depths, kwargs
		points = [point.tensor if hasattr(point, 'tensor') else point for point in points]
		if radar is not None:
			radar = [point.tensor if hasattr(point, 'tensor') else point for point in radar]

		camera_bev = self.extract_camera_features(
			img,
			points,
			radar,
			camera2ego,
			lidar2ego,
			lidar2camera,
			lidar2image,
			camera_intrinsics,
			camera2lidar,
			img_aug_matrix,
			lidar_aug_matrix,
			metas,
		)
		lidar_bev = self.extract_lidar_features(points)
		fused_bev = self.fuser([camera_bev, lidar_bev])

		decoder_features = self.decoder['backbone'](fused_bev)
		decoder_features = self.decoder['neck'](decoder_features)[0]
		outputs: dict[str, Any] = {}
		if 'object' in self.heads:
			outputs['object'] = self.heads['object'](decoder_features, metas=metas)
		if 'map' in self.heads:
			if self.training:
				outputs['map'] = self.heads['map'](decoder_features, gt_masks_bev)
			else:
				outputs['map'] = self.heads['map'](decoder_features)

		return {
			**outputs,
			'features': {
				'camera_bev': camera_bev,
				'lidar_bev': lidar_bev,
				'fused_bev': fused_bev,
				'decoder_bev': decoder_features,
			},
			'targets': {
				'gt_bboxes_3d': gt_bboxes_3d,
				'gt_labels_3d': gt_labels_3d,
			},
		}


if __name__ == '__main__':
	if not torch.cuda.is_available():
		print(
			'BEVFusion demo expects CUDA for voxelization and BEV pooling; skipping forward on CPU-only environment.'
		)
	else:
		device = torch.device('cuda')
		model = BEVFusion().to(device).eval()

		batch_size = 2
		num_cams = 6
		image_h, image_w = model.image_size
		point_dim = model.encoders['lidar']['backbone'].in_channels

		img = torch.randn(batch_size, num_cams, 3, image_h, image_w, device=device)

		points: list[torch.Tensor] = []
		for _ in range(batch_size):
			xyz = torch.rand(6000, 3, device=device)
			xyz[:, 0] = (
				xyz[:, 0] * (model.point_cloud_range[3] - model.point_cloud_range[0])
				+ model.point_cloud_range[0]
			)
			xyz[:, 1] = (
				xyz[:, 1] * (model.point_cloud_range[4] - model.point_cloud_range[1])
				+ model.point_cloud_range[1]
			)
			xyz[:, 2] = (
				xyz[:, 2] * (model.point_cloud_range[5] - model.point_cloud_range[2])
				+ model.point_cloud_range[2]
			)
			extra = torch.randn(6000, point_dim - 3, device=device)
			points.append(torch.cat([xyz, extra], dim=1))

		eye4 = torch.eye(4, device=device)
		camera2ego = eye4.view(1, 1, 4, 4).repeat(batch_size, num_cams, 1, 1)
		lidar2ego = eye4.view(1, 4, 4).repeat(batch_size, 1, 1)
		lidar2camera = eye4.view(1, 1, 4, 4).repeat(batch_size, num_cams, 1, 1)
		lidar2image = eye4.view(1, 1, 4, 4).repeat(batch_size, num_cams, 1, 1)
		camera_intrinsics = eye4.view(1, 1, 4, 4).repeat(batch_size, num_cams, 1, 1)
		camera_intrinsics[..., 0, 0] = 400.0
		camera_intrinsics[..., 1, 1] = 400.0
		camera_intrinsics[..., 0, 2] = image_w / 2.0
		camera_intrinsics[..., 1, 2] = image_h / 2.0
		camera2lidar = eye4.view(1, 1, 4, 4).repeat(batch_size, num_cams, 1, 1)
		img_aug_matrix = eye4.view(1, 1, 4, 4).repeat(batch_size, num_cams, 1, 1)
		lidar_aug_matrix = eye4.view(1, 4, 4).repeat(batch_size, 1, 1)
		metas = [{'box_type_3d': None} for _ in range(batch_size)]

		with torch.no_grad():
			outputs = model(
				img=img,
				points=points,
				camera2ego=camera2ego,
				lidar2ego=lidar2ego,
				lidar2camera=lidar2camera,
				lidar2image=lidar2image,
				camera_intrinsics=camera_intrinsics,
				camera2lidar=camera2lidar,
				img_aug_matrix=img_aug_matrix,
				lidar_aug_matrix=lidar_aug_matrix,
				metas=metas,
			)

		print(outputs['object']['heatmap'].shape)
		print(outputs['features']['camera_bev'].shape)
		print(outputs['features']['lidar_bev'].shape)
		print(outputs['features']['fused_bev'].shape)
		print(outputs['features']['decoder_bev'].shape)
